from __future__ import annotations

import json
import logging
import re
import sys
import textwrap
import threading
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from .config import Settings

_CONFIGURED = False
_PRETTY = False

RUN_LOG_FILENAME = "log.txt"
RUN_LOG_RECORDS_FILENAME = "log_registros.txt"
_RUN_LOG_WIDTH = 120
_RUN_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_RUN_LOG_HANDLERS: list[logging.Handler] = []
_ORIGINAL_STREAMS: tuple | None = None

# Logger que escreve SO no `log.txt` da run (nao propaga para o root):
# leva para o arquivo o que o terminal mostra sem passar pelo logging —
# banners de estagio do Rich, `typer.echo` da CLI, traceback de falha.
_terminal_logger = logging.getLogger("auditframework.terminal")
_terminal_logger.propagate = False

console = Console()

_NOISY_THIRD_PARTY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "sentence_transformers",
    "faiss",
    "playwright",
    "openai",
    "anthropic",
)

STAGE_COLORS: dict[str, str] = {
    "extraction": "cyan",
    "ingestion": "yellow",
    "indexing": "magenta",
    "judging": "blue",
    "reporting": "green",
}
_DEFAULT_STAGE_COLOR = "white"


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(settings: Settings) -> None:
    """Configura o logger raiz uma unica vez para todo o pipeline.

    Substitui as tres configuracoes de logging distintas hoje existentes
    (CorpusForge, audit_with_llm, syntex-sem-logging) por um unico ponto
    de configuracao, compartilhado por todos os estagios.

    Em `log_format="text"` usa `RichHandler` (cores por nivel, tracebacks
    legiveis) em vez do `StreamHandler` plano anterior — `log_format="json"`
    fica intocado, pois e o formato usado por consumo de maquina/agregador
    de log, onde decoracao visual so atrapalharia o parsing."""
    global _CONFIGURED, _PRETTY
    if _CONFIGURED:
        return

    _PRETTY = settings.log_format == "text"

    if _PRETTY:
        handler: logging.Handler = RichHandler(
            console=console,
            show_path=False,
            markup=False,
            rich_tracebacks=True,
        )
        handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())

    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())
    root.handlers.clear()
    root.addHandler(handler)

    if settings.log_level.upper() != "DEBUG":
        for name in _NOISY_THIRD_PARTY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_LINE_PARTS_RE = re.compile(r"(\r\n|\r|\n)")


class _WrappedFormatter(logging.Formatter):
    """Formato de leitura do `log.txt`: quebra em `_RUN_LOG_WIDTH` colunas e
    recua as continuacoes ate o inicio da mensagem (depois da data e do
    nivel), como o Rich faz na tela. Quebras de linha da propria mensagem
    (JSON do juiz, traceback) viram linhas recuadas. Palavras maiores que a
    largura (URLs, caminhos) nao sao partidas: a linha passa da largura."""

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        indent = " " * (len(record.asctime) + 10)  # data + espaco + nivel (8) + espaco
        out: list[str] = []
        for index, line in enumerate(text.split("\n")):
            wrapped = textwrap.wrap(
                line,
                width=_RUN_LOG_WIDTH,
                initial_indent=indent if index else "",
                subsequent_indent=indent,
                break_long_words=False,
                break_on_hyphens=False,
            )
            out.extend(wrapped or [""])
        return "\n".join(out)


class _OneLineFormatter(logging.Formatter):
    """Formato do `log_registros.txt`: um registro por linha, com as quebras
    de linha da mensagem escapadas (`\\n`), para busca com `grep` e
    leitura por script."""

    def format(self, record: logging.LogRecord) -> str:
        return super().format(record).replace("\r", "\\r").replace("\n", "\\n")


class _TerminalMirror:
    """Envolve `sys.stdout`/`sys.stderr`: repassa tudo ao terminal e grava
    no `log.txt` cada linha completa, sem codigos de cor. Uma linha
    reescrita com `\\r` (barra de progresso do tqdm) entra uma vez, no
    estado em que terminou — ou no ultimo, se a run foi interrompida."""

    def __init__(self, stream):
        self._stream = stream
        self._line = ""
        self._lock = threading.Lock()
        self._writing = threading.local()

    def write(self, text: str) -> int:
        written = self._stream.write(text)
        if getattr(self._writing, "active", False):
            return written
        self._writing.active = True
        try:
            with self._lock:
                for part in _LINE_PARTS_RE.split(_ANSI_RE.sub("", text)):
                    if part in ("\n", "\r\n"):
                        self._emit()
                    elif part == "\r":
                        self._line = ""
                    else:
                        self._line += part
        finally:
            self._writing.active = False
        return written

    def _emit(self) -> None:
        line, self._line = self._line.rstrip(), ""
        if line.strip():
            _terminal_logger.info(line)

    def close_mirror(self) -> None:
        with self._lock:
            self._emit()

    def flush(self) -> None:
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def attach_run_log(run_dir: Path) -> Path:
    """Passa a gravar tambem em `run_dir/log.txt` tudo o que aparece no
    terminal durante a run, em texto puro (sem cores) e com data/hora por
    linha: os registros de log (com a mensagem inteira, sem a quebra de
    linha que o Rich aplica na tela), avisos do modulo `warnings` e tudo o
    que bibliotecas escrevem em stdout/stderr, inclusive barras de
    progresso (cada uma no estado final, nao a cada atualizacao).

    Grava dois arquivos com o mesmo conteudo: `log.txt`, para leitura, com
    as linhas longas quebradas e recuadas; e `log_registros.txt`, com um
    registro por linha, para busca e script.

    Abre em modo de acrescimo: um `audit resume` continua o log da mesma
    run em vez de sobrescreve-lo. Saida escrita direto no descritor de
    arquivo por codigo C ou por subprocessos nao passa por aqui."""
    global _ORIGINAL_STREAMS
    detach_run_log()
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / RUN_LOG_FILENAME
    for name, formatter in (
        (RUN_LOG_FILENAME, _WrappedFormatter(_RUN_LOG_FORMAT)),
        (RUN_LOG_RECORDS_FILENAME, _OneLineFormatter(_RUN_LOG_FORMAT)),
    ):
        handler = logging.FileHandler(run_dir / name, mode="a", encoding="utf-8")
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)
        _terminal_logger.addHandler(handler)
        _RUN_LOG_HANDLERS.append(handler)

    # O console do Rich (RichHandler, banners) passa a escrever direto no
    # terminal original: esses registros ja chegam ao arquivo pelo handler
    # acima, inteiros — pelo espelho entrariam duplicados e quebrados.
    _ORIGINAL_STREAMS = (sys.stdout, sys.stderr, console._file)
    console.file = sys.stdout
    sys.stdout = _TerminalMirror(sys.stdout)
    sys.stderr = _TerminalMirror(sys.stderr)
    logging.captureWarnings(True)
    return path


def detach_run_log() -> None:
    """Para de gravar no `log.txt`, devolve stdout/stderr originais e fecha
    o arquivo."""
    global _ORIGINAL_STREAMS
    if not _RUN_LOG_HANDLERS:
        return
    logging.captureWarnings(False)
    if _ORIGINAL_STREAMS is not None:
        out, err, console_file = _ORIGINAL_STREAMS
        for mirror in (sys.stdout, sys.stderr):
            if isinstance(mirror, _TerminalMirror):
                mirror.close_mirror()
        if isinstance(sys.stdout, _TerminalMirror):
            sys.stdout = out
        if isinstance(sys.stderr, _TerminalMirror):
            sys.stderr = err
        console._file = console_file
        _ORIGINAL_STREAMS = None
    for handler in _RUN_LOG_HANDLERS:
        logging.getLogger().removeHandler(handler)
        _terminal_logger.removeHandler(handler)
        handler.close()
    _RUN_LOG_HANDLERS.clear()


def record_terminal_output(message: str) -> None:
    """Grava no `log.txt` uma linha que o terminal mostrou por fora do
    logging. Sem log de run anexado, nao faz nada."""
    if _RUN_LOG_HANDLERS:
        _terminal_logger.info(message)


def record_exception(message: str) -> None:
    """Grava no `log.txt` a excecao em tratamento, com traceback. Nao
    repete no terminal, onde o traceback ja aparece."""
    if _RUN_LOG_HANDLERS:
        _terminal_logger.exception(message)


def _stage_color(name: str) -> str:
    return STAGE_COLORS.get(name, _DEFAULT_STAGE_COLOR)


def stage_banner(name: str, index: int, total: int) -> None:
    """Marca visualmente o inicio de um estagio do pipeline — uma regua
    colorida (modo pretty) ou uma linha de log equivalente (modo json/
    antes de `setup_logging` rodar), para que o output de uma run longa
    fique facil de escanear por estagio."""
    if _PRETTY:
        color = _stage_color(name)
        console.rule(f"[bold {color}]Estágio {index}/{total} · {name}[/bold {color}]", style=color)
        record_terminal_output(f"Estágio {index}/{total} · {name}")
    else:
        get_logger(__name__).info("Executando estagio %d/%d: %s", index, total, name)


def stage_done(name: str, elapsed_seconds: float) -> None:
    """Marca a conclusao de um estagio, com o tempo gasto."""
    if _PRETTY:
        color = _stage_color(name)
        console.print(f"[{color}]✓ {name} concluído em {elapsed_seconds:.1f}s[/{color}]")
        record_terminal_output(f"✓ {name} concluído em {elapsed_seconds:.1f}s")
    else:
        get_logger(__name__).info("Estagio %s concluido em %.1fs", name, elapsed_seconds)


def stage_skipped(name: str) -> None:
    """Marca um estagio pulado por ja ter sido concluido (`audit resume`)."""
    if _PRETTY:
        console.print(f"[dim]⏭ Pulando estágio já concluído: {name}[/dim]")
        record_terminal_output(f"⏭ Pulando estágio já concluído: {name}")
    else:
        get_logger(__name__).info("Pulando estagio ja concluido: %s", name)
