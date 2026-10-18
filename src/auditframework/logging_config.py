from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from .config import Settings

_CONFIGURED = False
_PRETTY = False

RUN_LOG_FILENAME = "log.txt"
_RUN_LOG_HANDLER: logging.Handler | None = None

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


def attach_run_log(run_dir: Path) -> Path:
    """Passa a gravar tambem em `run_dir/log.txt` tudo o que o pipeline
    loga no terminal, em texto puro (sem cores) e com data/hora por linha.

    Abre em modo de acrescimo: um `audit resume` continua o log da mesma
    run em vez de sobrescreve-lo. Barras de progresso do `tqdm` nao entram
    — escrevem direto no stderr e virariam centenas de linhas repetidas."""
    global _RUN_LOG_HANDLER
    detach_run_log()
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / RUN_LOG_FILENAME
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    _terminal_logger.addHandler(handler)
    _RUN_LOG_HANDLER = handler
    return path


def detach_run_log() -> None:
    """Para de gravar no `log.txt` e fecha o arquivo."""
    global _RUN_LOG_HANDLER
    if _RUN_LOG_HANDLER is None:
        return
    logging.getLogger().removeHandler(_RUN_LOG_HANDLER)
    _terminal_logger.removeHandler(_RUN_LOG_HANDLER)
    _RUN_LOG_HANDLER.close()
    _RUN_LOG_HANDLER = None


def record_terminal_output(message: str) -> None:
    """Grava no `log.txt` uma linha que o terminal mostrou por fora do
    logging. Sem log de run anexado, nao faz nada."""
    if _RUN_LOG_HANDLER is not None:
        _terminal_logger.info(message)


def record_exception(message: str) -> None:
    """Grava no `log.txt` a excecao em tratamento, com traceback. Nao
    repete no terminal, onde o traceback ja aparece."""
    if _RUN_LOG_HANDLER is not None:
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
