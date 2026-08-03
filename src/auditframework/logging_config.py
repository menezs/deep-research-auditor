from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

from rich.console import Console
from rich.logging import RichHandler

from .config import Settings

_CONFIGURED = False
_PRETTY = False

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
    else:
        get_logger(__name__).info("Executando estagio %d/%d: %s", index, total, name)


def stage_done(name: str, elapsed_seconds: float) -> None:
    """Marca a conclusao de um estagio, com o tempo gasto."""
    if _PRETTY:
        color = _stage_color(name)
        console.print(f"[{color}]✓ {name} concluído em {elapsed_seconds:.1f}s[/{color}]")
    else:
        get_logger(__name__).info("Estagio %s concluido em %.1fs", name, elapsed_seconds)


def stage_skipped(name: str) -> None:
    """Marca um estagio pulado por ja ter sido concluido (`audit resume`)."""
    if _PRETTY:
        console.print(f"[dim]⏭ Pulando estágio já concluído: {name}[/dim]")
    else:
        get_logger(__name__).info("Pulando estagio ja concluido: %s", name)
