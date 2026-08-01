from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

from .config import Settings

_CONFIGURED = False

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
    de configuracao, compartilhado por todos os estagios."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    if settings.log_format == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )

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
