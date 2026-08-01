from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class ReferenceStatus(str, Enum):
    PENDING = "pending"
    DOWNLOADED = "downloaded"
    DEAD = "dead"  # 404
    INACCESSIBLE = "inaccessible"  # 403/timeout/SSL apos esgotar retries
    ERROR = "error"


class Reference(BaseModel):
    """Uma referencia citada por uma resposta de Deep Research.

    `id` e derivado deterministicamente da URL normalizada (hash), nao de
    uma posicao sequencial de extracao — isso corrige o bug do CorpusForge
    em que a mesma URL podia receber ids diferentes entre execucoes por
    depender da ordem, nao-deterministica, de extracao via LLM."""

    id: str
    citation_markers: list[str] = Field(default_factory=list)
    raw_url: str
    normalized_url: str
    title: str | None = None
    status: ReferenceStatus = ReferenceStatus.PENDING
    http_status: int | None = None
    fetched_at: datetime | None = None
    error_message: str | None = None
    source_answer_id: str
    tool_name: str

    @staticmethod
    def id_for_url(normalized_url: str) -> str:
        return hashlib.sha256(normalized_url.encode("utf-8")).hexdigest()[:16]
