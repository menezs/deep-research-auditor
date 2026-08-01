from __future__ import annotations

from pydantic import BaseModel, Field


class AnswerChunk(BaseModel):
    """Um trecho da resposta de Deep Research, delimitado por marcadores
    de citacao (ex: "[1]", "[2]")."""

    id: str
    answer_id: str
    position: int
    text: str
    cited_reference_ids: list[str] = Field(default_factory=list)


class ReferenceChunk(BaseModel):
    """Um trecho semanticamente coerente de um Document indexado."""

    id: str
    reference_id: str
    section: str | None = None
    text: str
    token_count: int
    embedding_id: int
