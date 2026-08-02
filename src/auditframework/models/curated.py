from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievedPassage(BaseModel):
    """Um trecho recuperado da referencia citada, com proveniencia."""

    reference_chunk_id: str
    reference_id: str
    score: float
    text: str


class CuratedDocument(BaseModel):
    """Contexto curado para o juiz LLM, montado a partir dos trechos
    recuperados para um unico AnswerChunk."""

    answer_chunk_id: str
    passages: list[RetrievedPassage] = Field(default_factory=list)
    assembled_context: str
    skip_reason: str | None = None
    """Nao-None quando o chunk nao deve ser submetido ao juiz LLM: no modo
    de recuperacao escopado por citacao (padrao), a(s) referencia(s)
    citada(s) pelo chunk nao tem conteudo indexado (nao baixada(s)/
    inacessivel(is)) ou o chunk nao cita nenhuma referencia."""
