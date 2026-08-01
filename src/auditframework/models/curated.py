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
    retrieval_degraded: bool = False
    """True quando a(s) referencia(s) citada(s) pelo chunk estavam mortas ou
    inacessiveis e a recuperacao precisou degradar (ex: buscar no corpus
    inteiro em vez de escopar pela referencia citada)."""
