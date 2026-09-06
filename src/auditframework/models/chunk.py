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
    sentence_count: int = 1
    """Frases no trecho. `> 1` indica que so a ultima frase esta ancorada
    pela citacao — as frases anteriores podem carregar afirmacoes sem
    fonte associada (ver secao "Afirmacoes possivelmente sem fonte" no
    relatorio)."""
    is_uncited_claim: bool = False
    """Paragrafo que faz afirmacao factual mas nao tem nenhuma citacao —
    separado de um chunk citado por fronteira de paragrafo (o `AnswerChunker`
    ancora o marcador `[N]` so no paragrafo em que ele aparece). Nao e
    julgado (vira SKIPPED), mas e listado em "Afirmacoes sem Citacao"."""


class ReferenceChunk(BaseModel):
    """Um trecho semanticamente coerente de um Document indexado."""

    id: str
    reference_id: str
    section: str | None = None
    text: str
    token_count: int
    embedding_id: int
