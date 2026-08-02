from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class AuditVerdict(str, Enum):
    """Apenas 3 categorias de classificacao de conteudo. Falhas tecnicas
    (saida do LLM juiz nao parseavel) nao viram uma 4a categoria — a
    excecao propaga e o chunk fica pendente para `audit resume`, em vez de
    ser coagida silenciosamente para um veredito de conteudo."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"


class AuditResult(BaseModel):
    """Veredito do juiz LLM para um unico AnswerChunk."""

    answer_chunk_id: str
    verdict: AuditVerdict
    justification: str
    cited_excerpts: list[str] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    judge_model: str


class SkippedChunk(BaseModel):
    """Chunk que nao foi submetido ao juiz LLM porque nao havia evidencia
    citada disponivel no modo de recuperacao atual (ver
    `Retriever.retrieve` / `CuratedDocument.skip_reason`). Nao e um
    veredito de conteudo — fica de fora de AuditVerdict de proposito."""

    answer_chunk_id: str
    reason: str
