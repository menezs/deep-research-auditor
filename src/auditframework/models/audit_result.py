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


class VerificationStep(BaseModel):
    """Um passo da cascata de verificacao aplicada a um veredito
    `UNSUPPORTED` inicial (ver `judging/verification/`)."""

    stage: str  # context_expansion | full_doc_scan | cross_reference
    verdict: AuditVerdict
    note: str


class AuditResult(BaseModel):
    """Veredito do juiz LLM para um unico AnswerChunk."""

    answer_chunk_id: str
    verdict: AuditVerdict
    justification: str
    cited_excerpts: list[str] = Field(default_factory=list)
    supporting_reference_ids: list[str] = Field(default_factory=list)
    """Subconjunto de `AnswerChunk.cited_reference_ids` cujo conteudo de
    fato sustenta a afirmacao, segundo o juiz. Uma referencia citada que
    nunca aparece aqui foi citada mas nao contribuiu com evidencia."""
    unsupported_aspects: list[str] = Field(default_factory=list)
    """Partes da afirmacao nao cobertas pela evidencia — preenchido mesmo
    sob veredito `supported` (sinal de "parcialmente suportada")."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    judge_model: str

    # --- Verificacao de vereditos UNSUPPORTED (cascata A -> B -> C) ---
    verification_stage: str = "baseline"
    """Etapa que produziu o veredito final: `baseline` (juiz inicial) ou
    `context_expansion` / `full_doc_scan` / `cross_reference`."""
    unsupported_confirmed: bool = False
    """True quando o veredito continuou UNSUPPORTED mesmo apos a varredura
    do documento citado inteiro (Etapa B) — o claim comprovadamente nao
    esta na referencia citada."""
    corroborated_by_other_reference: list[str] = Field(default_factory=list)
    """Referencias (nao citadas pelo chunk) que sustentam o claim,
    detectadas pela checagem no corpus inteiro (Etapa C) — sinal de
    citacao trocada, nao de alegacao infundada."""
    contradicted_by_other_reference: list[str] = Field(default_factory=list)
    verification_trail: list[VerificationStep] = Field(default_factory=list)


class SkippedChunk(BaseModel):
    """Chunk que nao foi submetido ao juiz LLM porque nao havia evidencia
    citada disponivel no modo de recuperacao atual (ver
    `Retriever.retrieve` / `CuratedDocument.skip_reason`). Nao e um
    veredito de conteudo — fica de fora de AuditVerdict de proposito."""

    answer_chunk_id: str
    reason: str
