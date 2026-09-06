from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from .audit_result import SkippedChunk
from .reference import Reference, ReferenceStatus


class ReferenceStats(BaseModel):
    reference_id: str
    url: str
    times_cited: int
    supported_count: int = 0
    unsupported_count: int = 0
    contradicted_count: int = 0
    supporting_citations: int = 0
    """Quantos dos chunks que citam esta referencia tiveram nela evidencia
    de suporte pleno (`ReferenceVerdict.relation == "supports"`). Se e
    `< times_cited`, a referencia foi citada mais vezes do que sustentou.
    Igual a `len(supports_positions)`."""
    partial_citations: int = 0
    """Chunks em que esta referencia sustenta so PARTE da afirmacao
    (`relation == "partial"`). Igual a `len(partial_positions)`."""
    status: ReferenceStatus

    # --- posicoes dos trechos (AnswerChunk.position) por relacao DA FONTE,
    # base das tabelas "Analise por Referencia" e "Verificacao por Fonte" ---
    supports_positions: list[int] = Field(default_factory=list)
    partial_positions: list[int] = Field(default_factory=list)
    absent_positions: list[int] = Field(default_factory=list)
    """Citada, mas o juiz classificou a relacao da fonte como `absent` — a
    fonte nao trata a afirmacao."""
    contradicts_positions: list[int] = Field(default_factory=list)
    unrated_positions: list[int] = Field(default_factory=list)
    """Citada, mas o juiz nao detalhou a relacao desta fonte especifica
    (sem `per_reference` nem credito em `supporting_reference_ids`)."""
    key_excerpt: str = ""
    """Um trecho literal da fonte, representativo do que ela sustenta —
    preferencialmente vindo de um veredito `supports`."""
    key_excerpt_position: int | None = None
    """Posicao (`AnswerChunk.position`) do trecho da resposta para o qual
    `key_excerpt` foi apresentado como evidencia."""


class PotentiallyUnsourcedChunk(BaseModel):
    """Chunk com varias frases em que so a ultima esta ancorada por uma
    citacao — as frases anteriores podem carregar afirmacoes sem fonte."""

    answer_chunk_id: str
    sentence_count: int
    cited_reference_ids: list[str] = Field(default_factory=list)
    excerpt: str


class UncitedClaimChunk(BaseModel):
    """Paragrafo que faz afirmacao factual e nao tem nenhuma citacao no
    documento (`AnswerChunk.is_uncited_claim`)."""

    answer_chunk_id: str
    excerpt: str


class SourceInfo(BaseModel):
    """Metadados forenses do arquivo de resposta auditado — extraidos no
    estagio de extracao, sem LLM. Campos ausentes viram `None`."""

    format: str
    size_bytes: int
    page_count: int | None = None
    pdf_title: str | None = None
    pdf_author: str | None = None
    pdf_creator: str | None = None
    pdf_producer: str | None = None
    created: str | None = None
    modified: str | None = None
    encrypted: bool = False
    browser_print: bool = False
    """PDF gerado por impressao de navegador (Skia/PDF + Chromium) —
    tipico de captura de resposta de Deep Research; sem autoria confiavel."""
    citation_markers_total: int = 0
    """Ocorrencias de `[N]` no corpo (o mesmo numero pode aparecer varias vezes)."""
    citation_markers_distinct: int = 0
    references_listed: int = 0
    references_never_cited: int = 0


class ToolStats(BaseModel):
    tool_name: str
    pct_supported: float
    pct_unsupported: float
    pct_contradicted: float
    total_chunks: int


class JudgeConfig(BaseModel):
    """Modelo e parametros do LLM-juiz usados nesta execucao — persistido
    no `Report` para permitir consulta posterior (ex: comparar runs feitas
    com juizes diferentes) sem depender de reconstituir o `Settings`
    original, que pode ter mudado desde entao."""

    provider: str
    model: str
    temperature: float
    max_retries: int
    retry_delay: float
    base_url: str | None = None


class Report(BaseModel):
    run_id: str
    answer_id: str
    tool_name: str
    answer_path: str | None = None
    generated_at: datetime

    judge_config: JudgeConfig | None = None
    source_info: SourceInfo | None = None

    total_chunks: int = 0

    pct_supported: float
    pct_unsupported: float
    pct_contradicted: float
    count_supported: int = 0
    count_unsupported: int = 0
    count_contradicted: int = 0
    count_skipped: int = 0
    count_partially_supported: int = 0
    """SUPPORTED com aspectos da afirmacao nao cobertos pela evidencia
    (`AuditResult.unsupported_aspects` nao vazio)."""
    skipped_reason_counts: dict[str, int] = Field(default_factory=dict)
    """Chunks SKIPPED agrupados por motivo (`sem_citacao` / `ref_sem_conteudo`)."""
    count_uncited_claims: int = 0
    """Trechos com afirmacao factual e nenhuma citacao no documento."""
    count_claim_chunks: int = 0
    """Trechos que carregam afirmacao (citados + sem citacao) — denominador
    de `count_uncited_claims`."""

    # --- Cascata de verificacao de vereditos UNSUPPORTED ---
    verification_ran: bool = False
    count_unsupported_confirmed: int = 0
    """UNSUPPORTED que sobreviveram a varredura do documento citado inteiro."""
    count_reclassified_by_verification: int = 0
    """Chunks cujo veredito inicial era UNSUPPORTED e a cascata reclassificou
    para SUPPORTED/CONTRADICTED."""
    verification_stage_counts: dict[str, int] = Field(default_factory=dict)
    """Quantos vereditos finais vieram de cada etapa (`context_expansion`,
    `full_doc_scan`, `cross_reference`)."""
    mis_cited_reference_count: int = 0
    """UNSUPPORTED corroborados por outra referencia baixada (provavel erro
    de citacao)."""

    uncredited_reference_count: int = 0
    """Referencias citadas por pelo menos um chunk, mas que nunca
    apareceram como fonte de suporte em nenhum veredito."""

    dead_references: list[Reference] = Field(default_factory=list)
    inaccessible_references: list[Reference] = Field(default_factory=list)
    skipped_chunks: list[SkippedChunk] = Field(default_factory=list)
    reference_stats: list[ReferenceStats] = Field(default_factory=list)
    potentially_unsourced_chunks: list[PotentiallyUnsourcedChunk] = Field(default_factory=list)
    uncited_claims: list[UncitedClaimChunk] = Field(default_factory=list)

    total_cost_usd: float = 0.0
    total_tokens: int = 0
    cost_tracked: bool = True
    """False quando o provider/modelo do juiz nao tem preco tabelado neste
    framework (ex: `openai`) — nesse caso `total_cost_usd` e um piso, nao o
    custo real, e o relatorio sinaliza isso explicitamente. Ver
    `common.pricing.is_cost_tracked`."""
    processing_time_seconds: float = 0.0
