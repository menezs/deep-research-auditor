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
    status: ReferenceStatus


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

    total_chunks: int = 0

    pct_supported: float
    pct_unsupported: float
    pct_contradicted: float
    count_supported: int = 0
    count_unsupported: int = 0
    count_contradicted: int = 0
    count_skipped: int = 0

    dead_references: list[Reference] = Field(default_factory=list)
    inaccessible_references: list[Reference] = Field(default_factory=list)
    skipped_chunks: list[SkippedChunk] = Field(default_factory=list)
    reference_stats: list[ReferenceStats] = Field(default_factory=list)

    total_cost_usd: float = 0.0
    total_tokens: int = 0
    cost_tracked: bool = True
    """False quando o provider/modelo do juiz nao tem preco tabelado neste
    framework (ex: `openai`) — nesse caso `total_cost_usd` e um piso, nao o
    custo real, e o relatorio sinaliza isso explicitamente. Ver
    `common.pricing.is_cost_tracked`."""
    processing_time_seconds: float = 0.0
