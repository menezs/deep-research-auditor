from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

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


class Report(BaseModel):
    run_id: str
    answer_id: str
    tool_name: str
    generated_at: datetime

    total_chunks: int = 0

    pct_supported: float
    pct_unsupported: float
    pct_contradicted: float

    dead_references: list[Reference] = Field(default_factory=list)
    inaccessible_references: list[Reference] = Field(default_factory=list)
    reference_stats: list[ReferenceStats] = Field(default_factory=list)

    total_cost_usd: float = 0.0
    total_tokens: int = 0
    processing_time_seconds: float = 0.0
