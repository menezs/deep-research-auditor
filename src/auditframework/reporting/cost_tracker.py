from __future__ import annotations

from dataclasses import dataclass

from ..models import AuditResult


@dataclass
class CostSummary:
    total_cost_usd: float
    total_prompt_tokens: int
    total_completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens


def summarize_cost(results: list[AuditResult]) -> CostSummary:
    """Agrega custo/tokens de todos os resultados de uma execucao.

    Substitui a pratica atual (audit_with_llm) de descartar o objeto de
    usage retornado pelo LLM logo apos ler o texto da resposta — aqui o
    `AuditResult.cost_usd`/`prompt_tokens`/`completion_tokens` (Fase 3) ja
    chega populado por resultado, entao a agregacao e uma soma direta."""
    return CostSummary(
        total_cost_usd=sum(r.cost_usd for r in results),
        total_prompt_tokens=sum(r.prompt_tokens for r in results),
        total_completion_tokens=sum(r.completion_tokens for r in results),
    )
