from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone

from ..models import (
    AnswerChunk,
    AuditResult,
    AuditVerdict,
    Reference,
    ReferenceStats,
    ReferenceStatus,
    Report,
    SkippedChunk,
    ToolStats,
)
from .cost_tracker import summarize_cost

_VERDICT_PCT_FIELDS: dict[AuditVerdict, str] = {
    AuditVerdict.SUPPORTED: "pct_supported",
    AuditVerdict.UNSUPPORTED: "pct_unsupported",
    AuditVerdict.CONTRADICTED: "pct_contradicted",
}

_VERDICT_COUNT_FIELDS: dict[AuditVerdict, str] = {
    AuditVerdict.SUPPORTED: "count_supported",
    AuditVerdict.UNSUPPORTED: "count_unsupported",
    AuditVerdict.CONTRADICTED: "count_contradicted",
}


def _verdict_percentages(results: list[AuditResult], total: int | None = None) -> dict[str, float]:
    """Percentual de cada veredito. `total` e o denominador — por padrao
    `len(results)` (ex: `ToolStats`, que nao tem nocao de chunks pulados);
    `aggregate_report` passa `len(chunks)` explicitamente para que
    SUPPORTED/UNSUPPORTED/CONTRADICTED/SKIPPED somem 100% de fato quando
    existem chunks pulados (nao julgados)."""
    total = len(results) if total is None else total
    if not total:
        return {field_name: 0.0 for field_name in _VERDICT_PCT_FIELDS.values()}
    counts = Counter(r.verdict for r in results)
    return {
        field_name: (counts.get(verdict, 0) / total) * 100.0
        for verdict, field_name in _VERDICT_PCT_FIELDS.items()
    }


def _verdict_counts(results: list[AuditResult]) -> dict[str, int]:
    counts = Counter(r.verdict for r in results)
    return {field_name: counts.get(verdict, 0) for verdict, field_name in _VERDICT_COUNT_FIELDS.items()}


def build_reference_stats(
    chunks: list[AnswerChunk], references: list[Reference], results: list[AuditResult]
) -> list[ReferenceStats]:
    """Estatisticas por referencia: quantas vezes foi citada e como os
    chunks que a citam foram julgados. Junta `AnswerChunk.cited_reference_ids`
    (a citacao real, ja resolvida na Fase 2) com o `AuditResult` do juiz
    (Fase 3) via `answer_chunk_id` — no audit_with_llm original essa
    correspondencia era feita hoje so manualmente, lendo o `.md` a mao."""
    result_by_chunk = {r.answer_chunk_id: r for r in results}
    reference_by_id = {ref.id: ref for ref in references}

    times_cited: dict[str, int] = defaultdict(int)
    verdict_counts: dict[str, Counter] = defaultdict(Counter)

    for chunk in chunks:
        result = result_by_chunk.get(chunk.id)
        for ref_id in chunk.cited_reference_ids:
            times_cited[ref_id] += 1
            if result is not None:
                verdict_counts[ref_id][result.verdict] += 1

    stats = [
        ReferenceStats(
            reference_id=ref_id,
            url=reference_by_id[ref_id].raw_url,
            times_cited=count,
            supported_count=verdict_counts[ref_id].get(AuditVerdict.SUPPORTED, 0),
            unsupported_count=verdict_counts[ref_id].get(AuditVerdict.UNSUPPORTED, 0),
            contradicted_count=verdict_counts[ref_id].get(AuditVerdict.CONTRADICTED, 0),
            status=reference_by_id[ref_id].status,
        )
        for ref_id, count in times_cited.items()
        if ref_id in reference_by_id
    ]
    stats.sort(key=lambda s: s.times_cited, reverse=True)
    return stats


def aggregate_report(
    *,
    run_id: str,
    answer_id: str,
    tool_name: str,
    chunks: list[AnswerChunk],
    references: list[Reference],
    results: list[AuditResult],
    skipped: list[SkippedChunk] | None = None,
    processing_time_seconds: float = 0.0,
    generated_at: datetime | None = None,
) -> Report:
    """Monta o `Report` final de uma execucao — a peca que hoje nao existe
    em codigo algum nos tres repositorios originais; todo relatorio rico
    (percentual por referencia, referencias mortas, custo total) e escrito
    manualmente a partir do JSON bruto no audit_with_llm."""
    percentages = _verdict_percentages(results, total=len(chunks))
    counts = _verdict_counts(results)
    cost = summarize_cost(results)
    skipped = skipped or []

    return Report(
        run_id=run_id,
        answer_id=answer_id,
        tool_name=tool_name,
        generated_at=generated_at or datetime.now(timezone.utc),
        total_chunks=len(chunks),
        pct_supported=percentages["pct_supported"],
        pct_unsupported=percentages["pct_unsupported"],
        pct_contradicted=percentages["pct_contradicted"],
        count_supported=counts["count_supported"],
        count_unsupported=counts["count_unsupported"],
        count_contradicted=counts["count_contradicted"],
        count_skipped=len(skipped),
        dead_references=[r for r in references if r.status == ReferenceStatus.DEAD],
        inaccessible_references=[r for r in references if r.status == ReferenceStatus.INACCESSIBLE],
        skipped_chunks=skipped,
        reference_stats=build_reference_stats(chunks, references, results),
        total_cost_usd=cost.total_cost_usd,
        total_tokens=cost.total_tokens,
        processing_time_seconds=processing_time_seconds,
    )


def aggregate_tool_stats(tool_name: str, results: list[AuditResult]) -> ToolStats:
    """Estatisticas de uma unica ferramenta (ChatGPT/Gemini/Perplexity),
    para permitir comparar varias execucoes lado a lado (`render.py` monta
    a tabela comparativa a partir de uma lista de `ToolStats`)."""
    percentages = _verdict_percentages(results)
    return ToolStats(
        tool_name=tool_name,
        pct_supported=percentages["pct_supported"],
        pct_unsupported=percentages["pct_unsupported"],
        pct_contradicted=percentages["pct_contradicted"],
        total_chunks=len(results),
    )
