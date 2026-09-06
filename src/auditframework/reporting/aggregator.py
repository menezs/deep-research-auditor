from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone

from ..models import (
    AnswerChunk,
    AuditResult,
    AuditVerdict,
    JudgeConfig,
    PotentiallyUnsourcedChunk,
    Reference,
    SourceInfo,
    UncitedClaimChunk,
    ReferenceStats,
    ReferenceStatus,
    Report,
    SkippedChunk,
    ToolStats,
)
from ..common.pricing import is_cost_tracked
from .cost_tracker import summarize_cost

_MULTI_CLAIM_SENTENCE_THRESHOLD = 3

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


def _verification_summary(results: list[AuditResult]) -> dict:
    """Agrega o efeito da cascata de verificacao de UNSUPPORTED
    (`judging/verification/`)."""
    verified = [r for r in results if r.verification_trail]
    reclassified = [r for r in verified if r.verification_stage != "baseline" and r.verdict != AuditVerdict.UNSUPPORTED]
    return {
        "verification_ran": bool(verified),
        "count_unsupported_confirmed": sum(
            1 for r in verified if r.verdict == AuditVerdict.UNSUPPORTED and r.unsupported_confirmed
        ),
        "count_reclassified_by_verification": len(reclassified),
        "verification_stage_counts": dict(Counter(r.verification_stage for r in reclassified)),
        "mis_cited_reference_count": sum(
            1 for r in verified if r.verdict == AuditVerdict.UNSUPPORTED and r.corroborated_by_other_reference
        ),
    }


def build_reference_stats(
    chunks: list[AnswerChunk], references: list[Reference], results: list[AuditResult]
) -> list[ReferenceStats]:
    """Estatisticas por referencia: quantas vezes foi citada e como os
    chunks que a citam foram julgados. Junta `AnswerChunk.cited_reference_ids`
    (a citacao real, ja resolvida na Fase 2) com o `AuditResult` do juiz
    (Fase 3) via `answer_chunk_id` — no audit_with_llm original essa
    correspondencia era feita hoje so manualmente, lendo o `.md` a mao.

    Cobre TODAS as referencias extraidas (`references`), nao so as que
    aparecem em `chunk.cited_reference_ids` — uma referencia listada na
    resposta mas nunca citada por nenhum chunk auditado (times_cited=0)
    ainda deve ser visivel na secao 4 do relatorio, em vez de desaparecer
    silenciosamente."""
    result_by_chunk = {r.answer_chunk_id: r for r in results}

    times_cited: dict[str, int] = defaultdict(int)
    verdict_counts: dict[str, Counter] = defaultdict(Counter)
    supports_pos: dict[str, list[int]] = defaultdict(list)
    partial_pos: dict[str, list[int]] = defaultdict(list)
    absent_pos: dict[str, list[int]] = defaultdict(list)
    contradicts_pos: dict[str, list[int]] = defaultdict(list)
    unrated_pos: dict[str, list[int]] = defaultdict(list)
    key_excerpt: dict[str, str] = {}
    key_excerpt_pos: dict[str, int] = {}
    key_from_supports: set[str] = set()

    for chunk in chunks:
        result = result_by_chunk.get(chunk.id)
        pr_by_ref = {pr.reference_id: pr for pr in result.per_reference} if result else {}
        for ref_id in chunk.cited_reference_ids:
            times_cited[ref_id] += 1
            if result is None:
                continue
            verdict_counts[ref_id][result.verdict] += 1
            pr = pr_by_ref.get(ref_id)
            relation = pr.relation if pr is not None else None
            if relation is None and ref_id in result.supporting_reference_ids:
                relation = "supports"  # fallback quando o juiz so preencheu o campo antigo
            if relation == "supports":
                supports_pos[ref_id].append(chunk.position)
            elif relation == "partial":
                partial_pos[ref_id].append(chunk.position)
            elif relation == "contradicts":
                contradicts_pos[ref_id].append(chunk.position)
            elif relation == "absent":
                absent_pos[ref_id].append(chunk.position)
            elif result.per_reference:
                # o juiz detalhou as fontes e deixou esta de fora → nao contribuiu
                absent_pos[ref_id].append(chunk.position)
            else:  # nenhum detalhamento por fonte neste veredito
                unrated_pos[ref_id].append(chunk.position)

            # trecho representativo: prioriza o excerto de um `supports`
            if pr is not None and pr.excerpt:
                if relation == "supports" and ref_id not in key_from_supports:
                    key_excerpt[ref_id] = pr.excerpt
                    key_excerpt_pos[ref_id] = chunk.position
                    key_from_supports.add(ref_id)
                elif ref_id not in key_excerpt:
                    key_excerpt[ref_id] = pr.excerpt
                    key_excerpt_pos[ref_id] = chunk.position
            if (
                ref_id not in key_excerpt
                and ref_id in result.supporting_reference_ids
                and result.cited_excerpts
            ):
                key_excerpt[ref_id] = result.cited_excerpts[0]
                key_excerpt_pos[ref_id] = chunk.position

    stats = [
        ReferenceStats(
            reference_id=ref.id,
            url=ref.raw_url,
            times_cited=times_cited.get(ref.id, 0),
            supported_count=verdict_counts[ref.id].get(AuditVerdict.SUPPORTED, 0),
            unsupported_count=verdict_counts[ref.id].get(AuditVerdict.UNSUPPORTED, 0),
            contradicted_count=verdict_counts[ref.id].get(AuditVerdict.CONTRADICTED, 0),
            supporting_citations=len(supports_pos.get(ref.id, [])),
            partial_citations=len(partial_pos.get(ref.id, [])),
            status=ref.status,
            supports_positions=sorted(supports_pos.get(ref.id, [])),
            partial_positions=sorted(partial_pos.get(ref.id, [])),
            absent_positions=sorted(absent_pos.get(ref.id, [])),
            contradicts_positions=sorted(contradicts_pos.get(ref.id, [])),
            unrated_positions=sorted(unrated_pos.get(ref.id, [])),
            key_excerpt=key_excerpt.get(ref.id, ""),
            key_excerpt_position=key_excerpt_pos.get(ref.id),
        )
        for ref in references
    ]
    stats.sort(key=lambda s: s.times_cited, reverse=True)
    return stats


def _multi_claim_chunks(chunks: list[AnswerChunk]) -> list[PotentiallyUnsourcedChunk]:
    out: list[PotentiallyUnsourcedChunk] = []
    for chunk in chunks:
        if not chunk.cited_reference_ids:
            continue
        if chunk.sentence_count < _MULTI_CLAIM_SENTENCE_THRESHOLD:
            continue
        excerpt = " ".join(chunk.text.split())
        out.append(
            PotentiallyUnsourcedChunk(
                answer_chunk_id=chunk.id,
                sentence_count=chunk.sentence_count,
                cited_reference_ids=list(chunk.cited_reference_ids),
                excerpt=excerpt if len(excerpt) <= 300 else excerpt[:299].rstrip() + "…",
            )
        )
    return out


def _uncited_claims(chunks: list[AnswerChunk]) -> list[UncitedClaimChunk]:
    out: list[UncitedClaimChunk] = []
    for chunk in chunks:
        if not chunk.is_uncited_claim:
            continue
        excerpt = " ".join(chunk.text.split())
        out.append(
            UncitedClaimChunk(
                answer_chunk_id=chunk.id,
                excerpt=excerpt if len(excerpt) <= 300 else excerpt[:299].rstrip() + "…",
            )
        )
    return out


def _skipped_reason_counts(skipped: list[SkippedChunk]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for chunk in skipped:
        reason = chunk.reason.lower()
        if "nao cita nenhuma referencia" in reason or "não cita nenhuma referência" in reason:
            counts["sem_citacao"] += 1
        elif "conteudo indexado" in reason or "conteúdo indexado" in reason:
            counts["ref_sem_conteudo"] += 1
        else:
            counts["outro"] += 1
    return dict(counts)


def aggregate_report(
    *,
    run_id: str,
    answer_id: str,
    tool_name: str,
    answer_path: str | None = None,
    chunks: list[AnswerChunk],
    references: list[Reference],
    results: list[AuditResult],
    skipped: list[SkippedChunk] | None = None,
    processing_time_seconds: float = 0.0,
    generated_at: datetime | None = None,
    judge_config: JudgeConfig | None = None,
    source_info: SourceInfo | None = None,
) -> Report:
    """Monta o `Report` final de uma execucao — a peca que hoje nao existe
    em codigo algum nos tres repositorios originais; todo relatorio rico
    (percentual por referencia, referencias mortas, custo total) e escrito
    manualmente a partir do JSON bruto no audit_with_llm."""
    percentages = _verdict_percentages(results, total=len(chunks))
    counts = _verdict_counts(results)
    verification = _verification_summary(results)
    cost = summarize_cost(results)
    skipped = skipped or []
    cost_tracked = (
        is_cost_tracked(judge_config.provider, judge_config.model) if judge_config is not None else True
    )

    reference_stats = build_reference_stats(chunks, references, results)
    partially_supported = sum(
        1 for r in results if r.verdict == AuditVerdict.SUPPORTED and r.unsupported_aspects
    )
    uncredited_refs = sum(
        1
        for s in reference_stats
        if s.times_cited > 0 and s.supporting_citations == 0 and s.partial_citations == 0
    )

    uncited_claims = _uncited_claims(chunks)
    claim_chunks = sum(1 for c in chunks if c.cited_reference_ids or c.is_uncited_claim)

    return Report(
        run_id=run_id,
        answer_id=answer_id,
        tool_name=tool_name,
        answer_path=answer_path,
        generated_at=generated_at or datetime.now(timezone.utc),
        judge_config=judge_config,
        source_info=source_info,
        total_chunks=len(chunks),
        pct_supported=percentages["pct_supported"],
        pct_unsupported=percentages["pct_unsupported"],
        pct_contradicted=percentages["pct_contradicted"],
        count_supported=counts["count_supported"],
        count_unsupported=counts["count_unsupported"],
        count_contradicted=counts["count_contradicted"],
        count_skipped=len(skipped),
        count_partially_supported=partially_supported,
        skipped_reason_counts=_skipped_reason_counts(skipped),
        uncredited_reference_count=uncredited_refs,
        verification_ran=verification["verification_ran"],
        count_unsupported_confirmed=verification["count_unsupported_confirmed"],
        count_reclassified_by_verification=verification["count_reclassified_by_verification"],
        verification_stage_counts=verification["verification_stage_counts"],
        mis_cited_reference_count=verification["mis_cited_reference_count"],
        dead_references=[r for r in references if r.status == ReferenceStatus.DEAD],
        inaccessible_references=[r for r in references if r.status == ReferenceStatus.INACCESSIBLE],
        skipped_chunks=skipped,
        reference_stats=reference_stats,
        potentially_unsourced_chunks=_multi_claim_chunks(chunks),
        uncited_claims=uncited_claims,
        count_uncited_claims=len(uncited_claims),
        count_claim_chunks=claim_chunks,
        total_cost_usd=cost.total_cost_usd,
        total_tokens=cost.total_tokens,
        cost_tracked=cost_tracked,
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
