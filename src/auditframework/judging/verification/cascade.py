from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ...indexing.embeddings import Embedder
from ...indexing.reranker import Reranker
from ...indexing.retriever import Retriever
from ...indexing.vector_store import VectorStore
from ...ingestion.registry import ReferenceRegistry
from ...common.llm_client import LLMClient
from ...logging_config import get_logger
from ...models import AnswerChunk, AuditResult, AuditVerdict, ReferenceVerdict, VerificationStep
from ..judge import Verifier

logger = get_logger(__name__)

_NOTE_MAX = 280


def _truncate(text: str, limit: int = _NOTE_MAX) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _dedup(items: list[str]) -> list[str]:
    return list(dict.fromkeys(i for i in items if i))


@dataclass
class StageOutcome:
    """Resultado de uma etapa da cascata. A etapa nunca mexe no
    `AuditResult` diretamente — devolve isto e o `VerificationCascade`
    monta o resultado final."""

    verdict: AuditVerdict
    justification: str
    note: str
    cited_excerpts: list[str] = field(default_factory=list)
    supporting_reference_ids: list[str] = field(default_factory=list)
    unsupported_aspects: list[str] = field(default_factory=list)
    per_reference: list[ReferenceVerdict] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    inconclusive: bool = False
    corroborated_by: list[str] = field(default_factory=list)
    contradicted_by: list[str] = field(default_factory=list)


class VerificationStage(Protocol):
    name: str

    def run(self, chunk: AnswerChunk, current: AuditResult) -> StageOutcome: ...


class VerificationCascade:
    """Chain of Responsibility aplicada a um veredito `UNSUPPORTED` inicial.

    Roda as etapas em ordem e para na primeira que muda o veredito para
    algo diferente de UNSUPPORTED. Cada chamada de LLM feita pelas etapas
    e somada ao custo/tokens do `AuditResult` final. A trilha completa
    (`verification_trail`) fica no proprio resultado."""

    _CONFIRMED_BY = "full_doc_scan"

    def __init__(self, stages: list[VerificationStage]) -> None:
        self._stages = stages

    def run(self, chunk: AnswerChunk, baseline: AuditResult) -> AuditResult:
        result = baseline.model_copy(deep=True)
        result.verification_trail.append(
            VerificationStep(stage="baseline", verdict=baseline.verdict, note=_truncate(baseline.justification))
        )
        for stage in self._stages:
            if result.verdict != AuditVerdict.UNSUPPORTED:
                break
            try:
                outcome = stage.run(chunk, result)
            except Exception:  # etapa best-effort: nunca derruba o veredito inicial
                logger.exception("Etapa de verificacao %s falhou para o chunk %s", stage.name, chunk.id)
                result.verification_trail.append(
                    VerificationStep(stage=stage.name, verdict=result.verdict, note="etapa falhou (ver log)")
                )
                continue

            result.prompt_tokens += outcome.prompt_tokens
            result.completion_tokens += outcome.completion_tokens
            result.cost_usd += outcome.cost_usd
            result.latency_ms += outcome.latency_ms
            result.verification_trail.append(
                VerificationStep(stage=stage.name, verdict=outcome.verdict, note=_truncate(outcome.note))
            )
            result.corroborated_by_other_reference = _dedup(
                result.corroborated_by_other_reference + outcome.corroborated_by
            )
            result.contradicted_by_other_reference = _dedup(
                result.contradicted_by_other_reference + outcome.contradicted_by
            )

            if outcome.per_reference:
                # uma etapa mais completa (varredura do doc inteiro) tem a
                # palavra final sobre a relacao de cada fonte citada
                result.per_reference = outcome.per_reference

            if outcome.verdict != AuditVerdict.UNSUPPORTED:
                result.verdict = outcome.verdict
                result.justification = outcome.justification
                result.cited_excerpts = outcome.cited_excerpts or result.cited_excerpts
                result.supporting_reference_ids = outcome.supporting_reference_ids
                result.unsupported_aspects = outcome.unsupported_aspects
                result.verification_stage = stage.name
            elif stage.name == self._CONFIRMED_BY and not outcome.inconclusive:
                result.unsupported_confirmed = True

        return result


def build_verification_cascade(
    *,
    embedder: Embedder,
    vector_store: VectorStore,
    reranker: Reranker | None,
    registry: ReferenceRegistry,
    llm_client: LLMClient,
    top_k: int,
    rerank_top_k: int,
    neighbor_window: int,
    verification_rerank_top_k: int,
    full_doc_window_tokens: int,
    full_doc_window_overlap: int,
) -> VerificationCascade:
    from .context_expansion import ContextExpansionStage
    from .cross_reference import CrossReferenceStage
    from .full_doc_scan import FullDocumentScanStage

    verifier = Verifier(llm_client)
    retriever = Retriever(embedder, vector_store, reranker, top_k, rerank_top_k)
    return VerificationCascade(
        [
            ContextExpansionStage(
                retriever, verifier, neighbor_window=neighbor_window, rerank_top_k=verification_rerank_top_k
            ),
            FullDocumentScanStage(
                registry, llm_client, window_tokens=full_doc_window_tokens, window_overlap=full_doc_window_overlap
            ),
            CrossReferenceStage(retriever, verifier),
        ]
    )
