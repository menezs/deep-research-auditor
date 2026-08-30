from __future__ import annotations

from ...common.errors import LLMParseError
from ...indexing.retriever import Retriever
from ...logging_config import get_logger
from ...models import AnswerChunk, AuditResult, AuditVerdict
from ..judge import Verifier
from .cascade import StageOutcome, _dedup

logger = get_logger(__name__)


class CrossReferenceStage:
    """Etapa C — checagem cruzada contra o corpus inteiro (anotacao).

    Recupera evidencia em TODAS as referencias baixadas (nao so as
    citadas) e re-julga. NUNCA muda o veredito para SUPPORTED — a
    auditoria e sobre a fonte citada. Apenas anota:
    `corroborated_by_other_reference` (claim verdadeiro, citacao trocada)
    ou `contradicted_by_other_reference`."""

    name = "cross_reference"

    def __init__(self, retriever: Retriever, verifier: Verifier) -> None:
        self._retriever = retriever
        self._verifier = verifier

    def run(self, chunk: AnswerChunk, current: AuditResult) -> StageOutcome:
        curated = self._retriever.retrieve_whole_corpus(chunk)
        if not curated.passages:
            return StageOutcome(
                verdict=AuditVerdict.UNSUPPORTED,
                justification=current.justification,
                note="corpus sem trechos relevantes",
            )
        try:
            result = self._verifier.verify(chunk, curated)
        except LLMParseError as exc:
            logger.warning("Juiz nao parseavel na checagem cruzada (chunk %s): %s", chunk.id, exc)
            return StageOutcome(
                verdict=AuditVerdict.UNSUPPORTED,
                justification=current.justification,
                note="juiz nao parseavel nesta etapa",
                inconclusive=True,
            )

        cited = set(chunk.cited_reference_ids)
        other_refs = _dedup([p.reference_id for p in curated.passages if p.reference_id not in cited])

        outcome = StageOutcome(
            verdict=AuditVerdict.UNSUPPORTED,  # C nunca reclassifica para SUPPORTED
            justification=current.justification,
            note=f"corpus inteiro: {result.verdict.value}",
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost_usd=result.cost_usd,
            latency_ms=result.latency_ms,
        )
        if result.verdict == AuditVerdict.SUPPORTED and other_refs:
            outcome.corroborated_by = other_refs
            outcome.note = f"claim sustentado por outra(s) referencia(s) nao citada(s): {', '.join(other_refs)}"
        elif result.verdict == AuditVerdict.CONTRADICTED and other_refs:
            outcome.contradicted_by = other_refs
            outcome.note = f"claim contradito por outra(s) referencia(s) nao citada(s): {', '.join(other_refs)}"
        return outcome
