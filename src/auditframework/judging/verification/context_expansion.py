from __future__ import annotations

from ...common.errors import LLMParseError
from ...indexing.retriever import Retriever
from ...logging_config import get_logger
from ...models import AnswerChunk, AuditResult, AuditVerdict
from ..judge import Verifier
from .cascade import StageOutcome

logger = get_logger(__name__)


class ContextExpansionStage:
    """Etapa A — small-to-big na propria referencia citada.

    Refaz a recuperacao escopada com um `rerank_top_k` maior e expande
    cada trecho com os vizinhos imediatos (mesmo documento), remontando o
    contexto em ordem de documento. Cobre o caso em que o fato existia na
    referencia citada, mas caiu na fronteira de um chunk de 512 tokens ou
    logo fora da janela de rerank do julgamento inicial."""

    name = "context_expansion"

    def __init__(self, retriever: Retriever, verifier: Verifier, *, neighbor_window: int, rerank_top_k: int) -> None:
        self._retriever = retriever
        self._verifier = verifier
        self._neighbor_window = neighbor_window
        self._rerank_top_k = rerank_top_k

    def run(self, chunk: AnswerChunk, current: AuditResult) -> StageOutcome:
        curated = self._retriever.retrieve_expanded(
            chunk, neighbor_window=self._neighbor_window, rerank_top_k=self._rerank_top_k
        )
        if curated.skip_reason is not None or not curated.passages:
            return StageOutcome(
                verdict=AuditVerdict.UNSUPPORTED,
                justification=current.justification,
                note=f"expansao de contexto sem trechos ({curated.skip_reason or 'vazio'})",
                inconclusive=True,
            )
        try:
            result = self._verifier.verify(chunk, curated)
        except LLMParseError as exc:
            logger.warning("Juiz nao parseavel na expansao de contexto (chunk %s): %s", chunk.id, exc)
            return StageOutcome(
                verdict=AuditVerdict.UNSUPPORTED,
                justification=current.justification,
                note="juiz nao parseavel nesta etapa",
                inconclusive=True,
            )
        return StageOutcome(
            verdict=result.verdict,
            justification=result.justification,
            cited_excerpts=result.cited_excerpts,
            note=result.justification,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost_usd=result.cost_usd,
            latency_ms=result.latency_ms,
        )
