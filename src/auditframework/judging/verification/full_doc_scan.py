from __future__ import annotations

from ...common.errors import LLMParseError
from ...common.llm_client import LLMClient, LLMUsage
from ...ingestion.registry import ReferenceRegistry
from ...logging_config import get_logger
from ...models import AnswerChunk, AuditResult, AuditVerdict
from .cascade import StageOutcome
from .prompts import WINDOW_SCAN_SYSTEM_MESSAGE, WindowAssessment, build_window_scan_prompt

logger = get_logger(__name__)


class FullDocumentScanStage:
    """Etapa B — varredura do documento citado INTEIRO, sem recuperacao.

    Le o markdown completo de cada referencia citada, fatia em janelas
    grandes e pergunta ao juiz, por janela, se ela sustenta / contradiz /
    nao trata o claim. Elimina o recall da recuperacao como variavel: se
    esta etapa mantiver UNSUPPORTED, o claim comprovadamente nao esta na
    referencia citada (`unsupported_confirmed`)."""

    name = "full_doc_scan"

    def __init__(
        self, registry: ReferenceRegistry, llm_client: LLMClient, *, window_tokens: int, window_overlap: int
    ) -> None:
        self._registry = registry
        self._llm = llm_client
        self._window_tokens = window_tokens
        self._window_overlap = window_overlap

    def run(self, chunk: AnswerChunk, current: AuditResult) -> StageOutcome:
        available = [rid for rid in chunk.cited_reference_ids if self._registry.has_document(rid)]
        if not available:
            return StageOutcome(
                verdict=AuditVerdict.UNSUPPORTED,
                justification=current.justification,
                note="nenhum documento citado disponivel para varredura completa",
                inconclusive=True,
            )

        windows: list[tuple[str, str]] = []
        for rid in available:
            markdown = self._registry.document_path(rid).read_text(encoding="utf-8")
            windows.extend((rid, w) for w in self._split(markdown))

        pt = ct = lat = 0
        cost = 0.0
        contradiction: tuple[str, WindowAssessment] | None = None

        for rid, window in windows:
            try:
                assessment, usage = self._assess(chunk.text, window)
            except LLMParseError as exc:
                logger.warning("Juiz nao parseavel numa janela de %s (chunk %s): %s", rid, chunk.id, exc)
                continue
            pt += usage.prompt_tokens
            ct += usage.completion_tokens
            cost += usage.cost_usd
            lat += usage.latency_ms
            if assessment.relation == "supports":
                return StageOutcome(
                    verdict=AuditVerdict.SUPPORTED,
                    justification=assessment.justification,
                    cited_excerpts=assessment.cited_excerpts,
                    supporting_reference_ids=[rid],
                    note=f"suporte encontrado na varredura completa de {rid}",
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    cost_usd=cost,
                    latency_ms=lat,
                )
            if assessment.relation == "contradicts" and contradiction is None:
                contradiction = (rid, assessment)

        if contradiction is not None:
            rid, assessment = contradiction
            return StageOutcome(
                verdict=AuditVerdict.CONTRADICTED,
                justification=assessment.justification,
                cited_excerpts=assessment.cited_excerpts,
                note=f"contradicao encontrada na varredura completa de {rid}",
                prompt_tokens=pt,
                completion_tokens=ct,
                cost_usd=cost,
                latency_ms=lat,
            )

        return StageOutcome(
            verdict=AuditVerdict.UNSUPPORTED,
            justification=current.justification,
            note=f"claim ausente das {len(windows)} janela(s) do(s) documento(s) citado(s)",
            prompt_tokens=pt,
            completion_tokens=ct,
            cost_usd=cost,
            latency_ms=lat,
        )

    def _assess(self, claim: str, window: str) -> tuple[WindowAssessment, LLMUsage]:
        return self._llm.complete_json(
            system_message=WINDOW_SCAN_SYSTEM_MESSAGE,
            user_prompt=build_window_scan_prompt(claim, window),
            schema=WindowAssessment,
        )

    def _split(self, markdown: str) -> list[str]:
        from semantic_text_splitter import MarkdownSplitter

        splitter = MarkdownSplitter.from_tiktoken_model(
            "gpt-3.5-turbo", self._window_tokens, overlap=self._window_overlap
        )
        return [w for w in splitter.chunks(markdown) if w.strip()]
