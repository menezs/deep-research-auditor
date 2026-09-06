from __future__ import annotations

import pytest

from auditframework.common.llm_client import LLMUsage
from auditframework.judging.verification.cascade import StageOutcome, VerificationCascade, _dedup, _truncate
from auditframework.judging.verification.context_expansion import ContextExpansionStage
from auditframework.judging.verification.cross_reference import CrossReferenceStage
from auditframework.judging.verification.full_doc_scan import FullDocumentScanStage
from auditframework.judging.verification.prompts import WindowAssessment
from auditframework.models import AnswerChunk, AuditResult, AuditVerdict, CuratedDocument, RetrievedPassage


def _chunk(cited=("refA",)) -> AnswerChunk:
    return AnswerChunk(id="c1", answer_id="a1", position=0, text="claim de teste", cited_reference_ids=list(cited))


def _baseline() -> AuditResult:
    return AuditResult(
        answer_chunk_id="c1",
        verdict=AuditVerdict.UNSUPPORTED,
        justification="baseline: sem evidencia",
        judge_model="fake",
    )


def _result(verdict: AuditVerdict, **kw) -> AuditResult:
    return AuditResult(
        answer_chunk_id="c1", verdict=verdict, justification=kw.get("justification", "j"),
        cited_excerpts=kw.get("cited_excerpts", []),
        prompt_tokens=kw.get("prompt_tokens", 0), completion_tokens=kw.get("completion_tokens", 0),
        cost_usd=kw.get("cost_usd", 0.0), latency_ms=kw.get("latency_ms", 0), judge_model="fake",
    )


class _FakeStage:
    def __init__(self, name: str, outcome: StageOutcome):
        self.name = name
        self._outcome = outcome
        self.calls = 0

    def run(self, chunk, current) -> StageOutcome:
        self.calls += 1
        return self._outcome


class TestHelpers:
    def test_truncate(self):
        assert _truncate("a  b\n c") == "a b c"
        assert _truncate("x" * 500).endswith("…")

    def test_dedup_preserves_order_and_drops_empty(self):
        assert _dedup(["a", "", "b", "a", "c"]) == ["a", "b", "c"]


class TestCascade:
    def test_stops_at_first_stage_that_flips_the_verdict(self):
        a = _FakeStage("context_expansion", StageOutcome(AuditVerdict.SUPPORTED, "achei suporte", "nota"))
        b = _FakeStage("full_doc_scan", StageOutcome(AuditVerdict.UNSUPPORTED, "", "nota"))
        c = _FakeStage("cross_reference", StageOutcome(AuditVerdict.UNSUPPORTED, "", "nota"))

        result = VerificationCascade([a, b, c]).run(_chunk(), _baseline())

        assert result.verdict == AuditVerdict.SUPPORTED
        assert result.verification_stage == "context_expansion"
        assert (a.calls, b.calls, c.calls) == (1, 0, 0)
        assert [s.stage for s in result.verification_trail] == ["baseline", "context_expansion"]

    def test_confirms_unsupported_after_full_doc_scan_and_still_runs_cross_reference(self):
        a = _FakeStage("context_expansion", StageOutcome(AuditVerdict.UNSUPPORTED, "", "nada"))
        b = _FakeStage("full_doc_scan", StageOutcome(AuditVerdict.UNSUPPORTED, "", "ausente das 3 janelas"))
        c = _FakeStage(
            "cross_reference",
            StageOutcome(AuditVerdict.UNSUPPORTED, "", "corroborado", corroborated_by=["refB"]),
        )

        result = VerificationCascade([a, b, c]).run(_chunk(), _baseline())

        assert result.verdict == AuditVerdict.UNSUPPORTED
        assert result.unsupported_confirmed is True
        assert result.corroborated_by_other_reference == ["refB"]
        assert (a.calls, b.calls, c.calls) == (1, 1, 1)

    def test_inconclusive_full_doc_scan_does_not_set_confirmed(self):
        b = _FakeStage(
            "full_doc_scan", StageOutcome(AuditVerdict.UNSUPPORTED, "", "sem documento", inconclusive=True)
        )
        result = VerificationCascade([b]).run(_chunk(), _baseline())
        assert result.unsupported_confirmed is False

    def test_accumulates_tokens_and_cost_across_stages(self):
        a = _FakeStage(
            "context_expansion",
            StageOutcome(AuditVerdict.UNSUPPORTED, "", "n", prompt_tokens=100, completion_tokens=10, cost_usd=0.01),
        )
        b = _FakeStage(
            "full_doc_scan",
            StageOutcome(AuditVerdict.UNSUPPORTED, "", "n", prompt_tokens=200, completion_tokens=20, cost_usd=0.02),
        )
        result = VerificationCascade([a, b]).run(_chunk(), _baseline())
        assert result.prompt_tokens == 300
        assert result.completion_tokens == 30
        assert result.cost_usd == pytest.approx(0.03)

    def test_failing_stage_does_not_crash_the_cascade(self):
        class _Boom:
            name = "context_expansion"

            def run(self, chunk, current):
                raise RuntimeError("boom")

        result = VerificationCascade([_Boom()]).run(_chunk(), _baseline())
        assert result.verdict == AuditVerdict.UNSUPPORTED
        assert result.verification_trail[-1].note == "etapa falhou (ver log)"


class _FakeRetriever:
    def __init__(self, curated: CuratedDocument):
        self._curated = curated

    def retrieve_expanded(self, chunk, *, neighbor_window, rerank_top_k):
        return self._curated

    def retrieve(self, chunk):
        return self._curated

    def retrieve_whole_corpus(self, chunk):
        return self._curated


class _FakeVerifier:
    def __init__(self, result: AuditResult):
        self._result = result

    def verify(self, chunk, curated) -> AuditResult:
        return self._result


def _curated(passages=True) -> CuratedDocument:
    p = [RetrievedPassage(reference_chunk_id="refA-1", reference_id="refA", score=0.9, text="t")] if passages else []
    return CuratedDocument(answer_chunk_id="c1", passages=p, assembled_context="ctx" if passages else "")


class TestContextExpansionStage:
    def test_reclassifies_when_expanded_context_supports(self):
        stage = ContextExpansionStage(
            _FakeRetriever(_curated()), _FakeVerifier(_result(AuditVerdict.SUPPORTED, justification="agora suporta")),
            neighbor_window=1, rerank_top_k=40,
        )
        outcome = stage.run(_chunk(), _baseline())
        assert outcome.verdict == AuditVerdict.SUPPORTED
        assert outcome.justification == "agora suporta"

    def test_inconclusive_when_no_passages(self):
        stage = ContextExpansionStage(
            _FakeRetriever(_curated(passages=False)), _FakeVerifier(_result(AuditVerdict.SUPPORTED)),
            neighbor_window=1, rerank_top_k=40,
        )
        outcome = stage.run(_chunk(), _baseline())
        assert outcome.verdict == AuditVerdict.UNSUPPORTED
        assert outcome.inconclusive is True


class TestCrossReferenceStage:
    def test_never_flips_to_supported_but_records_corroboration(self):
        curated = CuratedDocument(
            answer_chunk_id="c1",
            passages=[RetrievedPassage(reference_chunk_id="refB-0", reference_id="refB", score=0.8, text="t")],
            assembled_context="ctx",
        )
        stage = CrossReferenceStage(_FakeRetriever(curated), _FakeVerifier(_result(AuditVerdict.SUPPORTED)))
        outcome = stage.run(_chunk(cited=["refA"]), _baseline())
        assert outcome.verdict == AuditVerdict.UNSUPPORTED
        assert outcome.corroborated_by == ["refB"]

    def test_records_contradiction_from_other_reference(self):
        curated = CuratedDocument(
            answer_chunk_id="c1",
            passages=[RetrievedPassage(reference_chunk_id="refC-0", reference_id="refC", score=0.8, text="t")],
            assembled_context="ctx",
        )
        stage = CrossReferenceStage(_FakeRetriever(curated), _FakeVerifier(_result(AuditVerdict.CONTRADICTED)))
        outcome = stage.run(_chunk(cited=["refA"]), _baseline())
        assert outcome.verdict == AuditVerdict.UNSUPPORTED
        assert outcome.contradicted_by == ["refC"]


class _FakeRegistry:
    def __init__(self, docs: dict[str, str], tmp_path):
        self._paths = {}
        for ref_id, text in docs.items():
            p = tmp_path / f"{ref_id}.md"
            p.write_text(text, encoding="utf-8")
            self._paths[ref_id] = p

    def has_document(self, ref_id: str) -> bool:
        return ref_id in self._paths

    def document_path(self, ref_id: str):
        return self._paths[ref_id]


class _WindowLLM:
    def __init__(self, by_keyword: dict[str, str]):
        self._by_keyword = by_keyword
        self.calls = 0

    def complete_json(self, *, system_message, user_prompt, schema):
        self.calls += 1
        relation = "not_addressed"
        for kw, rel in self._by_keyword.items():
            if kw in user_prompt:
                relation = rel
                break
        return (
            schema(relation=relation, justification="j", cited_excerpts=[]),
            LLMUsage(prompt_tokens=50, completion_tokens=5, latency_ms=1, cost_usd=0.0),
        )


class TestFullDocumentScanStage:
    def test_inconclusive_when_no_cited_document(self, tmp_path):
        stage = FullDocumentScanStage(
            _FakeRegistry({}, tmp_path), _WindowLLM({}), window_tokens=200, window_overlap=20
        )
        outcome = stage.run(_chunk(cited=["refA"]), _baseline())
        assert outcome.inconclusive is True
        assert outcome.verdict == AuditVerdict.UNSUPPORTED

    def test_supported_short_circuits_remaining_windows(self, tmp_path):
        pytest.importorskip("semantic_text_splitter")
        doc = "# Doc\n\n" + ("paragrafo neutro. " * 60) + "\n\nAQUI_O_FATO confirma o claim.\n\n" + ("mais texto. " * 60)
        stage = FullDocumentScanStage(
            _FakeRegistry({"refA": doc}, tmp_path), _WindowLLM({"AQUI_O_FATO": "supports"}),
            window_tokens=120, window_overlap=10,
        )
        outcome = stage.run(_chunk(cited=["refA"]), _baseline())
        assert outcome.verdict == AuditVerdict.SUPPORTED

    def test_all_windows_not_addressed_stays_unsupported(self, tmp_path):
        pytest.importorskip("semantic_text_splitter")
        stage = FullDocumentScanStage(
            _FakeRegistry({"refA": "# Doc\n\n" + ("texto irrelevante. " * 80)}, tmp_path),
            _WindowLLM({}), window_tokens=120, window_overlap=10,
        )
        outcome = stage.run(_chunk(cited=["refA", "refB"]), _baseline())
        assert outcome.verdict == AuditVerdict.UNSUPPORTED
        assert outcome.inconclusive is False
        # afirmacao comprovadamente ausente de cada fonte varrida
        assert {(pr.reference_id, pr.relation) for pr in outcome.per_reference} == {("refA", "absent")}

    def test_partial_window_annotates_partial_and_stays_unconfirmed(self, tmp_path):
        pytest.importorskip("semantic_text_splitter")
        doc = "# Doc\n\n" + ("texto neutro. " * 60) + "\n\nPARTE_DO_FATO aparece aqui.\n\n" + ("mais. " * 60)
        stage = FullDocumentScanStage(
            _FakeRegistry({"refA": doc, "refB": "# B\n\n" + ("nada a ver. " * 80)}, tmp_path),
            _WindowLLM({"PARTE_DO_FATO": "partial"}), window_tokens=120, window_overlap=10,
        )
        outcome = stage.run(_chunk(cited=["refA", "refB"]), _baseline())
        assert outcome.verdict == AuditVerdict.UNSUPPORTED
        # nao e "confirmado ausente" — parte do fato esta na fonte
        assert outcome.inconclusive is True
        rel = {pr.reference_id: pr.relation for pr in outcome.per_reference}
        assert rel == {"refA": "partial", "refB": "absent"}
