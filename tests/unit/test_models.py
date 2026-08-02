from datetime import datetime, timezone

from auditframework.models import (
    AnswerChunk,
    AuditVerdict,
    CuratedDocument,
    Document,
    Reference,
    ReferenceStatus,
    Report,
    RetrievedPassage,
    SkippedChunk,
)


def test_reference_id_is_stable_for_same_normalized_url():
    url = "https://example.com/artigo"
    assert Reference.id_for_url(url) == Reference.id_for_url(url)


def test_reference_id_differs_for_different_urls():
    assert Reference.id_for_url("https://a.com") != Reference.id_for_url("https://b.com")


def test_reference_roundtrip():
    ref = Reference(
        id=Reference.id_for_url("https://example.com/artigo"),
        citation_markers=["[1]", "[3]"],
        raw_url="https://Example.com/artigo/",
        normalized_url="https://example.com/artigo",
        status=ReferenceStatus.DOWNLOADED,
        source_answer_id="answer-1",
        tool_name="ChatGPT",
    )
    dumped = ref.model_dump_json()
    restored = Reference.model_validate_json(dumped)
    assert restored == ref


def test_document_requires_reference_id():
    doc = Document(
        reference_id="abc123",
        markdown_path="data/runs/x/documents/abc123.md",
        content_hash="deadbeef",
        fetch_method="requests",
        word_count=120,
    )
    assert doc.reference_id == "abc123"


def test_answer_chunk_default_references_is_empty_list():
    chunk = AnswerChunk(id="c1", answer_id="a1", position=0, text="texto sem citacao")
    assert chunk.cited_reference_ids == []


def test_curated_document_defaults_to_no_skip_reason():
    curated = CuratedDocument(
        answer_chunk_id="c1",
        passages=[
            RetrievedPassage(reference_chunk_id="rc1", reference_id="r1", score=0.8, text="trecho")
        ],
        assembled_context="trecho",
    )
    assert curated.skip_reason is None
    assert curated.passages[0].reference_id == "r1"


def test_curated_document_tracks_skip_reason():
    curated = CuratedDocument(answer_chunk_id="c1", assembled_context="", skip_reason="sem evidencia citada")
    assert curated.skip_reason == "sem evidencia citada"
    assert curated.passages == []


def test_skipped_chunk_roundtrip():
    skipped = SkippedChunk(answer_chunk_id="c1", reason="referencia nao baixada")
    restored = SkippedChunk.model_validate_json(skipped.model_dump_json())
    assert restored == skipped


def test_audit_verdict_has_exactly_three_content_categories():
    # apenas 3 categorias de classificacao de conteudo devem existir;
    # falhas tecnicas (parsing do juiz) nao viram um AuditVerdict, veja
    # Verifier.verify / test_judge.py.
    assert {v.value for v in AuditVerdict} == {"supported", "unsupported", "contradicted"}


def test_report_minimal_construction():
    report = Report(
        run_id="run-1",
        answer_id="answer-1",
        tool_name="Gemini",
        generated_at=datetime.now(timezone.utc),
        pct_supported=50.0,
        pct_unsupported=30.0,
        pct_contradicted=20.0,
    )
    assert report.dead_references == []
    assert report.total_cost_usd == 0.0
