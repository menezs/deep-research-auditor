from pathlib import Path

from auditframework.ingestion.registry import ReferenceRegistry
from auditframework.models import Document, Reference, ReferenceStatus


def _make_reference(ref_id: str = "ref1") -> Reference:
    return Reference(
        id=ref_id,
        citation_markers=["[1]"],
        raw_url="https://example.com/a",
        normalized_url="https://example.com/a",
        status=ReferenceStatus.DOWNLOADED,
        source_answer_id="answer-1",
        tool_name="ChatGPT",
    )


def test_load_references_returns_empty_list_when_no_file(tmp_path: Path):
    registry = ReferenceRegistry(tmp_path)
    assert registry.load_references() == []


def test_save_and_load_references_roundtrip(tmp_path: Path):
    registry = ReferenceRegistry(tmp_path)
    refs = [_make_reference("ref1"), _make_reference("ref2")]
    registry.save_references(refs)

    reloaded = registry.load_references()
    assert {r.id for r in reloaded} == {"ref1", "ref2"}


def test_save_document_creates_markdown_and_metadata(tmp_path: Path):
    registry = ReferenceRegistry(tmp_path)
    document = Document(
        reference_id="ref1",
        markdown_path=Path("documents/ref1.md"),
        content_hash="deadbeef",
        fetch_method="requests",
        word_count=3,
    )
    registry.save_document(document, "conteudo em markdown")

    assert registry.has_document("ref1") is True
    assert registry.document_path("ref1").read_text(encoding="utf-8") == "conteudo em markdown"
    assert registry.metadata_path("ref1").exists()


def test_has_document_false_when_not_saved(tmp_path: Path):
    registry = ReferenceRegistry(tmp_path)
    assert registry.has_document("nao-existe") is False


def test_load_document_returns_persisted_metadata(tmp_path: Path):
    registry = ReferenceRegistry(tmp_path)
    document = Document(
        reference_id="ref1",
        markdown_path=Path("documents/ref1.md"),
        content_hash="deadbeef",
        fetch_method="requests",
        word_count=3,
    )
    registry.save_document(document, "conteudo")

    loaded = registry.load_document("ref1")

    assert loaded == document


def test_no_leftover_tmp_files_after_save(tmp_path: Path):
    registry = ReferenceRegistry(tmp_path)
    registry.save_references([_make_reference()])
    document = Document(
        reference_id="ref1",
        markdown_path=Path("documents/ref1.md"),
        content_hash="deadbeef",
        fetch_method="requests",
        word_count=3,
    )
    registry.save_document(document, "conteudo")

    tmp_files = list(tmp_path.rglob("*.tmp"))
    assert tmp_files == []
