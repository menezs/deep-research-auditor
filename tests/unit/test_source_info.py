from pathlib import Path

from auditframework.extraction.source_info import _is_browser_print, _pdf_date, build_source_info
from auditframework.models import AnswerChunk, Reference


def _ref(ref_id: str) -> Reference:
    return Reference(
        id=ref_id,
        citation_markers=[f"[{ref_id}]"],
        raw_url=f"https://example.com/{ref_id}",
        normalized_url=f"https://example.com/{ref_id}",
        source_answer_id="a1",
        tool_name="Perplexity",
    )


def test_pdf_date_is_parsed_to_readable_form():
    assert _pdf_date("D:20260830183346+00'00'") == "2026-08-30 18:33:46 UTC"
    assert _pdf_date("D:20240101") == "2024-01-01 00:00:00"
    assert _pdf_date(None) is None
    assert _pdf_date("qualquer coisa") == "qualquer coisa"


def test_browser_print_detection():
    assert _is_browser_print("Chromium", "Skia/PDF m127") is True
    assert _is_browser_print(None, "Skia/PDF m151 Google Docs Renderer") is True
    assert _is_browser_print("Microsoft Word", "Microsoft Word") is False


def test_marker_topology_and_reference_counts(tmp_path: Path):
    md = tmp_path / "resposta.md"
    md.write_text(
        "Afirmação A [1]. Afirmação B [2][1]. Afirmação C sem citação. Afirmação D [9].",
        encoding="utf-8",
    )
    refs = [_ref("1"), _ref("2"), _ref("3")]
    chunks = [
        AnswerChunk(id="c0", answer_id="a", position=0, text="Afirmação A", cited_reference_ids=["1"]),
        AnswerChunk(id="c1", answer_id="a", position=1, text="Afirmação B", cited_reference_ids=["2", "1"]),
    ]
    si = build_source_info(path=md, body=md.read_text(encoding="utf-8"), references=refs, chunks=chunks)

    assert si.format == ".md"
    assert si.size_bytes > 0
    assert si.page_count is None
    assert si.citation_markers_total == 4  # [1] [2] [1] [9]
    assert si.citation_markers_distinct == 3  # 1, 2, 9
    assert si.references_listed == 3
    assert si.references_never_cited == 1  # ref "3"
