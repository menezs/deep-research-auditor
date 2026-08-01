from auditframework.indexing.chunkers import AnswerChunker
from auditframework.models import Reference


def _ref(ref_id: str, markers: list[str]) -> Reference:
    return Reference(
        id=ref_id,
        citation_markers=markers,
        raw_url=f"https://example.com/{ref_id}",
        normalized_url=f"https://example.com/{ref_id}",
        source_answer_id="a1",
        tool_name="ChatGPT",
    )


def test_splits_by_bracket_markers_and_resolves_reference_ids():
    text = "Texto sobre o tema A [1] Texto sobre o tema B [2]"
    refs = [_ref("refA", ["[1]"]), _ref("refB", ["[2]"])]

    chunks = AnswerChunker().chunk(text, answer_id="a1", references=refs)

    assert len(chunks) == 2
    assert chunks[0].cited_reference_ids == ["refA"]
    assert chunks[1].cited_reference_ids == ["refB"]


def test_consecutive_markers_are_grouped_into_one_chunk_boundary():
    text = "Alguma afirmacao apoiada por varias fontes [1][2][3]. Outro trecho."
    refs = [_ref("refA", ["[1]"]), _ref("refB", ["[2]"]), _ref("refC", ["[3]"])]

    chunks = AnswerChunker().chunk(text, answer_id="a1", references=refs)

    assert chunks[0].cited_reference_ids == ["refA", "refB", "refC"]


def test_html_sup_and_unicode_superscript_markers_are_recognized():
    text = "Fato com nota <sup>1</sup> Outro fato com nota unicode¹"
    refs = [_ref("refA", ["[1]"])]

    chunks = AnswerChunker().chunk(text, answer_id="a1", references=refs)

    assert len(chunks) == 2
    assert chunks[0].cited_reference_ids == ["refA"]
    assert chunks[1].cited_reference_ids == ["refA"]


def test_unresolved_marker_yields_empty_cited_references():
    text = "Uma afirmacao citando algo que nao esta na lista de fontes [9]."
    chunks = AnswerChunker().chunk(text, answer_id="a1", references=[])

    assert chunks[0].cited_reference_ids == []


def test_text_without_any_marker_becomes_single_chunk():
    text = "Um paragrafo qualquer sem nenhuma citacao."
    chunks = AnswerChunker().chunk(text, answer_id="a1", references=[])

    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].cited_reference_ids == []


def test_empty_text_yields_no_chunks():
    assert AnswerChunker().chunk("   ", answer_id="a1", references=[]) == []


def test_trailing_text_after_last_marker_becomes_final_chunk_without_references():
    text = "Trecho citado [1]. Trecho final sem citacao."
    refs = [_ref("refA", ["[1]"])]

    chunks = AnswerChunker().chunk(text, answer_id="a1", references=refs)

    assert chunks[-1].cited_reference_ids == []
    assert "Trecho final" in chunks[-1].text
