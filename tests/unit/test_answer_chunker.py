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


def test_sentence_count_flags_multi_claim_chunks():
    text = (
        "Primeira afirmação sem fonte. Segunda afirmação sem fonte. "
        "Terceira afirmação, agora citada [1]"
    )
    chunks = AnswerChunker().chunk(text, answer_id="a1", references=[_ref("refA", ["[1]"])])

    anchored = [c for c in chunks if c.cited_reference_ids]
    assert len(anchored) == 1
    assert anchored[0].cited_reference_ids == ["refA"]
    assert anchored[0].sentence_count == 3


def test_single_sentence_chunk_has_sentence_count_one():
    chunks = AnswerChunker().chunk("Uma afirmação só [1]", answer_id="a1", references=[_ref("refA", ["[1]"])])
    assert chunks[0].sentence_count == 1


def test_uncited_paragraph_before_a_citation_is_split_off_as_uncited_claim():
    """Regressao samsung: um parágrafo sem marcador que precede um parágrafo
    citado NÃO deve ser julgado contra a citação do vizinho."""
    text = (
        "A Samsung foi fundada em 1938 por Lee Byung-chul. [1]\n\n"
        "Com o tempo, a companhia investiu fortemente em semicondutores, telas e "
        "telefones móveis, tornando-se uma marca global.\n\n"
        "Hoje é um dos maiores conglomerados da Coreia do Sul. [2]"
    )
    refs = [_ref("refA", ["[1]"]), _ref("refB", ["[2]"])]

    chunks = AnswerChunker().chunk(text, answer_id="a1", references=refs)

    cited = {tuple(c.cited_reference_ids): c for c in chunks if c.cited_reference_ids}
    assert set(cited) == {("refA",), ("refB",)}
    # o parágrafo dos semicondutores fica isolado, sem citação
    uncited = [c for c in chunks if c.is_uncited_claim]
    assert len(uncited) == 1
    assert "semicondutores" in uncited[0].text
    assert uncited[0].cited_reference_ids == []
    # e o chunk citado por [2] é só a última frase, não o parágrafo anterior
    assert "semicondutores" not in cited[("refB",)].text
    assert "conglomerados" in cited[("refB",)].text


def test_same_paragraph_sentences_stay_with_the_trailing_citation():
    """Frases do MESMO parágrafo que a citação continuam ancoradas por ela
    (não viram 'sem citação')."""
    text = "Primeira frase da ideia. Segunda frase, agora com fonte. [1]"
    chunks = AnswerChunker().chunk(text, answer_id="a1", references=[_ref("refA", ["[1]"])])

    assert len(chunks) == 1
    assert chunks[0].cited_reference_ids == ["refA"]
    assert chunks[0].is_uncited_claim is False
    assert "Primeira frase" in chunks[0].text


def test_heading_and_tag_artifacts_are_dropped_not_flagged():
    text = "</u>\n\n## Um cabeçalho qualquer\n\nNas décadas seguintes a empresa cresceu muito. [1]"
    chunks = AnswerChunker().chunk(text, answer_id="a1", references=[_ref("refA", ["[1]"])])

    assert len(chunks) == 1
    assert chunks[0].cited_reference_ids == ["refA"]
    assert "cabeçalho" not in chunks[0].text
    assert "</u>" not in chunks[0].text


def test_markdown_tables_are_linearized_not_fragmented():
    text = (
        "Introdução em prosa normal. [1]\n\n"
        "|**Ano**|**Evento**|**Significado**|\n"
        "|---|---|---|\n"
        "|**1938**|Fundação da Samsung<br>em Taegu<sup>2</sup>|Início como casa comercial<sup>2</sup>.|\n"
        "|**1969**|Criação da<br>Samsung Electronics<sup>3</sup>|Entrada no setor eletrónico<sup>3</sup>.|\n\n"
        "Parágrafo final. [1]"
    )
    refs = [_ref("refA", ["[1]"]), _ref("refB", ["[2]"]), _ref("refC", ["[3]"])]

    chunks = AnswerChunker().chunk(text, answer_id="a1", references=refs)

    # nenhum chunk com lixo de tabela
    for c in chunks:
        assert "|" not in c.text and "<br>" not in c.text and not c.text.startswith((".", "—"))
    # a linha da tabela virou um chunk coerente e citado
    row = next(c for c in chunks if "1938" in c.text)
    assert "Fundação da Samsung" in row.text
    assert row.cited_reference_ids == ["refB"]
    # o cabeçalho da tabela (sem citação, curto) foi descartado, não virou "sem citação"
    assert not any("Significado" in c.text for c in chunks)
    assert not any(c.is_uncited_claim for c in chunks)
