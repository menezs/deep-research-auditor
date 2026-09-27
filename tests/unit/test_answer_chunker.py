import pytest

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


def test_text_without_any_marker_yields_no_chunks():
    """Sem citação não há relação afirmação-fonte para auditar."""
    text = "Um paragrafo qualquer sem nenhuma citacao."
    assert AnswerChunker().chunk(text, answer_id="a1", references=[]) == []


def test_empty_text_yields_no_chunks():
    assert AnswerChunker().chunk("   ", answer_id="a1", references=[]) == []


def test_trailing_text_after_the_last_marker_is_dropped():
    text = "Trecho citado [1]. Trecho final sem citacao."
    refs = [_ref("refA", ["[1]"])]

    chunks = AnswerChunker().chunk(text, answer_id="a1", references=refs)

    assert len(chunks) == 1
    assert chunks[0].cited_reference_ids == ["refA"]
    assert "Trecho final" not in chunks[0].text


def test_citation_anchors_only_its_own_sentence_not_the_whole_paragraph():
    """O marcador ancora a frase em que aparece; as frases anteriores do
    mesmo parágrafo não entram no trecho julgado — elas não têm fonte."""
    text = (
        "Primeira afirmação sem fonte. Segunda afirmação sem fonte. "
        "Terceira afirmação, agora citada [1]"
    )
    chunks = AnswerChunker().chunk(text, answer_id="a1", references=[_ref("refA", ["[1]"])])

    assert len(chunks) == 1
    assert chunks[0].cited_reference_ids == ["refA"]
    assert chunks[0].text == "Terceira afirmação, agora citada"
    assert "Primeira afirmação" not in chunks[0].text


def test_uncited_paragraph_before_a_citation_is_never_judged_against_it():
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

    cited = {tuple(c.cited_reference_ids): c for c in chunks}
    assert set(cited) == {("refA",), ("refB",)}
    # o parágrafo dos semicondutores não tem citação: não vira chunk nenhum
    assert not any("semicondutores" in c.text for c in chunks)
    assert "conglomerados" in cited[("refB",)].text


def test_uncited_sentence_inside_a_cited_paragraph_is_dropped():
    """A frase sem marcador sai do trecho julgado e não vira chunk — o
    fatiamento só por parágrafo colava as duas num único trecho."""
    text = "Primeira frase da ideia. Segunda frase, agora com fonte. [1]"
    chunks = AnswerChunker().chunk(text, answer_id="a1", references=[_ref("refA", ["[1]"])])

    assert [c.text for c in chunks] == ["Segunda frase, agora com fonte."]
    assert chunks[0].cited_reference_ids == ["refA"]


def test_marker_glued_to_the_period_anchors_the_sentence_that_just_ended():
    """Em "frase.[4]" o ponto encosta no marcador: a fronteira de sentença
    não pode cortar ali, senão o trecho sairia vazio."""
    text = "Ensaios com redução de jornada mostram ganho de bem-estar.[4]"
    chunks = AnswerChunker().chunk(text, answer_id="a1", references=[_ref("refD", ["[4]"])])

    assert len(chunks) == 1
    assert chunks[0].cited_reference_ids == ["refD"]
    assert chunks[0].text == "Ensaios com redução de jornada mostram ganho de bem-estar."


def test_abbreviation_and_decimal_periods_do_not_split_the_claim():
    """`art.`/`Reg.`/`1.238` não são fim de frase — o regex ingênuo anterior
    cortava o trecho no meio."""
    text = "A UE (Reg. UE 2024/1689) e o art. 1.238 do CC tratam de risco sistêmico [1]"
    chunks = AnswerChunker().chunk(text, answer_id="a1", references=[_ref("refA", ["[1]"])])

    assert len(chunks) == 1
    assert chunks[0].text.startswith("A UE (Reg. UE 2024/1689)")
    assert "art. 1.238" in chunks[0].text


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
    # o cabeçalho da tabela (sem citação) foi descartado
    assert not any("Significado" in c.text for c in chunks)
    # todo chunk emitido cita alguma referência
    assert all(c.cited_reference_ids for c in chunks)


def test_marker_without_a_reference_entry_still_becomes_a_chunk():
    """O trecho tem citação — só não há entrada na lista para ela. Continua
    sendo um chunk (vira SKIPPED no julgamento, com esse motivo), diferente
    de um trecho que não cita nada."""
    chunks = AnswerChunker().chunk(
        "Uma afirmação citando algo fora da lista [9].", answer_id="a1", references=[]
    )
    assert len(chunks) == 1
    assert chunks[0].cited_reference_ids == []
    assert chunks[0].text == "Uma afirmação citando algo fora da lista."


def test_chunk_records_the_marker_it_actually_uses():
    """A mesma URL pode estar listada sob dois números (`[8]` e `[10]`). O
    trecho tem de registrar o que ele cita, senão o relatório mostra o
    primeiro número da referência em vez do citado."""
    ref = Reference(
        id="refA", citation_markers=["[8]", "[10]"],
        raw_url="https://example.com/a", normalized_url="https://example.com/a",
        source_answer_id="a1", tool_name="Perplexity",
    )
    chunks = AnswerChunker().chunk(
        "Primeira afirmação [8]. Segunda afirmação [10].", answer_id="a1", references=[ref]
    )

    assert [c.cited_markers for c in chunks] == [["[8]"], ["[10]"]]
    assert all(c.cited_reference_ids == ["refA"] for c in chunks)


def test_consecutive_markers_are_all_recorded():
    refs = [_ref("refA", ["[1]"]), _ref("refB", ["[2]"])]
    chunks = AnswerChunker().chunk("Afirmação apoiada por duas fontes [1][2].", answer_id="a1", references=refs)
    assert chunks[0].cited_markers == ["[1]", "[2]"]


def test_unresolved_marker_is_recorded_even_without_a_reference():
    chunks = AnswerChunker().chunk("Cita algo fora da lista [9].", answer_id="a1", references=[])
    assert chunks[0].cited_markers == ["[9]"] and chunks[0].cited_reference_ids == []


class TestRobustnessAcrossInputShapes:
    """Nenhuma forma de entrada pode provocar erro, e todo chunk emitido tem
    de carregar marcador — é a invariante que o resto do pipeline assume."""

    SHAPES = [
        "",
        "   \n\n\t ",
        "Prosa sem citação nenhuma.",
        "[1]",
        "[1][2][3]",
        "[1] no início da frase.",
        "Frase.[1]",
        "## Só um cabeçalho [1]",
        "|a|b|\n|---|---|\n|1[1]|2|\n",
        "Texto com <sup>3</sup> e ¹ juntos.",
        "[cite_start]Texto do Gemini[cite: 12, 13]",
        "Afirmação [ 5 ] com espaço no marcador.",
        "Afirmação [0] com zero.",
        "Afirmação [999999] com número gigante.",
        "art. 1.238 e Reg. UE 2024/1689 [1]",
        "Aspas “abertas e não fechadas [1]",
        "(parêntese não fechado [1]",
        "Frase sem fim [1",
        "\n\n\n[1]\n\n\n",
        "—— [1]",
    ]

    @pytest.mark.parametrize("text", SHAPES, ids=range(len(SHAPES)))
    def test_never_raises_and_every_chunk_has_a_marker(self, text):
        refs = [_ref("refA", ["[1]"]), _ref("refB", ["[2]"]), _ref("refC", ["[3]"])]
        chunks = AnswerChunker().chunk(text, answer_id="a1", references=refs)

        for chunk in chunks:
            assert chunk.cited_markers, f"chunk sem marcador: {chunk.text!r}"
            assert chunk.text.strip() == chunk.text and chunk.text
        positions = [c.position for c in chunks]
        assert positions == list(range(len(chunks)))


class TestSentenceTail:
    """O marcador também sustenta o RESTO da frase dele. Sem isso, uma
    citação no meio da frase ficava só com o pedaço anterior a ela, e a parte
    que qualifica o dado era descartada por não ter marcador próprio."""

    def test_mid_sentence_marker_recovers_the_rest_of_the_sentence(self):
        text = (
            "HPV vaccination reduces cervical cancer by 60–90%, with the greatest protection when [1] "
            "vaccination occurs before age 17."
        )
        chunks = AnswerChunker().chunk(text, answer_id="a1", references=[_ref("refA", ["[1]"])])

        assert len(chunks) == 1
        assert chunks[0].text.endswith("vaccination occurs before age 17.")
        assert "protection when" in chunks[0].text

    def test_marker_before_the_final_period_recovers_the_period(self):
        chunks = AnswerChunker().chunk(
            "Uma afirmação qualquer [1]. Outra coisa [2].", answer_id="a1",
            references=[_ref("refA", ["[1]"]), _ref("refB", ["[2]"])],
        )
        assert [c.text for c in chunks] == ["Uma afirmação qualquer.", "Outra coisa."]

    def test_two_markers_in_one_sentence_do_not_overlap(self):
        """A cauda entre os dois pertence ao segundo, pela regra de
        walk-back — o primeiro não pode levá-la."""
        chunks = AnswerChunker().chunk(
            "Vale para A [1] e também para B [2].", answer_id="a1",
            references=[_ref("refA", ["[1]"]), _ref("refB", ["[2]"])],
        )
        assert [c.text for c in chunks] == ["Vale para A", "e também para B."]

    def test_marker_glued_to_a_finished_sentence_does_not_absorb_the_next(self):
        chunks = AnswerChunker().chunk(
            "Primeira frase termina aqui.[4] Segunda frase é outra coisa.", answer_id="a1",
            references=[_ref("refD", ["[4]"])],
        )
        assert len(chunks) == 1
        assert chunks[0].text == "Primeira frase termina aqui."
        assert "Segunda frase" not in chunks[0].text

    def test_marker_on_the_sentence_boundary_keeps_its_own_chunk(self):
        """Regressão: em "...quando [1] a frase termina.[2]" o `[2]` fica
        exatamente na fronteira. Levar a cauda para o `[1]` deixava o `[2]`
        sem trecho e apagava uma citação da auditoria."""
        chunks = AnswerChunker().chunk(
            "O estudo mostrou que [1] a redução foi consistente.[2] Outro ponto [3].", answer_id="a1",
            references=[_ref("refA", ["[1]"]), _ref("refB", ["[2]"]), _ref("refC", ["[3]"])],
        )
        cited = [c.cited_reference_ids for c in chunks]
        assert ["refB"] in cited, "o [2] perdeu o chunk dele"
        assert ["refA"] in cited and ["refC"] in cited

    def test_marker_at_the_start_of_a_line_stays_discarded(self):
        """Regressão: marcador sem NADA antes dele é entrada de lista de
        fontes (`[1] Título da obra.`), não afirmação. A cauda não pode
        ressuscitar esses trechos — e a checagem de conteúdo precisa ignorar
        as tags, porque `<u>` tem uma letra dentro."""
        for text in ("[1] World Journal of Oncology.", "> <u>[1]</u> Frontiers | Cervical cancer review"):
            chunks = AnswerChunker().chunk(text, answer_id="a1", references=[_ref("refA", ["[1]"])])
            assert chunks == [], f"nao deveria emitir chunk para {text!r}"
