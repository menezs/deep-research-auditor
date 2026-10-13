from auditframework.extraction.normalize import (
    MarkerStats,
    add_extraction_warnings,
    is_exponent,
    normalize_answer,
    normalize_markers,
    remove_tool_tokens,
)
from auditframework.extraction.reference_extractor import extract_references

PU_OPEN, PU_SEP, PU_CLOSE = "", "", ""


def _bloco(*turns: str) -> str:
    return PU_OPEN + "cite" + "".join(PU_SEP + t for t in turns) + PU_CLOSE


class TestRemoveToolTokens:
    def test_block_form_is_removed_with_the_space_before_it(self):
        texto = f"cadeia de valor da IA.[1][3] {_bloco('turn9view0', 'turn15search0')} A semelhanca"
        limpo, blocos, avulsos = remove_tool_tokens(texto)
        assert limpo == "cadeia de valor da IA.[1][3] A semelhanca"
        assert (blocos, avulsos) == (1, 0)

    def test_block_at_end_of_paragraph_keeps_the_line_break(self):
        texto = f"aprovado em 2024.[4] {_bloco('turn19view0')}\n\nNovo paragrafo [5]."
        limpo, blocos, _ = remove_tool_tokens(texto)
        assert limpo == "aprovado em 2024.[4]\n\nNovo paragrafo [5]."
        assert blocos == 1

    def test_plain_form_without_private_use_chars_is_removed(self):
        texto = "chatbots e geradores [2] citeturn9view2turn10search1turn10search2 seguinte"
        limpo, blocos, _ = remove_tool_tokens(texto)
        assert limpo == "chatbots e geradores [2] seguinte"
        assert blocos == 1

    def test_the_word_cite_in_prose_is_not_touched(self):
        texto = "Cite cada afirmacao; isso pode excitar o leitor."
        assert remove_tool_tokens(texto) == (texto, 0, 0)

    def test_unknown_block_head_keeps_readable_content_and_counts_private_chars(self):
        texto = f"Segundo {PU_OPEN}entity{PU_SEP}Fulano de Tal{PU_CLOSE}, o efeito foi pequeno [1]."
        limpo, blocos, avulsos = remove_tool_tokens(texto)
        assert limpo == "Segundo entityFulano de Tal, o efeito foi pequeno [1]."
        assert (blocos, avulsos) == (0, 3)


class TestNormalizeMarkers:
    def test_grouped_markers_become_individual_brackets(self):
        texto = "geradores de midia sintetica [2, 6, 9]. Deveres do artigo 50 [1, 2, 5, 9]"
        stats = MarkerStats()
        assert normalize_markers(texto, stats) == (
            "geradores de midia sintetica [2][6][9]. Deveres do artigo 50 [1][2][5][9]"
        )
        assert stats.grouped_expanded == 2

    def test_semicolon_and_range_forms_are_expanded(self):
        assert normalize_markers("a [3; 4] b [7-9] c [10–11]") == "a [3][4] b [7][8][9] c [10][11]"

    def test_years_and_huge_ranges_are_not_touched(self):
        texto = "entre [2020, 2021] e a lista [1-100]"
        assert normalize_markers(texto) == texto

    def test_html_sup_and_cite_forms(self):
        assert normalize_markers("fato<sup>1, 2</sup> e [cite_start]outro[cite: 3, 4]") == "fato[1][2] e outro[3][4]"

    def test_unicode_superscripts_are_citations_when_the_body_has_no_brackets(self):
        stats = MarkerStats()
        assert normalize_markers("Fato importante¹ e outro fato¹²", stats) == "Fato importante[1] e outro fato[12]"
        assert stats.superscripts_converted == 2

    def test_unicode_superscripts_are_left_alone_when_the_body_uses_brackets(self):
        texto = "A taxa caiu 1,73 m² por ano [15]. A atencao custa O(n²·d) [1]."
        assert normalize_markers(texto) == texto

    def test_exponent_is_preserved_even_without_brackets(self):
        stats = MarkerStats()
        assert normalize_markers("area de 5 km² segundo o estudo¹", stats) == "area de 5 km² segundo o estudo[1]"
        assert stats.exponents_preserved == 1

    def test_sup_exponent_becomes_unicode_superscript(self):
        assert normalize_markers("1,73 m<sup>2</sup> mais lento") == "1,73 m² mais lento"

    def test_normalization_is_idempotent(self):
        texto = "a [1, 2] b<sup>3</sup> c [cite: 4]"
        uma = normalize_markers(texto)
        assert normalize_markers(uma) == uma


class TestIsExponent:
    def test_units_and_variables(self):
        assert is_exponent("1,73 m", "2")
        assert is_exponent("custa O(n", "2")
        assert is_exponent("O(n·d", "2")
        assert is_exponent("f(x)", "3")

    def test_words_and_other_digits_are_not_exponents(self):
        assert not is_exponent("bilheteira", "2")
        assert not is_exponent("1,73 m", "5")
        assert not is_exponent("A UE", "2")


DUAS_LISTAS = (
    "# Relatorio\n\n"
    "A vacina reduziu casos de cancer cervical na Suecia [1]. "
    "Casos de cancer cervical caem com a vacinacao [2].\n\n"
    "## Referências\n\n"
    "Outra fonte qualquer sobre pagamentos. Disponível em: https://pix.example/a\n"
    "Mais uma fonte sobre bancos. Disponível em: https://banco.example/b\n\n"
    "#### Referências citadas\n\n"
    "Vacina HPV reduz cancer cervical na Suecia, https://vacina.example/suecia\n"
    "Casos de cancer cervical caem com vacinacao, https://cancer.example/casos\n"
)


class TestResolveReferenceLists:
    def test_primary_list_is_chosen_by_evidence_even_when_it_comes_second(self):
        texto, rel = normalize_answer(DUAS_LISTAS)

        assert rel.primary_list_choice == "evidence"
        assert [i.role for i in rel.reference_lists] == ["secondary", "primary"]
        refs = {r.normalized_url: r for r in extract_references(texto, source_answer_id="a", tool_name="t")}
        assert refs["https://vacina.example/suecia"].citation_markers == ["[1]"]
        assert refs["https://cancer.example/casos"].citation_markers == ["[2]"]
        assert refs["https://pix.example/a"].from_secondary_list is True

    def test_canonical_output_has_one_primary_list_and_the_asterism(self):
        texto, _ = normalize_answer(DUAS_LISTAS)
        assert texto.count("## Referências") == 1
        assert "[1] Vacina HPV reduz cancer cervical na Suecia, https://vacina.example/suecia" in texto
        assert "⁂" in texto

    def test_without_evidence_falls_back_to_the_last_list_and_warns(self):
        texto = (
            "Uma afirmacao qualquer [1].\n\n"
            "## Referências\n\nAlfa beta, https://um.example/1\n\n"
            "## Fontes\n\nGama delta, https://dois.example/2\n"
        )
        saida, rel = normalize_answer(texto)

        assert rel.primary_list_choice == "position"
        assert [i.role for i in rel.reference_lists] == ["secondary", "primary"]
        assert any("posicao" in w for w in rel.warnings)
        refs = {r.normalized_url: r for r in extract_references(saida, source_answer_id="a", tool_name="t")}
        assert refs["https://dois.example/2"].citation_markers == ["[1]"]

    def test_existing_asterism_layout_is_left_untouched(self):
        texto = "Corpo [1].\n\n## References\n\n[1] Titulo, https://a.example/x\n\n⁂\n\n1. https://b.example/y\n"
        saida, rel = normalize_answer(texto)
        assert saida == texto
        assert rel.primary_list_choice == "asterism"

    def test_single_list_is_left_untouched(self):
        texto = "Corpo [1].\n\n## Referências\n\nTitulo, https://a.example/x\n"
        saida, rel = normalize_answer(texto)
        assert saida == texto
        assert rel.primary_list_choice == "single"

    def test_content_after_the_lists_is_preserved(self):
        texto = DUAS_LISTAS + "\n## Apendice\n\nNota final sem fonte.\n"
        saida, _ = normalize_answer(texto)
        assert "## Apendice\n\nNota final sem fonte." in saida

    def test_body_without_markers_is_not_rewritten_and_warns(self):
        texto = DUAS_LISTAS.replace(" [1]", "").replace(" [2]", "")
        saida, rel = normalize_answer(texto)
        assert saida == texto
        assert rel.body_markers == 0
        assert any("sem marcadores" in w for w in rel.warnings)


class TestNormalizeAnswer:
    def test_report_counts_every_transformation(self):
        texto = (
            "Afirmacao A [1, 2]. citeturn9view0 Afirmacao B<sup>3</sup>.\n\n"
            "## Referências\n\n[1] A, https://a.example\n[2] B, https://b.example\n[3] C, https://c.example\n"
        )
        saida, rel = normalize_answer(texto)

        assert "" not in saida and "[1][2]" in saida and "B[3]" in saida
        assert (rel.tool_tokens_removed, rel.grouped_markers_expanded, rel.superscripts_converted) == (1, 1, 1)
        assert rel.body_markers == 3

    def test_is_idempotent(self):
        uma, _ = normalize_answer(DUAS_LISTAS)
        duas, _ = normalize_answer(uma)
        assert duas == uma

    def test_extraction_warnings_list_markers_without_entry(self):
        _, rel = normalize_answer("Corpo [1] e [9].\n\n## Referências\n\n[1] A, https://a.example\n")
        add_extraction_warnings(rel, cited_markers={"[1]", "[9]"}, listed_markers={"[1]"})
        assert rel.body_markers_without_entry == ["[9]"]
        assert any("sem entrada" in w for w in rel.warnings)
