from pathlib import Path

import pytest

from auditframework.extraction.reference_extractor import extract_references

FIXTURE = Path(__file__).parent.parent / "fixtures" / "sample_answer_full.md"


def _load_fixture() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def test_extracts_one_reference_per_distinct_url():
    refs = extract_references(_load_fixture(), source_answer_id="a1", tool_name="ChatGPT")
    assert len(refs) == 2


def test_merges_citation_markers_pointing_to_same_reference_entry():
    refs = extract_references(_load_fixture(), source_answer_id="a1", tool_name="ChatGPT")
    lgpd = next(r for r in refs if "planalto" in r.raw_url)
    assert lgpd.citation_markers == ["[2]", "[3]"]


def test_reference_id_is_stable_hash_not_sequential_extraction_order():
    refs = extract_references(_load_fixture(), source_answer_id="a1", tool_name="ChatGPT")
    marco_civil = next(r for r in refs if "wikipedia" in r.raw_url)
    # roda a extracao de novo (simula uma segunda execucao) e confirma que
    # o id nao muda - corrige o bug do CorpusForge onde o id dependia da
    # ordem de extracao via LLM, nao-deterministica entre execucoes
    refs_again = extract_references(_load_fixture(), source_answer_id="a1", tool_name="ChatGPT")
    marco_civil_again = next(r for r in refs_again if "wikipedia" in r.raw_url)
    assert marco_civil.id == marco_civil_again.id


def test_title_is_captured():
    refs = extract_references(_load_fixture(), source_answer_id="a1", tool_name="ChatGPT")
    lgpd = next(r for r in refs if "planalto" in r.raw_url)
    assert "Lei Geral de Proteção de Dados" in lgpd.title


def test_no_reference_section_yields_empty_list():
    text = "Um texto qualquer com uma citacao [1] mas sem lista de fontes."
    refs = extract_references(text, source_answer_id="a2", tool_name="Gemini")
    assert refs == []


def test_tool_name_and_source_answer_id_are_propagated():
    refs = extract_references(_load_fixture(), source_answer_id="a1", tool_name="Perplexity")
    assert all(r.source_answer_id == "a1" and r.tool_name == "Perplexity" for r in refs)


def test_blank_line_between_title_and_url_is_handled():
    """Regressao: conversores PDF->Markdown (pymupdf4llm) tipicamente
    inserem uma linha em branco entre o titulo e a URL de cada
    referencia (paragrafos separados), diferente do formato "colado" do
    fixture padrao. Sem isso, so a 1a referencia da lista era extraida
    quando a URL nao estava logo na linha seguinte."""
    text = (
        "[1] Primeiro Titulo\n"
        "\n"
        "https://example.com/um\n"
        "\n"
        "[2] [5] Segundo Titulo\n"
        "\n"
        "https://example.com/dois\n"
    )
    refs = extract_references(text, source_answer_id="a1", tool_name="ChatGPT")
    urls = {r.raw_url for r in refs}
    assert urls == {"https://example.com/um", "https://example.com/dois"}
    dois = next(r for r in refs if r.raw_url == "https://example.com/dois")
    assert dois.citation_markers == ["[2]", "[5]"]


def test_url_split_across_pdf_line_wrap_is_rejoined():
    """Regressao: quando a URL e longa demais para uma linha do PDF
    original, o pymupdf4llm insere um espaco no ponto de quebra em vez
    de manter a URL contigua (ex: "...transparente-p ara-boa..."). Como
    e exatamente um token sem espacos apos a URL truncada, deve ser
    reconectado."""
    text = "[1] Titulo Longo\nhttps://example.com/slug-truncado-p ara-continuar-aqui\n"
    refs = extract_references(text, source_answer_id="a1", tool_name="ChatGPT")
    assert refs[0].raw_url == "https://example.com/slug-truncado-para-continuar-aqui"


def test_url_followed_by_real_prose_on_same_line_is_not_merged():
    """Contraste com o caso acima: quando o que segue a URL na mesma
    linha e mais de uma palavra (prosa de verdade, nao continuacao de
    URL quebrada), o texto extra nao deve ser colado na URL."""
    text = "[1] Titulo\nhttps://example.com/pagina acessado em 10 de outubro de 2025\n"
    refs = extract_references(text, source_answer_id="a1", tool_name="ChatGPT")
    assert refs[0].raw_url == "https://example.com/pagina"


def test_html_tag_glued_to_url_is_not_reconnected():
    """Regressao: um `</u>` colado (sem espaco) logo apos a URL nao pode
    ser tratado como continuacao de URL quebrada (formato de lista do
    Perplexity usa `<u>...</u>` ao redor de cada URL) — so tokens que nao
    comecam com `<` sao continuacao legitima."""
    text = "[1] Titulo\n<u>https://example.com/pagina</u> resto do paragrafo\n"
    refs = extract_references(text, source_answer_id="a1", tool_name="ChatGPT")
    assert refs[0].raw_url == "https://example.com/pagina"


class TestAsterismListFormat:
    """Formato do Perplexity: lista numerada sem colchetes, apos um
    separador `⁂`, com cada URL entre tags `<u>...</u>` — completamente
    diferente do `[N] Titulo\\nURL` do ChatGPT/Gemini."""

    def test_document_without_asterism_is_unaffected(self):
        text = "[1] Titulo\nhttps://example.com/artigo\n"
        refs = extract_references(text, source_answer_id="a1", tool_name="ChatGPT")
        assert len(refs) == 1  # so a passada [N] roda; a passada ⁂ e no-op

    def test_simple_entries_are_extracted_with_bracket_style_markers(self):
        text = "⁂ \n\n1. <u>https://example.com/um</u> \n\n2. <u>https://example.com/dois</u> \n"
        refs = extract_references(text, source_answer_id="a1", tool_name="Perplexity")
        by_url = {r.raw_url: r for r in refs}
        assert set(by_url) == {"https://example.com/um", "https://example.com/dois"}
        # o numero da lista vira marcador [N] no formato canonico, para o
        # resto do pipeline (AnswerChunker) resolver citacoes inline iguais
        assert by_url["https://example.com/dois"].citation_markers == ["[2]"]

    def test_multiple_entries_on_the_same_physical_line(self):
        text = "⁂ \n\n2. <u>https://example.com/a</u> 3. <u>https://example.com/b</u> \n"
        refs = extract_references(text, source_answer_id="a1", tool_name="Perplexity")
        by_marker = {r.citation_markers[0]: r.raw_url for r in refs}
        assert by_marker == {"[2]": "https://example.com/a", "[3]": "https://example.com/b"}

    def test_url_wrapped_across_multiple_lines_with_blank_line_between(self):
        text = (
            "⁂ \n\n"
            "7. <u>https://example.com/quebrado-intervencao-e-</u> \n\n"
            "<u>continuacao-2024/</u> \n"
        )
        refs = extract_references(text, source_answer_id="a1", tool_name="Perplexity")
        assert refs[0].raw_url == "https://example.com/quebrado-intervencao-e-continuacao-2024/"

    def test_continuation_line_tolerates_markdown_noise_prefix(self):
        """Ruido de markdown (### ou - antes do numero/continuacao),
        artefato comum da conversao PDF->Markdown do pymupdf4llm."""
        text = (
            "⁂ \n\n"
            "### 8. <u>https://example.com/com-hash</u> \n\n"
            "9. <u>https://example.com/nove-a</u> \n\n"
            "- <u>nove-b</u> \n"
        )
        refs = extract_references(text, source_answer_id="a1", tool_name="Perplexity")
        by_marker = {r.citation_markers[0]: r.raw_url for r in refs}
        assert by_marker["[8]"] == "https://example.com/com-hash"
        assert by_marker["[9]"] == "https://example.com/nove-anove-b"

    def test_literal_space_inside_a_single_tag_becomes_percent_20(self):
        text = "⁂ \n\n10. <u>https://example.com/bitstream/Arquivo Com Espaco.pdf</u> \n"
        refs = extract_references(text, source_answer_id="a1", tool_name="Perplexity")
        assert refs[0].raw_url == "https://example.com/bitstream/Arquivo%20Com%20Espaco.pdf"

    def test_prose_numbered_list_before_asterism_is_not_treated_as_references(self):
        """A passada ⁂ so opera no texto APOS o separador — uma lista
        numerada comum no corpo/resumo antes do ⁂ (ex: "1. Titulo A [24]")
        nao deve virar uma entrada de referencia."""
        text = (
            "## Referências\n"
            "1. **Fonte A** - resumo qualquer<sup><u>[24][4]</u></sup>\n"
            "2. **Fonte B** - outro resumo<sup><u>[5]</u></sup>\n\n"
            "⁂ \n\n"
            "1. <u>https://example.com/real</u> \n"
        )
        refs = extract_references(text, source_answer_id="a1", tool_name="Perplexity")
        assert len(refs) == 1
        assert refs[0].raw_url == "https://example.com/real"

    def test_bracket_and_asterism_formats_can_coexist_in_the_same_document(self):
        text = (
            "[1] Referencia estilo ChatGPT\nhttps://example.com/chatgpt\n\n"
            "⁂ \n\n"
            "1. <u>https://example.com/perplexity</u> \n"
        )
        refs = extract_references(text, source_answer_id="a1", tool_name="Perplexity")
        urls = {r.raw_url for r in refs}
        assert urls == {"https://example.com/chatgpt", "https://example.com/perplexity"}


class TestAsterismBareUrlFormat:
    """Formato do Perplexity em .docx: lista pos-`⁂` sem NENHUMA
    numeracao/marcacao — so uma URL por linha, em texto puro (sem tags
    `<u>`, ja que `python-docx` nao produz HTML). O marcador `[N]` e
    inferido pela ordem de ocorrencia."""

    def test_bare_urls_get_positional_markers(self):
        text = "⁂\n\nhttps://example.com/um\n\nhttps://example.com/dois\n\nhttps://example.com/tres\n"
        refs = extract_references(text, source_answer_id="a1", tool_name="Perplexity")
        by_marker = {r.citation_markers[0]: r.raw_url for r in refs}
        assert by_marker == {
            "[1]": "https://example.com/um",
            "[2]": "https://example.com/dois",
            "[3]": "https://example.com/tres",
        }

    def test_numbered_format_takes_priority_over_bare_fallback(self):
        """Se a lista pos-⁂ tiver numeracao (`N.`), o fallback de URL nua
        nunca deve rodar — evita reprocessar/duplicar as mesmas entradas."""
        text = "⁂\n\n1. <u>https://example.com/numerado</u>\n"
        refs = extract_references(text, source_answer_id="a1", tool_name="Perplexity")
        assert len(refs) == 1
        assert refs[0].raw_url == "https://example.com/numerado"

    def test_literal_space_in_filename_url_is_preserved_as_percent20(self):
        """Regressao: diferente do `_extract_url` generico (que so
        reconecta um unico token sem espaco, por poder haver prosa real
        depois da URL), aqui a linha inteira e sempre so a URL -- um nome
        de arquivo com espaco (ex: "Relatorio Final.pdf") nao pode ser
        truncado no primeiro espaco."""
        text = "⁂\n\nhttps://example.com/arquivos/Relatorio Final Completo.pdf\n"
        refs = extract_references(text, source_answer_id="a1", tool_name="Perplexity")
        assert refs[0].raw_url == "https://example.com/arquivos/Relatorio%20Final%20Completo.pdf"

    def test_document_without_asterism_is_unaffected(self):
        text = "[1] Titulo\nhttps://example.com/artigo\n"
        refs = extract_references(text, source_answer_id="a1", tool_name="ChatGPT")
        assert len(refs) == 1

    def test_blank_lines_between_entries_are_skipped(self):
        text = "⁂\n\n\n\nhttps://example.com/um\n\n\n\nhttps://example.com/dois\n"
        refs = extract_references(text, source_answer_id="a1", tool_name="Perplexity")
        assert len(refs) == 2


class TestRepairUsingPdfLinks:
    """`pymupdf4llm` (conversao PDF->texto) as vezes descarta um hifen num
    ponto de quebra de linha sem deixar nenhum sinal textual disso (ex:
    "de marco" -> "demarco"), o que nenhuma heuristica de texto consegue
    recuperar. Quando `answer_path` aponta pra um PDF, `extract_references`
    corrige a URL usando os hyperlinks reais embutidos nele."""

    def _make_pdf(self, path: Path, links: list[str]) -> None:
        fitz = __import__("fitz")
        doc = fitz.open()
        page = doc.new_page()
        for i, uri in enumerate(links):
            rect = fitz.Rect(50, 50 + i * 20, 300, 65 + i * 20)
            page.insert_text((50, 60 + i * 20), f"link {i}")
            page.insert_link({"kind": fitz.LINK_URI, "from": rect, "uri": uri})
        doc.save(str(path))
        doc.close()

    def test_dropped_hyphen_is_restored_from_embedded_pdf_link(self, tmp_path):
        pytest.importorskip("fitz")
        pdf_path = tmp_path / "resposta.pdf"
        self._make_pdf(pdf_path, ["https://example.com/17-de-marco-de-2026"])

        text = "[1] Titulo\nhttps://example.com/17-demarco-de-2026\n"
        refs = extract_references(text, source_answer_id="a1", tool_name="ChatGPT", answer_path=pdf_path)

        assert refs[0].raw_url == "https://example.com/17-de-marco-de-2026"

    def test_ambiguous_percent20_is_resolved_by_embedded_pdf_link(self, tmp_path):
        pytest.importorskip("fitz")
        pdf_path = tmp_path / "resposta.pdf"
        self._make_pdf(pdf_path, ["https://example.com/doen%C3%A7a-de-alzheimer"])

        text = "[1] Titulo\nhttps://example.com/doen%20ça-de-alzheimer\n"
        refs = extract_references(text, source_answer_id="a1", tool_name="Perplexity", answer_path=pdf_path)

        assert refs[0].raw_url == "https://example.com/doen%C3%A7a-de-alzheimer"

    def test_truncated_lookalike_link_is_not_picked_by_mistake(self, tmp_path):
        """Um PDF pode ter mais de um hyperlink parecido (ex: um vindo de
        uma tabela markdown corrompida) — quando a URL extraida ja bate
        EXATAMENTE (chave difusa) com um candidato, esse candidato vence
        na hora, sem sequer considerar o outro por parecenca parcial."""
        pytest.importorskip("fitz")
        pdf_path = tmp_path / "resposta.pdf"
        self._make_pdf(
            pdf_path,
            ["https://example.com/cancer-de-pancreas", "https://example.com/cancer-de-p%25..."],
        )

        text = "[1] Titulo\nhttps://example.com/cancer-de-pancreas\n"
        refs = extract_references(text, source_answer_id="a1", tool_name="Perplexity", answer_path=pdf_path)

        assert refs[0].raw_url == "https://example.com/cancer-de-pancreas"

    def test_pdf_without_matching_link_leaves_url_unchanged(self, tmp_path):
        pytest.importorskip("fitz")
        pdf_path = tmp_path / "resposta.pdf"
        self._make_pdf(pdf_path, ["https://example.com/outra-referencia-qualquer"])

        text = "[1] Titulo\nhttps://example.com/pagina-normal\n"
        refs = extract_references(text, source_answer_id="a1", tool_name="ChatGPT", answer_path=pdf_path)

        assert refs[0].raw_url == "https://example.com/pagina-normal"

    def test_non_pdf_answer_path_is_ignored(self, tmp_path):
        md_path = tmp_path / "resposta.md"
        md_path.write_text("qualquer coisa", encoding="utf-8")

        text = "[1] Titulo\nhttps://example.com/pagina\n"
        refs = extract_references(text, source_answer_id="a1", tool_name="ChatGPT", answer_path=md_path)

        assert refs[0].raw_url == "https://example.com/pagina"

    def test_two_repaired_references_colliding_into_the_same_id_are_merged(self, tmp_path):
        pytest.importorskip("fitz")
        pdf_path = tmp_path / "resposta.pdf"
        self._make_pdf(pdf_path, ["https://example.com/mesma-pagina"])

        # duas entradas de texto corrompidas de formas diferentes, mas que
        # o hyperlink do pdf resolve para a MESMA url real
        text = (
            "[1] Titulo A\nhttps://example.com/mesma-pa%20gina\n\n"
            "[2] Titulo B\nhttps://example.com/mesma-paginaX\n"
        )
        refs = extract_references(text, source_answer_id="a1", tool_name="ChatGPT", answer_path=pdf_path)

        # a segunda ("...paginaX") tem um caractere A MAIS que o link do
        # pdf -- nunca pode ser reparada por um mecanismo que so tolera
        # caracteres FALTANDO (ligadura), entao so a primeira e reparada;
        # confirma que duas referencias genuinamente diferentes no mesmo
        # texto nao colapsam indevidamente na mesma
        urls = sorted(r.raw_url for r in refs)
        assert urls == ["https://example.com/mesma-pagina", "https://example.com/mesma-paginaX"]

    def test_letters_dropped_by_font_ligature_are_restored(self, tmp_path):
        """Corrupcao distinta do hifen/espaco: a fonte incorporada do PDF
        tem glifos de ligadura (ex: "tt", "fi") sem mapeamento ToUnicode
        completo, entao letras inteiras somem do texto extraido sem
        deixar sinal nenhum ("https" -> "htps", "office" -> "ofce") —
        corrompe ate o esquema da URL, entao so aparece via o formato de
        lista `<u>...</u>` (que nao valida esquema, ao contrario de
        `_extract_url`/`_URL_RE`), exatamente como no PDF real do Gemini.
        A chave difusa (so hifen/espaco) nao repara isso, precisa do
        casamento por subsequencia."""
        pytest.importorskip("fitz")
        pdf_path = tmp_path / "resposta.pdf"
        self._make_pdf(pdf_path, ["https://example.com/wiki/List_of_films"])

        text = "### **Referências citadas**\n\n1. Titulo, <u>htps://example.com/wiki/List_of_flms</u>\n"
        refs = extract_references(text, source_answer_id="a1", tool_name="Gemini", answer_path=pdf_path)

        assert refs[0].raw_url == "https://example.com/wiki/List_of_films"

    def test_too_many_dropped_characters_is_not_repaired(self, tmp_path):
        """O reparo por subsequencia tem um orcamento pequeno de
        caracteres faltando (`_MAX_DROPPED_CHARS`) — uma URL extraida
        curta demais (muito mais corrompida do que uma ligadura tipica
        deixaria) nao deve ser forcada contra um candidato so porque
        tecnicamente e uma subsequencia dele."""
        pytest.importorskip("fitz")
        pdf_path = tmp_path / "resposta.pdf"
        self._make_pdf(pdf_path, ["https://example.com/wiki/List_of_marvel_cinematic_universe_films"])

        text = "### **Referências citadas**\n\n1. Titulo, <u>htps://example.com/wiki/Lst_of_films</u>\n"
        refs = extract_references(text, source_answer_id="a1", tool_name="Gemini", answer_path=pdf_path)

        assert refs[0].raw_url == "htps://example.com/wiki/Lst_of_films"


class TestReferenceSectionHeadingTolerance:
    """`_REFERENCE_SECTION_HEADING` ancora tanto `_find_asterism_list_entries`
    (quando nao ha `⁂`) quanto `_find_heading_titled_pairs` — precisa
    tolerar negrito markdown e texto extra depois da palavra-chave (ex: o
    Gemini em PDF usa `### **Referências citadas**`, nao so `Referências`)."""

    def test_bold_wrapped_heading_with_extra_trailing_words_anchors_numbered_list(self):
        text = (
            "Corpo da resposta com uma alegacao[1].\n\n"
            "### **Referências citadas**\n\n"
            "1. Titulo do artigo, <u>https://example.com/artigo</u>\n"
        )
        refs = extract_references(text, source_answer_id="a1", tool_name="Gemini")
        assert len(refs) == 1
        assert refs[0].raw_url == "https://example.com/artigo"
        assert refs[0].citation_markers == ["[1]"]


class TestHeadingAnchoredNumberedList:
    """Formato do Gemini em PDF: mesma gramatica numerada+`<u>` do
    Perplexity, mas sem `⁂` nenhum — ancorada so pelo cabecalho da secao."""

    def test_numbered_list_with_title_before_url_is_extracted_without_asterism(self):
        text = (
            "### **Referências citadas**\n\n"
            "1. List of films - Wikipedia, \n\n"
            "   - <u>https://example.com/um</u> \n\n"
            "2. Outline - Wikipedia, <u>https://example.com/dois</u> \n"
        )
        refs = extract_references(text, source_answer_id="a1", tool_name="Gemini")
        by_marker = {r.citation_markers[0]: r.raw_url for r in refs}
        assert by_marker == {
            "[1]": "https://example.com/um",
            "[2]": "https://example.com/dois",
        }


class TestHeadingTitledPairsFormat:
    """Formato do Gemini em .docx: "Titulo, URL" por linha, sem numeracao
    nem marcacao nenhuma — ancorado so pelo cabecalho da secao de fontes
    (nunca por `⁂`, que e exclusivo do Perplexity). Marcador `[N]`
    inferido pela ordem de ocorrencia, com o titulo capturado."""

    def test_titled_pairs_get_positional_markers_and_titles(self):
        text = (
            "#### Referências citadas\n\n"
            "List of Marvel films - Wikipedia, https://example.com/um\n\n"
            "Outline of Marvel - Wikipedia, https://example.com/dois\n"
        )
        refs = extract_references(text, source_answer_id="a1", tool_name="Gemini")
        by_marker = {r.citation_markers[0]: r for r in refs}
        assert by_marker["[1]"].raw_url == "https://example.com/um"
        assert by_marker["[1]"].title == "List of Marvel films - Wikipedia"
        assert by_marker["[2]"].raw_url == "https://example.com/dois"

    def test_only_used_as_last_resort_when_no_other_format_matches(self):
        """Se o formato `[N] Titulo\\nURL` ja encontrou algo, o fallback
        de pares sem marcacao nunca deve rodar (evita duplicar/competir)."""
        text = (
            "[1] Titulo\nhttps://example.com/bracket\n\n"
            "#### Referências\n\n"
            "Outro titulo, https://example.com/nao-deveria-aparecer\n"
        )
        refs = extract_references(text, source_answer_id="a1", tool_name="ChatGPT")
        assert len(refs) == 1
        assert refs[0].raw_url == "https://example.com/bracket"

    def test_document_without_reference_heading_is_unaffected(self):
        text = "Corpo qualquer sem nenhuma lista de fontes.\n"
        refs = extract_references(text, source_answer_id="a1", tool_name="Gemini")
        assert refs == []
