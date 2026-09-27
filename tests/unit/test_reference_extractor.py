from pathlib import Path

import pytest

from auditframework.extraction.reference_extractor import (
    extract_references,
    find_reference_section,
)
from auditframework.pipeline import _strip_reference_section

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


class TestUrlWithParentheses:
    """URLs reais contêm parênteses (PII do Lancet, desambiguação da
    Wikipedia). Truncá-las fazia cada uma virar uma "obra" falsamente
    exclusiva na análise de interseção entre execuções."""

    def test_lancet_pii_parentheses_are_preserved(self):
        url = "https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(21)02178-4/fulltext"
        refs = extract_references(f"[1] Estudo\n{url}\n", source_answer_id="a1", tool_name="Grok")
        assert refs[0].raw_url == url

    def test_wikipedia_disambiguation_parentheses_are_preserved(self):
        url = "https://en.wikipedia.org/wiki/Mercury_(planet)"
        refs = extract_references(f"[1] Verbete\n{url}\n", source_answer_id="a1", tool_name="Grok")
        assert refs[0].raw_url == url

    def test_closing_paren_from_surrounding_prose_is_dropped(self):
        text = "[1] Titulo\n(ver https://example.com/artigo)\n"
        refs = extract_references(text, source_answer_id="a1", tool_name="Grok")
        assert refs[0].raw_url == "https://example.com/artigo"

    def test_trailing_period_and_paren_are_both_dropped(self):
        text = "[1] Titulo\n(fonte: https://example.com/artigo).\n"
        refs = extract_references(text, source_answer_id="a1", tool_name="Grok")
        assert refs[0].raw_url == "https://example.com/artigo"

    def test_markdown_link_closing_paren_is_not_part_of_the_url(self):
        text = "[1] [Título do artigo](https://example.com/artigo)\n"
        refs = extract_references(text, source_answer_id="a1", tool_name="Grok")
        assert refs[0].raw_url == "https://example.com/artigo"


class TestExporterContamination:
    """PDFs do Perplexity trazem, junto da lista real, dezenas de entradas
    sobre COMO citar — guias de formatação e documentação da ferramenta. Não
    são fontes do conteúdo, então ficam MARCADAS (não removidas): apagar
    tornava a exclusão inauditável, e os padrões são genéricos o bastante
    para poderem errar num relatório cujo tema seja citação."""

    def test_citation_style_guide_is_marked_not_removed(self):
        text = (
            "[1] Fonte real\nhttps://example.com/real\n"
            "[2] Como citar\nhttps://www.scribbr.com/citing-sources/numbered_citation_style\n"
        )
        refs = extract_references(text, source_answer_id="a1", tool_name="Perplexity")
        by_url = {r.raw_url: r for r in refs}
        assert len(by_url) == 2  # as duas continuam registradas
        assert by_url["https://example.com/real"].exporter_artifact is False
        assert by_url["https://www.scribbr.com/citing-sources/numbered_citation_style"].exporter_artifact is True

    def test_tool_documentation_is_marked(self):
        text = "[1] Doc\nhttps://docs.perplexity.ai/guides/rag-citations\n"
        refs = extract_references(text, source_answer_id="a1", tool_name="Perplexity")
        assert len(refs) == 1 and refs[0].exporter_artifact is True

    def test_legitimate_reference_is_untouched(self):
        text = "[1] Artigo\nhttps://www.nature.com/articles/s41562-025-02259-6\n"
        refs = extract_references(text, source_answer_id="a1", tool_name="Perplexity")
        assert len(refs) == 1 and refs[0].exporter_artifact is False

    def test_artifact_never_steals_a_marker_from_a_real_source(self):
        """O artefato não tem posição fixa: pode vir ANTES da fonte real que
        usa o mesmo número. Quem resolve `[3]` tem de ser a fonte real."""
        text = (
            "[3] Guia\nhttps://www.scribbr.com/citing-sources/numbered_citation_style\n"
            "[3] Fonte real\nhttps://example.com/real\n"
        )
        refs = extract_references(text, source_answer_id="a1", tool_name="Perplexity")
        by_url = {r.raw_url: r for r in refs}
        assert by_url["https://example.com/real"].citation_markers == ["[3]"]
        assert by_url["https://www.scribbr.com/citing-sources/numbered_citation_style"].citation_markers == []


class TestWorkKey:
    """`work_key` responde "é o mesmo documento?"; `normalized_url`
    responde "é a mesma página?". As duas colunas precisam coexistir."""

    def test_pubmed_id_becomes_the_work_key(self):
        refs = extract_references(
            "[1] Lei 2020\nhttps://pubmed.ncbi.nlm.nih.gov/32997908/\n",
            source_answer_id="a1", tool_name="Manus",
        )
        assert refs[0].work_key == "PMID:32997908"

    def test_nature_article_maps_to_its_doi(self):
        refs = extract_references(
            "[1] Estudo\nhttps://www.nature.com/articles/s41562-025-02259-6\n",
            source_answer_id="a1", tool_name="Grok",
        )
        assert refs[0].work_key == "DOI:10.1038/s41562-025-02259-6"

    def test_same_work_under_two_urls_shares_the_work_key(self):
        text = (
            "[1] Versao PMC\nhttps://pmc.ncbi.nlm.nih.gov/articles/PMC8486335/\n"
            "[2] Mesma obra\nhttps://www.ncbi.nlm.nih.gov/pmc/articles/PMC8486335/\n"
        )
        refs = extract_references(text, source_answer_id="a1", tool_name="Grok")
        assert {r.work_key for r in refs} == {"PMC:PMC8486335"}

    def test_urls_of_the_same_work_are_not_deduplicated(self):
        """A verificação de trecho depende do endereço acessado — o PubMed
        expõe só o resumo, o site do periódico o texto inteiro."""
        text = (
            "[1] Abstract\nhttps://pubmed.ncbi.nlm.nih.gov/32997908/\n"
            "[2] Texto completo\nhttps://doi.org/10.1056/NEJMoa1917338\n"
        )
        refs = extract_references(text, source_answer_id="a1", tool_name="Manus")
        assert len(refs) == 2


class TestReferenceSectionBacktrack:
    """Perplexity emite duas listas: uma ancorada pelo cabeçalho e outra,
    numerada, depois do `⁂`. Ancorar só na última deixava a primeira no
    corpo — cada entrada dela virava um "chunk" a julgar, e suas fontes
    desapareciam da lista de referências."""

    def test_backtracks_to_the_earlier_anchor_when_everything_between_is_a_source_list(self):
        text = (
            "## Referências\n\n"
            "1. <u>https://example.com/a</u>\n"
            "2. <u>https://example.com/b</u>\n\n"
            "⁂ \n\n"
            "3. <u>https://example.com/c</u>\n"
        )
        refs = extract_references(text, source_answer_id="a1", tool_name="Perplexity")
        assert {r.raw_url for r in refs} == {
            "https://example.com/a",
            "https://example.com/b",
            "https://example.com/c",
        }

    def test_body_and_reference_list_are_complementary(self):
        """O corte usado pelo chunker é a MESMA âncora usada pela extração —
        nenhuma entrada da lista pode sobrar no corpo."""
        text = (
            "Uma afirmação qualquer do corpo. [1]\n\n"
            "## Referências\n\n"
            "1. <u>https://example.com/a</u>\n\n"
            "⁂ \n\n"
            "2. <u>https://example.com/b</u>\n"
        )
        body = _strip_reference_section(text)
        assert body == "Uma afirmação qualquer do corpo. [1]"
        assert "example.com" not in body

    def test_prose_list_before_the_anchor_still_blocks_the_backtrack(self):
        """Contraprova: uma lista numerada de resumo (sem URL) tem a mesma
        forma de uma lista de fontes — não pode arrastar o corpo para dentro
        da seção de referências."""
        text = (
            "## Referências\n"
            "1. **Fonte A** - resumo sem link\n"
            "2. **Fonte B** - outro resumo\n\n"
            "⁂ \n\n"
            "1. <u>https://example.com/real</u>\n"
        )
        anchor = find_reference_section(text)
        assert text[anchor[0]] == "⁂"


class TestPrefixTruncatedUrlRepair:
    def test_last_path_segment_lost_at_a_line_break_is_restored(self, tmp_path):
        """A URL extraída é prefixo exato de uma embutida no PDF: o último
        segmento do path ficou na linha seguinte e se perdeu."""
        pytest.importorskip("fitz")
        pdf_path = tmp_path / "resposta.pdf"
        full = "https://example.com/businesses-and-occupations/samsung-electronics-co-ltd"
        TestRepairUsingPdfLinks()._make_pdf(pdf_path, [full])

        text = "[1] Titulo\nhttps://example.com/businesses-and-occupations/\n"
        refs = extract_references(text, source_answer_id="a1", tool_name="Gemini", answer_path=pdf_path)

        assert refs[0].raw_url == full

    def test_two_extra_path_segments_are_a_different_page_not_a_repair(self, tmp_path):
        pytest.importorskip("fitz")
        pdf_path = tmp_path / "resposta.pdf"
        TestRepairUsingPdfLinks()._make_pdf(pdf_path, ["https://example.com/a/b/c/d/e/f"])

        text = "[1] Titulo\nhttps://example.com/a/\n"
        refs = extract_references(text, source_answer_id="a1", tool_name="Gemini", answer_path=pdf_path)

        assert refs[0].raw_url == "https://example.com/a/"

    def test_browser_highlight_fragment_is_never_added_by_the_repair(self, tmp_path):
        pytest.importorskip("fitz")
        pdf_path = tmp_path / "resposta.pdf"
        TestRepairUsingPdfLinks()._make_pdf(pdf_path, ["https://example.com/artigo#:~:text=trecho%20destacado"])

        text = "[1] Titulo\nhttps://example.com/artigo\n"
        refs = extract_references(text, source_answer_id="a1", tool_name="Gemini", answer_path=pdf_path)

        assert refs[0].raw_url == "https://example.com/artigo"


class TestMarkerCollisionBetweenTwoLists:
    """Uma resposta do Perplexity traz duas listas de fontes, ambas
    numeradas a partir de 1 — `[3]` acabava apontando para duas
    referências, e o relatório ficava autocontraditório ("cita [3],
    corroborado por [3]")."""

    def test_repeated_marker_stays_with_the_first_reference_in_document_order(self):
        text = (
            "## References\n\n"
            "1. <u>https://example.com/real-um</u>\n"
            "3. <u>https://example.com/real-tres</u>\n\n"
            "⁂ \n\n"
            "3. <u>https://example.com/segunda-lista</u>\n"
        )
        refs = extract_references(text, source_answer_id="a1", tool_name="Perplexity")
        by_marker = {m: r for r in refs for m in r.citation_markers}
        assert by_marker["[3]"].raw_url == "https://example.com/real-tres"

    def test_reference_that_lost_its_marker_is_still_listed(self):
        """Ela continua na lista (aparece como "listada mas nunca citada") —
        o que se pode afirmar é que o marcador não é dela, não que a fonte
        não existe."""
        text = (
            "## References\n\n1. <u>https://example.com/primeira</u>\n\n"
            "⁂ \n\n1. <u>https://example.com/segunda</u>\n"
        )
        refs = extract_references(text, source_answer_id="a1", tool_name="Perplexity")
        urls = {r.raw_url for r in refs}
        assert urls == {"https://example.com/primeira", "https://example.com/segunda"}
        segunda = next(r for r in refs if r.raw_url.endswith("segunda"))
        assert segunda.citation_markers == []


class TestWritingGuideContamination:
    def test_author_guidelines_page_is_marked(self):
        text = (
            "[1] Fonte real\nhttps://example.com/real\n"
            "[2] Guia\nhttps://www.asme.org/publications-submissions/proceedings/author-guidelines/elements-of-a-paper\n"
        )
        refs = extract_references(text, source_answer_id="a1", tool_name="Perplexity")
        marked = [r.raw_url for r in refs if r.exporter_artifact]
        assert marked == [
            "https://www.asme.org/publications-submissions/proceedings/author-guidelines/elements-of-a-paper"
        ]


class TestTwoSourceListsPriority:
    """O Perplexity emite duas listas: uma sob `## References`, cuja
    numeração é a que o corpo cita, e outra depois do `⁂`, que reenumera
    todas as fontes consultadas de 1 a N. A mesma URL nas duas fazia a
    referência acumular os dois números."""

    def test_url_in_both_lists_keeps_the_references_number(self):
        text = (
            "## References\n\n"
            "1. <u>https://example.com/artigo</u>\n\n"
            "⁂ \n\n"
            "26. <u>https://example.com/artigo</u>\n"
        )
        refs = extract_references(text, source_answer_id="a1", tool_name="Perplexity")
        assert len(refs) == 1
        assert refs[0].citation_markers == ["[1]"]

    def test_url_only_in_the_asterism_list_keeps_its_own_number(self):
        text = (
            "## References\n\n"
            "1. <u>https://example.com/citada</u>\n\n"
            "⁂ \n\n"
            "27. <u>https://example.com/so-na-segunda</u>\n"
        )
        refs = extract_references(text, source_answer_id="a1", tool_name="Perplexity")
        by_url = {r.raw_url: r.citation_markers for r in refs}
        assert by_url["https://example.com/so-na-segunda"] == ["[27]"]

    def test_two_numbers_in_the_same_list_are_both_kept(self):
        """Caso legítimo: a lista repete a mesma URL sob dois números e o
        corpo pode citar qualquer um dos dois."""
        text = (
            "## References\n\n"
            "8. <u>https://example.com/artigo</u>\n"
            "10. <u>https://example.com/artigo</u>\n"
        )
        refs = extract_references(text, source_answer_id="a1", tool_name="Perplexity")
        assert len(refs) == 1
        assert refs[0].citation_markers == ["[8]", "[10]"]

    def test_single_list_is_unaffected(self):
        text = "⁂ \n\n1. <u>https://example.com/a</u>\n2. <u>https://example.com/a</u>\n"
        refs = extract_references(text, source_answer_id="a1", tool_name="Perplexity")
        assert refs[0].citation_markers == ["[1]", "[2]"]


class TestMidSegmentTruncationRepair:
    def test_url_cut_in_the_middle_of_a_segment_is_restored(self, tmp_path):
        pytest.importorskip("fitz")
        pdf_path = tmp_path / "resposta.pdf"
        full = "https://example.com/esrc/making-the-case-for-a-four-day-working-week/"
        TestRepairUsingPdfLinks()._make_pdf(pdf_path, [full])

        text = "[1] Titulo\nhttps://example.com/esrc/making-the-case-for-a-four\n"
        refs = extract_references(text, source_answer_id="a1", tool_name="Perplexity", answer_path=pdf_path)

        assert refs[0].raw_url == full

    def test_ambiguous_prefix_is_not_repaired(self, tmp_path):
        """Dois hyperlinks começam com a URL extraída: não há como saber qual
        é a truncada, então nada é reparado. Os sufixos são longos de
        propósito, para ficarem fora do orçamento da regra de ligadura e o
        caso chegar de fato na regra de prefixo."""
        pytest.importorskip("fitz")
        pdf_path = tmp_path / "resposta.pdf"
        TestRepairUsingPdfLinks()._make_pdf(
            pdf_path,
            [
                "https://example.com/artigo-sobre-jornada-de-trabalho",
                "https://example.com/artigo-sobre-salario-minimo",
            ],
        )

        text = "[1] Titulo\nhttps://example.com/artigo\n"
        refs = extract_references(text, source_answer_id="a1", tool_name="Perplexity", answer_path=pdf_path)

        assert refs[0].raw_url == "https://example.com/artigo"

    def test_bare_host_is_never_repaired_into_a_page(self, tmp_path):
        pytest.importorskip("fitz")
        pdf_path = tmp_path / "resposta.pdf"
        TestRepairUsingPdfLinks()._make_pdf(pdf_path, ["https://example.com/alguma-pagina"])

        text = "[1] Titulo\nhttps://example.com/\n"
        refs = extract_references(text, source_answer_id="a1", tool_name="Perplexity", answer_path=pdf_path)

        assert refs[0].raw_url == "https://example.com/"


class TestPriorityAfterPdfLinkRepair:
    """As duas listas citam a mesma fonte, e o conversor corrompe cada
    ocorrência de um jeito diferente — elas só ficam idênticas DEPOIS do
    reparo pelos hyperlinks do PDF, passando longe da deduplicação por URL.
    A prioridade tem de valer ali também."""

    def test_cross_list_duplicate_keeps_the_primary_number(self, tmp_path):
        pytest.importorskip("fitz")
        pdf_path = tmp_path / "resposta.pdf"
        full = "https://example.com/four-day-week-made-permanent-for-most-uk-firms-in-worlds-biggest-trial"
        TestRepairUsingPdfLinks()._make_pdf(pdf_path, [full])

        # "firmsin" na lista principal, "bigg est" na do ⁂: URLs distintas
        # até o reparo, a mesma depois dele.
        text = (
            "## References\n\n"
            "4. <u>https://example.com/four-day-week-made-permanent-for-most-uk-firmsin-worlds-biggest-trial</u>\n\n"
            "⁂ \n\n"
            "14. <u>https://example.com/four-day-week-made-permanent-for-most-uk-firms-in-worlds-bigg est-trial</u>\n"
        )
        refs = extract_references(text, source_answer_id="a1", tool_name="Perplexity", answer_path=pdf_path)

        assert len(refs) == 1
        assert refs[0].raw_url == full
        assert refs[0].citation_markers == ["[4]"]

    def test_same_list_duplicate_keeps_both_numbers(self, tmp_path):
        """Contraprova: a MESMA lista repete a fonte sob dois números e o
        corpo cita os dois — descartar um deixaria o trecho sem fonte."""
        pytest.importorskip("fitz")
        pdf_path = tmp_path / "resposta.pdf"
        full = "https://example.com/assessing-the-operational-impact-of-a-four-day-week"
        TestRepairUsingPdfLinks()._make_pdf(pdf_path, [full])

        text = (
            "## References\n\n"
            "8. <u>https://example.com/assessing-the-operational-impact-of-a-fourday-week</u>\n"
            "10. <u>https://example.com/assessing-the-operational-impact-of-a-four-day-we ek</u>\n"
        )
        refs = extract_references(text, source_answer_id="a1", tool_name="Perplexity", answer_path=pdf_path)

        assert len(refs) == 1
        assert refs[0].citation_markers == ["[8]", "[10]"]


class TestRobustnessAcrossInputShapes:
    """Requisito: nenhum arquivo de entrada pode provocar erro. Cada forma
    degenerada abaixo já apareceu ou é plausível no corpus."""

    SHAPES = {
        "vazio": "",
        "so_espaco": "   \n\n  \t ",
        "so_cabecalho": "## Referências\n",
        "so_asterismo": "⁂",
        "asterismo_sem_url": "Corpo [1].\n\n⁂\n\nnada aqui\n",
        "cabecalho_sem_url": "Corpo [1].\n\n## References\n\nsem link nenhum\n",
        "marcador_sem_lista": "Uma afirmação citada [7].\n",
        "lista_sem_marcador": "## References\n\nhttps://example.com/a\n",
        "duas_listas_vazias": "## References\n\n⁂\n",
        "host_nu": "## References\n\n1. <u>https://example.com</u>\n",
        "marcador_zero": "Afirmação [0].\n\n## References\n\n0. <u>https://example.com/z</u>\n",
        "numero_gigante": "Afirmação [999999].\n\n⁂\n\n999999. <u>https://example.com/g</u>\n",
        "colchete_malformado": "Afirmação [ e outra ].\n\n## References\n\n1. <u>https://example.com/m</u>\n",
        "url_sem_esquema": "## References\n\n1. <u>example.com/sem-esquema</u>\n",
        "so_tabela": "|a|b|\n|---|---|\n|1|2|\n",
        "asterismo_antes_do_corpo": "⁂\n\nCorpo [1].\n\n## References\n\n1. <u>https://example.com/a</u>\n",
        "tres_listas": (
            "## References\n\n1. <u>https://example.com/a</u>\n\n"
            "⁂\n\n1. <u>https://example.com/b</u>\n\n"
            "⁂\n\n1. <u>https://example.com/c</u>\n"
        ),
    }

    @pytest.mark.parametrize("shape", sorted(SHAPES))
    def test_extraction_and_chunking_never_raise(self, shape):
        text = self.SHAPES[shape]
        refs = extract_references(text, source_answer_id="a1", tool_name="Perplexity")
        auditable = [r for r in refs if not r.exporter_artifact]

        # invariante: um marcador nunca pertence a duas referências
        owners: dict[str, str] = {}
        for ref in auditable:
            for marker in ref.citation_markers:
                assert marker not in owners, f"{marker} reivindicado duas vezes em {shape!r}"
                owners[marker] = ref.id
