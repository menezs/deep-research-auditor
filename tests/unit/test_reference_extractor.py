from pathlib import Path

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
