import pytest

from auditframework.common.errors import ExtractionError
from auditframework.ingestion.converters import convert_to_markdown

pytest.importorskip("trafilatura")
pytest.importorskip("pymupdf4llm")

_HTML = """
<html>
<head><title>Artigo de teste</title></head>
<body>
<article>
<h1>Marco Civil da Internet</h1>
<p>O Marco Civil da Internet estabelece principios, garantias, direitos e
deveres para o uso da internet no Brasil, incluindo neutralidade de rede
e protecao de dados pessoais dos usuarios.</p>
</article>
</body>
</html>
"""


def test_html_is_converted_to_markdown():
    markdown = convert_to_markdown(_HTML.encode("utf-8"), "text/html", "https://example.com/artigo")
    assert "Marco Civil da Internet" in markdown


def test_html_with_declared_non_utf8_charset_is_decoded_correctly():
    """Regressao: forcar UTF-8 no decode (bug original) transformava
    acentos em mojibake para paginas em ISO-8859-1/Windows-1252, comuns
    em sites de orgaos publicos brasileiros (ex: planalto.gov.br)."""
    html = (
        '<html><head><meta charset="iso-8859-1"></head><body><article>'
        "<p>Lei Nº 13.709 dispõe sobre proteção de dados pessoais.</p>"
        "</article></body></html>"
    )
    markdown = convert_to_markdown(html.encode("iso-8859-1"), "text/html", "https://example.com/lei")

    assert "Lei Nº 13.709" in markdown
    assert "proteção de dados pessoais" in markdown


def test_html_without_meta_charset_uses_http_header_charset():
    """Regressao: paginas sem `<meta charset>` no corpo (comum em sites
    de orgaos publicos) dependem inteiramente da deteccao heuristica do
    trafilatura, que pode confundir ISO-8859-1/Windows-1252 com outra
    codificacao de byte unico proxima (ex: produzindo "avaliaçăo" em vez
    de "avaliação"). O charset declarado no header HTTP Content-Type e
    mais confiavel e deve ter prioridade quando presente."""
    html_sem_meta = "<html><body><article><p>Relatório de avaliação de impacto à proteção de dados.</p></article></body></html>"
    markdown = convert_to_markdown(
        html_sem_meta.encode("iso-8859-1"),
        "text/html; charset=ISO-8859-1",
        "https://example.com/relatorio",
    )

    assert "avaliação" in markdown
    assert "proteção de dados" in markdown


def test_empty_html_raises_extraction_error():
    with pytest.raises(ExtractionError):
        convert_to_markdown(b"<html><body></body></html>", "text/html", "https://example.com/vazio")


def test_pdf_is_converted_to_markdown():
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Lei Geral de Protecao de Dados Pessoais")
    pdf_bytes = doc.tobytes()
    doc.close()

    markdown = convert_to_markdown(pdf_bytes, "application/pdf", "https://example.com/lgpd.pdf")

    assert "Lei Geral de Protecao de Dados Pessoais" in markdown


def test_pdf_detected_by_url_suffix_even_without_content_type():
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "conteudo do pdf")
    pdf_bytes = doc.tobytes()
    doc.close()

    markdown = convert_to_markdown(pdf_bytes, "", "https://example.com/arquivo.pdf")

    assert "conteudo do pdf" in markdown
