from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from auditframework.extraction.pdf_links import extract_pdf_hyperlink_urls


def _make_pdf(path: Path, links: list[str]) -> None:
    doc = fitz.open()
    page = doc.new_page()
    for i, uri in enumerate(links):
        rect = fitz.Rect(50, 50 + i * 20, 300, 65 + i * 20)
        page.insert_text((50, 60 + i * 20), f"link {i}")
        page.insert_link({"kind": fitz.LINK_URI, "from": rect, "uri": uri})
    doc.save(str(path))
    doc.close()


def test_extracts_uris_from_embedded_links(tmp_path: Path):
    pdf_path = tmp_path / "doc.pdf"
    _make_pdf(pdf_path, ["https://example.com/de-marco", "https://example.com/outra-pagina"])

    urls = extract_pdf_hyperlink_urls(pdf_path)

    assert urls == {"https://example.com/de-marco", "https://example.com/outra-pagina"}


def test_strips_scroll_to_text_fragment(tmp_path: Path):
    pdf_path = tmp_path / "doc.pdf"
    _make_pdf(pdf_path, ["https://example.com/artigo#:~:text=trecho%20citado"])

    urls = extract_pdf_hyperlink_urls(pdf_path)

    assert urls == {"https://example.com/artigo"}


def test_pdf_without_links_returns_empty_set(tmp_path: Path):
    pdf_path = tmp_path / "sem_links.pdf"
    _make_pdf(pdf_path, [])

    assert extract_pdf_hyperlink_urls(pdf_path) == set()


def test_nonexistent_pdf_returns_empty_set_instead_of_raising(tmp_path: Path):
    assert extract_pdf_hyperlink_urls(tmp_path / "nao-existe.pdf") == set()


def test_non_http_links_are_ignored(tmp_path: Path):
    pdf_path = tmp_path / "doc.pdf"
    _make_pdf(pdf_path, ["mailto:contato@example.com", "https://example.com/real"])

    urls = extract_pdf_hyperlink_urls(pdf_path)

    assert urls == {"https://example.com/real"}
