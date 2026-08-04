from __future__ import annotations

import re
import tempfile

from ..common.errors import ExtractionError

_CHARSET_RE = re.compile(r"charset=([\w-]+)", re.IGNORECASE)


def is_pdf_content(content_type: str, url: str) -> bool:
    return "application/pdf" in content_type.lower() or url.lower().split("?")[0].endswith(".pdf")


def convert_to_markdown(content: bytes, content_type: str, url: str) -> str:
    """Converte o conteudo baixado (HTML ou PDF) para Markdown, escolhendo
    o conversor por content-type/extensao (Strategy pattern)."""
    if is_pdf_content(content_type, url):
        return _pdf_to_markdown(content)
    return _html_to_markdown(content, content_type)


def _html_to_markdown(content: bytes, content_type: str) -> str:
    import trafilatura

    html = _decode_html(content, content_type)
    markdown = trafilatura.extract(html, output_format="markdown")
    if not markdown or not markdown.strip():
        raise ExtractionError("trafilatura nao conseguiu extrair conteudo principal do HTML")
    return markdown


def _decode_html(content: bytes, content_type: str) -> str:
    """Decodifica HTML priorizando sinais explicitos sobre sniffing.

    Muitos sites pt-BR (incluindo orgaos publicos como planalto.gov.br)
    nao declaram charset nem no header HTTP nem via `<meta charset>`, e a
    deteccao estatistica generica (usada pelo trafilatura e pelo proprio
    `requests.apparent_encoding`) pode confundir Windows-1252/ISO-8859-1
    com uma codificacao de byte unico de outra familia linguistica (ex:
    Windows-1250 Europa Central), produzindo mojibake como "avaliaçăo"
    em vez de "avaliação". A ordem de prioridade e:

    1. charset declarado no header `Content-Type` (mais confiavel).
    2. UTF-8, apenas se o decode for estritamente valido — texto latino
       em Windows-1252/ISO-8859-1 quase sempre falha aqui, entao um
       decode UTF-8 bem-sucedido e um sinal forte de que de fato e UTF-8.
    3. Windows-1252 como fallback final: cobre virtualmente todo o
       corpus tratado por este framework (referencias em portugues,
       espanhol e frances), e so e tentado depois que UTF-8 ja falhou.
    """
    charset_match = _CHARSET_RE.search(content_type)
    if charset_match:
        try:
            return content.decode(charset_match.group(1), errors="strict")
        except (LookupError, UnicodeDecodeError):
            pass

    try:
        return content.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        pass

    return content.decode("windows-1252", errors="replace")


def _pdf_to_markdown(content: bytes) -> str:
    import pymupdf4llm

    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        tmp.write(content)
        tmp.flush()
        markdown = pymupdf4llm.to_markdown(tmp.name)

    if not markdown or not markdown.strip():
        raise ExtractionError("pymupdf4llm nao conseguiu extrair texto do PDF")
    return markdown
