from __future__ import annotations

import json
import re
import tempfile

from ..common.errors import ExtractionError
from .fetcher import _reddit_thread_json_url

_CHARSET_RE = re.compile(r"charset=([\w-]+)", re.IGNORECASE)

_MAX_REDDIT_COMMENT_CHARS = 8000
_MAX_REDDIT_COMMENT_DEPTH = 2


def is_pdf_content(content_type: str, url: str) -> bool:
    return "application/pdf" in content_type.lower() or url.lower().split("?")[0].endswith(".pdf")


def is_reddit_json(content_type: str, url: str) -> bool:
    return "json" in content_type.lower() and _reddit_thread_json_url(url) is not None


def convert_to_markdown(content: bytes, content_type: str, url: str) -> str:
    """Converte o conteudo baixado (HTML, PDF ou JSON de thread do Reddit)
    para Markdown, escolhendo o conversor por content-type/URL (Strategy
    pattern)."""
    if is_reddit_json(content_type, url):
        return _reddit_json_to_markdown(content)
    if is_pdf_content(content_type, url):
        return _pdf_to_markdown(content)
    return _html_to_markdown(content, content_type)


def _iter_reddit_comments(listing, depth: int):
    """Percorre a arvore de comentarios (`kind == "t1"`) ate
    `_MAX_REDDIT_COMMENT_DEPTH` niveis, ignorando os nos `"more"`
    (placeholders de "carregar mais comentarios")."""
    try:
        children = listing["data"]["children"]
    except (KeyError, TypeError):
        return
    for child in children:
        if child.get("kind") != "t1":
            continue
        data = child.get("data", {})
        body = (data.get("body") or "").strip()
        if body:
            yield depth, data.get("author") or "[desconhecido]", data.get("score"), body
        replies = data.get("replies")
        if replies and depth + 1 < _MAX_REDDIT_COMMENT_DEPTH:
            yield from _iter_reddit_comments(replies, depth + 1)


def _reddit_json_to_markdown(content: bytes) -> str:
    """Renderiza a resposta do endpoint `.json` de uma thread do Reddit
    (`[listing_do_post, listing_dos_comentarios]`) como Markdown: titulo +
    corpo do post seguidos dos comentarios (ate 2 niveis, truncados em
    `_MAX_REDDIT_COMMENT_CHARS`). Substitui a extracao via trafilatura, que
    numa pagina de comentarios do Reddit devolve a sidebar de regras do
    subreddit em vez da discussao."""
    try:
        payload = json.loads(content.decode("utf-8", errors="replace"))
        post = payload[0]["data"]["children"][0]["data"]
        comment_listing = payload[1]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise ExtractionError(f"resposta JSON do Reddit em formato inesperado: {exc}") from exc

    lines: list[str] = []
    title = (post.get("title") or "").strip()
    if title:
        lines.append(f"# {title}")
    author = post.get("author") or "[desconhecido]"
    score = post.get("score")
    lines.append(f"*Postado por u/{author}" + (f" · {score} pontos*" if score is not None else "*"))
    selftext = (post.get("selftext") or "").strip()
    if selftext:
        lines.extend(["", selftext])

    comment_lines: list[str] = []
    total = 0
    for depth, comment_author, comment_score, body in _iter_reddit_comments(comment_listing, 0):
        total += len(body)
        if total > _MAX_REDDIT_COMMENT_CHARS:
            comment_lines.append("- *[comentários restantes truncados]*")
            break
        meta = f" ({comment_score})" if comment_score is not None else ""
        flat_body = body.replace("\n", " ")
        comment_lines.append(f"{'  ' * depth}- **u/{comment_author}{meta}**: {flat_body}")

    if comment_lines:
        lines.extend(["", "## Comentários", *comment_lines])

    markdown = "\n".join(lines).strip()
    if not markdown:
        raise ExtractionError("thread do Reddit sem titulo, corpo ou comentarios utilizaveis")
    return markdown


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
