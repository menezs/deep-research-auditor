from __future__ import annotations

import re
from pathlib import Path

from ..indexing.chunkers import _BRACKET_RE, _normalize_citation_markers
from ..logging_config import get_logger
from ..models import AnswerChunk, Reference, SourceInfo

logger = get_logger(__name__)


def build_source_info(
    *,
    path: Path,
    body: str,
    references: list[Reference],
    chunks: list[AnswerChunk],
) -> SourceInfo:
    """Metadados forenses do arquivo auditado — best-effort por formato,
    sem LLM. Uma falha ao ler os metadados nunca derruba a extracao."""
    suffix = path.suffix.lower()
    info = SourceInfo(format=suffix, size_bytes=_safe_size(path))

    try:
        if suffix == ".pdf":
            _fill_pdf(info, path)
        elif suffix == ".docx":
            _fill_docx(info, path)
    except Exception as exc:  # biblioteca externa: superficie de erro ampla
        logger.warning("Nao foi possivel ler os metadados de %s: %s", path.name, exc)

    markers = _BRACKET_RE.findall(_normalize_citation_markers(body))
    info.citation_markers_total = len(markers)
    info.citation_markers_distinct = len(set(markers))

    cited_ids = {rid for chunk in chunks for rid in chunk.cited_reference_ids}
    info.references_listed = len(references)
    info.references_never_cited = sum(1 for r in references if r.id not in cited_ids)
    return info


def _safe_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _fill_pdf(info: SourceInfo, path: Path) -> None:
    import fitz

    with fitz.open(path) as doc:
        info.page_count = doc.page_count
        info.encrypted = bool(doc.is_encrypted)
        meta = doc.metadata or {}
    info.pdf_title = _clean(meta.get("title"))
    info.pdf_author = _clean(meta.get("author"))
    info.pdf_creator = _clean(meta.get("creator"))
    info.pdf_producer = _clean(meta.get("producer"))
    info.created = _pdf_date(meta.get("creationDate"))
    info.modified = _pdf_date(meta.get("modDate"))
    info.browser_print = _is_browser_print(info.pdf_creator, info.pdf_producer)


def _fill_docx(info: SourceInfo, path: Path) -> None:
    import docx

    props = docx.Document(str(path)).core_properties
    info.pdf_title = _clean(props.title)
    info.pdf_author = _clean(props.author)
    info.created = str(props.created) if props.created else None
    info.modified = str(props.modified) if props.modified else None


_PDF_DATE_RE = re.compile(r"D:(\d{4})(\d{2})(\d{2})(\d{2})?(\d{2})?(\d{2})?")


def _pdf_date(value: str | None) -> str | None:
    """`D:20260830183346+00'00'` -> `2026-08-30 18:33:46 UTC`."""
    value = _clean(value)
    if not value:
        return None
    m = _PDF_DATE_RE.match(value)
    if not m:
        return value
    y, mo, d, hh, mm, ss = (p or "00" for p in m.groups())
    tz = " UTC" if "+00'00'" in value or value.endswith("Z") else ""
    return f"{y}-{mo}-{d} {hh}:{mm}:{ss}{tz}"


def _clean(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    return value or None


def _is_browser_print(creator: str | None, producer: str | None) -> bool:
    joined = f"{creator or ''} {producer or ''}".lower()
    return ("skia/pdf" in joined) or ("chromium" in joined and "pdf" in joined)
