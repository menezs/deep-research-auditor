from __future__ import annotations

import re

from ..models import AnswerChunk, Reference, ReferenceChunk

_BRACKET_RE = re.compile(r"\[\s*(\d+)\s*\]")
_SUP_RE = re.compile(r"<sup>\s*(\d+)\s*</sup>", re.IGNORECASE)
_SUB_RE = re.compile(r"<sub>\s*(\d+)\s*</sub>", re.IGNORECASE)

_UNICODE_SUPERSCRIPT_MAP = {
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
}
_UNICODE_SUPERSCRIPT_RE = re.compile("[" + "".join(_UNICODE_SUPERSCRIPT_MAP) + "]+")

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _normalize_citation_markers(text: str) -> str:
    """Converte `<sup>2</sup>`, `<sub>2</sub>` e superscripts unicode
    (²³¹...) para a forma canonica `[2]`, para que um unico regex
    reconheca todas as variacoes de marcador de citacao usadas pelas
    ferramentas de Deep Research — portado de
    `syntex/src/reference_extractor.py`."""
    text = _SUP_RE.sub(lambda m: f"[{m.group(1)}]", text)
    text = _SUB_RE.sub(lambda m: f"[{m.group(1)}]", text)
    text = _UNICODE_SUPERSCRIPT_RE.sub(
        lambda m: "[" + "".join(_UNICODE_SUPERSCRIPT_MAP[c] for c in m.group(0)) + "]",
        text,
    )
    return text


def _group_runs(matches: list[re.Match], text: str) -> list[tuple[list[str], int, int]]:
    """Agrupa marcadores adjacentes (ex: "[1][2][3]", sem texto entre
    eles) em uma unica run, retornando (markers, run_start, run_end)."""
    runs: list[tuple[list[str], int, int]] = []
    i = 0
    while i < len(matches):
        run_markers = [f"[{matches[i].group(1)}]"]
        run_end = matches[i].end()
        run_start = matches[i].start()
        j = i + 1
        while j < len(matches) and not text[run_end : matches[j].start()].strip():
            run_markers.append(f"[{matches[j].group(1)}]")
            run_end = matches[j].end()
            j += 1
        runs.append((run_markers, run_start, run_end))
        i = j
    return runs


def _build_marker_index(references: list[Reference]) -> dict[str, str]:
    index: dict[str, str] = {}
    for ref in references:
        for marker in ref.citation_markers:
            index[marker] = ref.id
    return index


def _resolve_markers(markers: list[str], marker_to_ref_id: dict[str, str]) -> list[str]:
    resolved = [marker_to_ref_id[m] for m in markers if m in marker_to_ref_id]
    return list(dict.fromkeys(resolved))


class AnswerChunker:
    """Divide o texto da resposta em trechos delimitados por marcadores
    de citacao, e resolve cada marcador para o `Reference.id` estavel
    correspondente (nao mais a string bruta "[1]") — portado de
    `syntex.ReferenceExtractor.extract_chunks_with_references`, com a
    resolucao de referencia (via `Reference.citation_markers`) somada."""

    def chunk(self, text: str, *, answer_id: str, references: list[Reference]) -> list[AnswerChunk]:
        normalized = _normalize_citation_markers(text)
        marker_to_ref_id = _build_marker_index(references)
        matches = list(_BRACKET_RE.finditer(normalized))

        if not matches:
            stripped = text.strip()
            if not stripped:
                return []
            return [
                AnswerChunk(id=f"{answer_id}-0", answer_id=answer_id, position=0, text=stripped, cited_reference_ids=[])
            ]

        chunks: list[AnswerChunk] = []
        cursor = 0
        position = 0
        for markers, run_start, run_end in _group_runs(matches, normalized):
            chunk_text = normalized[cursor:run_start].strip()
            if chunk_text:
                chunks.append(
                    AnswerChunk(
                        id=f"{answer_id}-{position}",
                        answer_id=answer_id,
                        position=position,
                        text=chunk_text,
                        cited_reference_ids=_resolve_markers(markers, marker_to_ref_id),
                    )
                )
                position += 1
            cursor = run_end

        trailing = normalized[cursor:].strip()
        if trailing:
            chunks.append(
                AnswerChunk(
                    id=f"{answer_id}-{position}",
                    answer_id=answer_id,
                    position=position,
                    text=trailing,
                    cited_reference_ids=[],
                )
            )
        return chunks


class DocumentChunker:
    """Divide um documento de referencia (markdown) em `ReferenceChunk`s
    coerentes com a estrutura de cabecalhos, respeitando um orcamento de
    tokens reais (via tiktoken) — portado de `syntex.SemanticChunker`,
    mas usando `MarkdownSplitter` (que ja entende cabecalhos nativamente
    em vez de um regex manual) e com `overlap` de fato aplicado (no
    syntex o parametro `overlap` era aceito mas nunca repassado ao
    splitter)."""

    def __init__(self, max_tokens: int = 512, overlap: int = 50, tiktoken_model: str = "gpt-3.5-turbo"):
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.tiktoken_model = tiktoken_model

    def chunk(self, *, reference_id: str, markdown: str, start_id: int = 0) -> list[ReferenceChunk]:
        from semantic_text_splitter import MarkdownSplitter
        import tiktoken

        splitter = MarkdownSplitter.from_tiktoken_model(self.tiktoken_model, self.max_tokens, overlap=self.overlap)
        encoding = tiktoken.encoding_for_model(self.tiktoken_model)

        chunks: list[ReferenceChunk] = []
        current_section: str | None = None
        for offset, text in enumerate(splitter.chunks(markdown)):
            header_match = _HEADER_RE.match(text)
            if header_match:
                current_section = header_match.group(2).strip()
            chunks.append(
                ReferenceChunk(
                    id=f"{reference_id}-{offset}",
                    reference_id=reference_id,
                    section=current_section,
                    text=text,
                    token_count=len(encoding.encode(text)),
                    embedding_id=start_id + offset,
                )
            )
        return chunks
