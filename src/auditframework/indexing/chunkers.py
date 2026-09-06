from __future__ import annotations

import re

from ..models import AnswerChunk, Reference, ReferenceChunk

_BRACKET_RE = re.compile(r"\[\s*(\d+)\s*\]")
_SUP_RE = re.compile(r"<sup>\s*(\d+)\s*</sup>", re.IGNORECASE)
_SUB_RE = re.compile(r"<sub>\s*(\d+)\s*</sub>", re.IGNORECASE)
# Gemini/Docs: `[cite_start]` (abre) e `[cite: 12]` / `[cite: 12, 13]` (fecha).
_CITE_START_RE = re.compile(r"\[cite_start\]", re.IGNORECASE)
_CITE_RE = re.compile(r"\[cite:\s*([\d,\s]+)\]", re.IGNORECASE)

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

# Tabelas markdown no corpo da resposta: o pymupdf4llm as devolve como
# `|celula|celula|` em linhas consecutivas, com `<br>` dentro das celulas.
# Sem tratar isso, o `AnswerChunker` fatiaria cada celula com marcador de
# citacao num fragmento ilegivel (`.|texto<br>mais texto`).
_TABLE_ROW_RE = re.compile(r"^[ \t]*\|.*\|[ \t]*$")
_TABLE_SEP_RE = re.compile(r"^[ \t]*\|[\s:|-]+\|[ \t]*$")
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_LEADING_NOISE_RE = re.compile(r"^[\s.,;:!?)\]|–—-]+")

# Fronteira de frase: `.`/`!`/`?` seguido de espaco + maiuscula (ASCII ou
# Latin-1 acentuada) ou fim do texto. Evita contar `.` de numeros
# (`1.338`, `v2.5`) como fim de frase.
_SENTENCE_BOUNDARY_RE = re.compile(r'[.!?]+(?:\s+(?=[A-ZÀ-Þ"\'(])|\s*$)')

# Wrappers de formatacao que o pymupdf4llm deixa ao redor de links e
# marcadores de citacao (`<u>[1] [2]</u>`) — sem valor semantico.
_TAG_ARTIFACT_RE = re.compile(r"</?(?:u|b|i|em|strong|span|mark)\s*>", re.IGNORECASE)
_WORD_RE = re.compile(r"\w{2,}")
_MIN_CLAIM_WORDS = 4


def _count_sentences(text: str) -> int:
    """Estimativa de frases num trecho — usada para sinalizar chunks em
    que so a ultima frase esta ancorada por uma citacao."""
    boundaries = len(_SENTENCE_BOUNDARY_RE.findall(text))
    if not text.rstrip().endswith((".", "!", "?")):
        boundaries += 1
    return max(1, boundaries)


def _strip_tag_artifacts(text: str) -> str:
    return _TAG_ARTIFACT_RE.sub("", text).strip()


def _looks_like_claim(text: str) -> bool:
    """Paragrafo que carrega afirmacao factual — separa prosa (que vale
    sinalizar quando nao tem fonte) de cabecalhos, linhas de tabela,
    titulos, URLs e fragmentos de frase quebrada."""
    body = _HEADER_RE.sub("", _strip_tag_artifacts(text)).strip()
    body = re.sub(r"^[-*\d.)\s\"'“]+|\*+", "", body).strip()  # bullet/numeracao/enfase inicial
    if not body or (not body[0].isupper() and body[0] not in "\"'“("):
        return False  # vazio ou continuacao de frase (comeca minusculo)
    words = _WORD_RE.findall(body)
    if len(words) < _MIN_CLAIM_WORDS:
        return False
    # prosa de verdade: termina em pontuacao final, ou e longa o bastante
    # para nao ser um cabecalho/linha de tabela
    return body.rstrip().endswith((".", "!", "?", "”", '"')) or len(words) >= 15


def _split_paragraphs(text: str) -> list[str]:
    """Paragrafos separados por linha em branco; um cabecalho markdown no
    meio de um bloco tambem inicia um novo paragrafo."""
    text = re.sub(r"(?m)(?<=.)\n(?=[ \t]*#{1,6}\s)", "\n\n", text)
    return [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]


_ROW_HAS_MARKER_RE = re.compile(
    r"<sup>\s*\d|<sub>\s*\d|\[\s*\d+\s*\]|\[cite:|[" + "".join(_UNICODE_SUPERSCRIPT_MAP) + "]"
)


def _linearize_row(row: str) -> str:
    parts: list[str] = []
    for cell in row.strip().strip("|").split("|"):
        cell = _BR_RE.sub(" ", cell).replace("**", "").replace("*", "").replace("_", "")
        cell = " ".join(cell.split()).strip(" -–—")
        if cell:
            parts.append(cell)
    return " — ".join(parts)


def _linearize_markdown_tables(text: str) -> str:
    """Converte cada bloco de tabela markdown (>= 2 linhas `|...|`
    consecutivas) em uma frase por linha (`celula — celula.`), descartando
    `|`, `<br>`, `**` e a linha `|---|`. Uma linha vira chunk so se tiver
    marcador de citacao ou for longa o bastante para ser prosa — linhas de
    cabecalho/rotulo (sem citacao, curtas) sao descartadas. Prosa fora de
    tabela fica intocada."""
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        j = i
        while j < len(lines) and _TABLE_ROW_RE.match(lines[j]):
            j += 1
        if j - i < 2:
            out.append(lines[i])
            i += 1
            continue
        for k in range(i, j):
            if _TABLE_SEP_RE.match(lines[k]):
                continue
            has_marker = _ROW_HAS_MARKER_RE.search(lines[k]) is not None
            linear = _linearize_row(lines[k])
            if not linear or (not has_marker and len(_WORD_RE.findall(linear)) < 20):
                continue
            if not linear.rstrip().endswith((".", "!", "?", ":")):
                linear += "."
            out.extend(("", linear, ""))
        i = j
    return "\n".join(out)


def _normalize_citation_markers(text: str) -> str:
    """Converte `<sup>2</sup>`, `<sub>2</sub>` e superscripts unicode
    (²³¹...) para a forma canonica `[2]`, para que um unico regex
    reconheca todas as variacoes de marcador de citacao usadas pelas
    ferramentas de Deep Research — portado de
    `syntex/src/reference_extractor.py`."""
    text = _CITE_START_RE.sub("", text)
    text = _CITE_RE.sub(
        lambda m: "".join(f"[{n.strip()}]" for n in m.group(1).split(",") if n.strip()),
        text,
    )
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
        normalized = _normalize_citation_markers(_linearize_markdown_tables(text))
        marker_to_ref_id = _build_marker_index(references)
        matches = list(_BRACKET_RE.finditer(normalized))

        chunks: list[AnswerChunk] = []

        def emit(raw_text: str, ref_ids: list[str], *, has_marker: bool = False) -> None:
            body = _LEADING_NOISE_RE.sub("", _strip_tag_artifacts(raw_text)).strip()
            if not body:
                return
            pos = len(chunks)
            chunks.append(
                AnswerChunk(
                    id=f"{answer_id}-{pos}",
                    answer_id=answer_id,
                    position=pos,
                    text=body,
                    cited_reference_ids=ref_ids,
                    sentence_count=_count_sentences(body),
                    is_uncited_claim=not has_marker and not ref_ids and _looks_like_claim(body),
                )
            )

        if not matches:
            stripped = normalized.strip()
            for para in _split_paragraphs(stripped) or [stripped]:
                emit(para, [])
            return chunks

        cursor = 0
        for markers, run_start, run_end in _group_runs(matches, normalized):
            raw_span = normalized[cursor:run_start]
            cursor = run_end
            span = raw_span.strip()
            if not span:
                continue
            ref_ids = _resolve_markers(markers, marker_to_ref_id)
            pieces = _split_paragraphs(span)
            if len(pieces) == 1:
                emit(span, ref_ids, has_marker=True)
                continue
            # O marcador `[N]` sustenta o paragrafo em que aparece (o ultimo
            # pedaco do vao). Se o vao NAO comeca com linha em branco, o
            # primeiro pedaco e a cauda do paragrafo da citacao anterior —
            # junta no ancorado. Os pedacos do meio sao paragrafos proprios
            # sem citacao nenhuma -> "Afirmacoes sem Citacao".
            anchored, lead = pieces[-1], pieces[:-1]
            starts_fresh = re.match(r"\s*\n[ \t]*\n", raw_span) is not None
            if not starts_fresh and lead:
                anchored = lead[0] + "\n\n" + anchored
                lead = lead[1:]
            for para in lead:
                if chunks and _looks_like_claim(para):
                    emit(para, [])  # afirmacao propria, entre dois paragrafos citados
                elif not _HEADER_RE.match(para):
                    anchored = para + "\n\n" + anchored  # nao perde conteudo
            emit(anchored, ref_ids, has_marker=True)

        trailing = normalized[cursor:].strip()
        for para in _split_paragraphs(trailing):
            if _looks_like_claim(para) or not chunks:
                emit(para, [])
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
