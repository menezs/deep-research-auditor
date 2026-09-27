from __future__ import annotations

import re

from ..models import AnswerChunk, Reference, ReferenceChunk
from .segmentation import sentence_boundaries

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

# Wrappers de formatacao que o pymupdf4llm deixa ao redor de links e
# marcadores de citacao (`<u>[1] [2]</u>`) — sem valor semantico.
_TAG_ARTIFACT_RE = re.compile(r"</?(?:u|b|i|em|strong|span|mark|sup|sub)\s*>", re.IGNORECASE)
_HAS_ALNUM_RE = re.compile(r"[0-9A-Za-zÀ-ÿ]")


def _strip_tag_artifacts(text: str) -> str:
    return _TAG_ARTIFACT_RE.sub("", text).strip()


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
    `|`, `<br>`, `**` e a linha `|---|`. So sobrevive a linha que tem
    marcador de citacao — linha de cabecalho/rotulo e linha de dado sem
    fonte nao teriam destino nenhum (nao sao auditaveis). Prosa fora de
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
            if not _ROW_HAS_MARKER_RE.search(lines[k]):
                continue
            linear = _linearize_row(lines[k])
            if not linear:
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
    """Divide o texto da resposta em trechos ancorados por marcadores de
    citacao, e resolve cada marcador para o `Reference.id` estavel
    correspondente (nao a string bruta "[1]").

    Um marcador `[N]` ancora o texto que vai dele PARA TRAS ate a primeira
    das tres fronteiras: fim do marcador anterior no mesmo paragrafo,
    fronteira de sentenca real (ver `segmentation`), ou inicio do
    paragrafo. Assim o trecho julgado fica restrito a afirmacao que aquela
    citacao de fato sustenta.

    A versao anterior fatiava so pelos vaos entre marcadores, quebrando
    apenas em fronteira de PARAGRAFO: um paragrafo com tres frases e uma
    citacao ao final virava um unico chunk com as tres, e o juiz recebia
    afirmacoes extras que a fonte nunca foi citada para sustentar.

    Texto SEM marcador nao vira chunk: a auditoria e sobre a relacao entre
    uma afirmacao e a fonte que ela cita, e sem citacao nao existe essa
    relacao para verificar. Isso tambem descarta o que nao e afirmacao
    (cabecalho, rotulo de tabela, eco do proprio prompt, cauda de frase
    cortada por um marcador anterior) sem precisar adivinhar quais desses
    "parecem" uma afirmacao."""

    def chunk(self, text: str, *, answer_id: str, references: list[Reference]) -> list[AnswerChunk]:
        normalized = _normalize_citation_markers(_linearize_markdown_tables(text))
        marker_to_ref_id = _build_marker_index(references)
        chunks: list[AnswerChunk] = []

        def emit(raw_text: str, markers: list[str], ref_ids: list[str]) -> None:
            body = _LEADING_NOISE_RE.sub("", _strip_tag_artifacts(raw_text)).strip()
            if not body or not _HAS_ALNUM_RE.search(body):
                return
            pos = len(chunks)
            chunks.append(
                AnswerChunk(
                    id=f"{answer_id}-{pos}",
                    answer_id=answer_id,
                    position=pos,
                    text=body,
                    cited_markers=markers,
                    cited_reference_ids=ref_ids,
                )
            )

        for block in _split_paragraphs(normalized):
            if _HEADER_RE.match(block):
                continue
            runs = _group_runs(list(_BRACKET_RE.finditer(block)), block)
            if not runs:
                continue
            boundaries = sentence_boundaries(block)
            previous_end = 0
            for markers, run_start, run_end in runs:
                # Uma fronteira so corta se sobra texto entre ela e o
                # marcador: em "frase.[4]" o ponto encosta no marcador, e o
                # `[4]` ancora a frase que acabou de terminar — nao um
                # trecho vazio.
                candidates = [
                    b for b in boundaries if previous_end <= b <= run_start and block[b:run_start].strip()
                ]
                left = max(candidates) if candidates else previous_end
                emit(block[left:run_start], markers, _resolve_markers(markers, marker_to_ref_id))
                previous_end = run_end

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
