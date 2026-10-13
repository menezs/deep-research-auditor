"""Normalizacao da resposta de Deep Research para o markdown canonico que o
pipeline espera, qualquer que seja a ferramenta ou o formato de origem.

Roda logo depois do loader (md/pdf/docx) e antes da extracao de referencias
e do chunking. Nada some sem registro: cada transformacao e contada no
`NormalizationReport`, gravado por execucao junto com o texto original."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

# Tokens internos do ChatGPT: `citeturn9view0turn15search0`.
# A cabeca observada no corpus e sempre `cite`; `filecite` e a variante para
# arquivos anexados. O espaco antes do bloco sai junto, para nao deixar
# "IA.[1]  A" com espaco duplo.
_TOOL_TOKEN_BLOCK_RE = re.compile(r"[ \t]*(?:cite|filecite)(?:[^]*)*")
# Mesmo token depois de um copiar-e-colar que perdeu os caracteres de uso
# privado: `citeturn9view2turn10search1`. Exige ao menos um `turnNxxxN`
# colado, para nunca casar a palavra "cite" em prosa.
_TOOL_TOKEN_PLAIN_RE = re.compile(r"[ \t]*\bcite(?:turn\d+[a-z]+\d+)+\b")
_PRIVATE_USE_RE = re.compile(r"[-]")


def remove_tool_tokens(text: str) -> tuple[str, int, int]:
    """Remove os tokens de citacao interna do ChatGPT e devolve
    `(texto, blocos_removidos, caracteres_privados_avulsos)`.

    Bloco com cabeca desconhecida (ex: `entity`) NAO e apagado inteiro: pode
    carregar texto legivel. Dele saem so os caracteres de uso privado,
    contados a parte para que o relatorio mostre que havia algo estranho."""
    removed = 0

    def _drop(_match: re.Match) -> str:
        nonlocal removed
        removed += 1
        return ""

    text = _TOOL_TOKEN_BLOCK_RE.sub(_drop, text)
    text = _TOOL_TOKEN_PLAIN_RE.sub(_drop, text)
    stray = len(_PRIVATE_USE_RE.findall(text))
    return _PRIVATE_USE_RE.sub("", text), removed, stray


UNICODE_SUPERSCRIPT_MAP = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
}
_DIGIT_TO_SUPERSCRIPT = {digit: sup for sup, digit in UNICODE_SUPERSCRIPT_MAP.items()}
_UNICODE_SUPERSCRIPT_RE = re.compile("[" + "".join(UNICODE_SUPERSCRIPT_MAP) + "]+")
_NUMBER_LIST = r"\d{1,3}(?:\s*[,;]\s*\d{1,3})+|\d{1,3}\s*[–—-]\s*\d{1,3}"
# Gemini/Docs: `[cite_start]` abre o trecho e `[cite: 12, 13]` fecha.
_CITE_START_RE = re.compile(r"\[cite_start\]", re.IGNORECASE)
_CITE_RE = re.compile(r"\[cite:\s*(" + _NUMBER_LIST + r"|\d{1,3})\s*\]", re.IGNORECASE)
_SUP_RE = re.compile(r"<sup>\s*(" + _NUMBER_LIST + r"|\d{1,3})\s*</sup>", re.IGNORECASE)
_SUB_RE = re.compile(r"<sub>\s*(\d{1,3})\s*</sub>", re.IGNORECASE)
_GROUPED_RE = re.compile(r"\[(" + _NUMBER_LIST + r")\]")
_BRACKET_MARKER_RE = re.compile(r"\[\d{1,3}\]")
# Expoente, nao citacao: `m²`, `km³`, `O(n²)`, `n·d²`, `f(x)³`. So 2 e 3, logo
# depois de uma ou duas letras minusculas soltas (unidade ou variavel) ou de
# um `)`. Citacao em sobrescrito vem colada a uma palavra ("casos²") ou a
# pontuacao ("ano.²"), nao a uma letra isolada.
_EXPONENT_PREFIX_RE = re.compile(r"(?:^|[\s\d(·/,])[a-z]{1,2}$|\)$")
_PREFIX_WINDOW = 12
_MAX_RANGE = 50


def to_superscript(digits: str) -> str:
    return "".join(_DIGIT_TO_SUPERSCRIPT[d] for d in digits)


def is_exponent(prefix: str, digits: str) -> bool:
    """`True` se o numero sobrescrito depois de `prefix` e expoente de
    unidade ou variavel, e nao marcador de citacao."""
    return digits in ("2", "3") and bool(_EXPONENT_PREFIX_RE.search(prefix))


def _expand(spec: str) -> list[str] | None:
    """`'1, 2'` -> `['1', '2']`; `'3-5'` -> `['3', '4', '5']`. `None` quando a
    forma nao e uma lista de citacoes (intervalo invertido ou grande demais)."""
    spec = spec.strip()
    interval = re.fullmatch(r"(\d{1,3})\s*[–—-]\s*(\d{1,3})", spec)
    if interval:
        start, end = int(interval.group(1)), int(interval.group(2))
        return [str(n) for n in range(start, end + 1)] if 0 < end - start <= _MAX_RANGE else None
    parts = [p for p in re.split(r"\s*[,;]\s*", spec) if p]
    return parts if parts and all(p.isdigit() for p in parts) else None


def _brackets(numbers: list[str]) -> str:
    return "".join(f"[{n}]" for n in numbers)


@dataclass
class MarkerStats:
    grouped_expanded: int = 0
    superscripts_converted: int = 0
    exponents_preserved: int = 0


def normalize_markers(text: str, stats: MarkerStats | None = None) -> str:
    """Leva todas as variacoes de marcador de citacao a forma canonica `[N]`.

    Sobrescrito Unicode (`¹`) so vira citacao quando o corpo da resposta NAO
    usa colchetes: num corpo que ja cita com `[N]`, `²` e expoente (`m²`,
    `O(n²)`) — converter gerava citacoes falsas da referencia [2]. Em qualquer
    caso, um expoente reconhecido por `is_exponent` e preservado. Idempotente."""
    from .reference_extractor import body_section_start

    stats = stats if stats is not None else MarkerStats()
    cut = body_section_start(text)
    body_uses_brackets = bool(_BRACKET_MARKER_RE.search(text if cut is None else text[:cut]))

    def numbers_or_original(match: re.Match) -> str:
        numbers = _expand(match.group(1))
        return _brackets(numbers) if numbers else match.group(0)

    def grouped(match: re.Match) -> str:
        numbers = _expand(match.group(1))
        if not numbers:
            return match.group(0)
        stats.grouped_expanded += 1
        return _brackets(numbers)

    def sup(match: re.Match) -> str:
        spec = match.group(1).strip()
        if is_exponent(match.string[max(0, match.start() - _PREFIX_WINDOW) : match.start()], spec):
            stats.exponents_preserved += 1
            return to_superscript(spec)
        numbers = _expand(spec)
        if not numbers:
            return match.group(0)
        stats.superscripts_converted += 1
        return _brackets(numbers)

    def unicode_sup(match: re.Match) -> str:
        digits = "".join(UNICODE_SUPERSCRIPT_MAP[c] for c in match.group(0))
        if is_exponent(match.string[max(0, match.start() - _PREFIX_WINDOW) : match.start()], digits):
            stats.exponents_preserved += 1
            return match.group(0)
        stats.superscripts_converted += 1
        return f"[{digits}]"

    text = _CITE_START_RE.sub("", text)
    text = _CITE_RE.sub(numbers_or_original, text)
    text = _SUP_RE.sub(sup, text)
    text = _SUB_RE.sub(lambda m: f"[{m.group(1)}]", text)
    text = _GROUPED_RE.sub(grouped, text)
    if not body_uses_brackets:
        text = _UNICODE_SUPERSCRIPT_RE.sub(unicode_sup, text)
    return text


_MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s+\S")
_ENTRY_PREFIX_RE = re.compile(r"^(?:[-*]\s+)?(?:\[\d{1,3}\]\s*|\d{1,3}[.)]\s+)")
_BRACKET_NUMBER_RE = re.compile(r"\[(\d{1,3})\]")
_WORD_RE = re.compile(r"[a-z0-9]{4,}")
_URL_IN_TEXT_RE = re.compile(r"https?://\S+")
# Janela de texto antes do marcador comparada com o titulo da entrada; e a
# vantagem minima que a melhor lista precisa ter sobre a segunda para a
# escolha ser por evidencia. Medido no Gemini `vacinacao_hpv`: 0.136 (lista
# certa) contra 0.088 — razao 1.5.
_EVIDENCE_WINDOW = 250
_MIN_EVIDENCE_MARGIN = 1.2


class ReferenceListInfo(BaseModel):
    heading: str
    entries: int
    role: Literal["primary", "secondary", "unresolved"] = "unresolved"
    evidence_score: float | None = None


class NormalizationReport(BaseModel):
    """O que a normalizacao mudou numa resposta. Gravado por execucao em
    `normalization_report.json`, ao lado do texto original (`input/answer_raw.md`)."""

    tool_tokens_removed: int = 0
    stray_private_chars_removed: int = 0
    grouped_markers_expanded: int = 0
    superscripts_converted: int = 0
    exponents_preserved: int = 0
    reference_lists: list[ReferenceListInfo] = Field(default_factory=list)
    primary_list_choice: Literal["single", "evidence", "position", "asterism"] | None = None
    dropped_list_lines: int = 0
    body_markers: int = 0
    body_markers_without_entry: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


@dataclass
class _ListSection:
    start: int
    end: int
    heading: str
    entries: list[str]
    dropped_lines: int = 0


def _reference_list_sections(text: str) -> list[_ListSection]:
    """Secoes de lista de fontes depois do corte do corpo: cada cabecalho de
    referencias reconhecido pelo extrator e as linhas com URL ate o proximo
    cabecalho. Linhas sem URL dentro da secao sao contadas, nao lidas."""
    from .reference_extractor import (
        _BARE_REFERENCE_HEADING,
        _REFERENCE_SECTION_HEADING,
        _dominated_by_urls,
        body_section_start,
    )

    cut = body_section_start(text)
    if cut is None:
        return []
    sections: list[_ListSection] = []
    current: _ListSection | None = None
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        is_reference_heading = offset >= cut and (
            bool(_REFERENCE_SECTION_HEADING.match(stripped))
            or (bool(_BARE_REFERENCE_HEADING.match(stripped)) and _dominated_by_urls(text[offset:]))
        )
        if is_reference_heading:
            current = _ListSection(start=offset, end=offset + len(line), heading=stripped, entries=[])
            sections.append(current)
        elif _MARKDOWN_HEADING_RE.match(stripped):
            current = None
        elif current is not None and stripped:
            if "://" in stripped:
                current.entries.append(stripped)
            else:
                current.dropped_lines += 1
            current.end = offset + len(line)
        offset += len(line)
    return [s for s in sections if s.entries]


def _tokens(text: str) -> set[str]:
    folded = unicodedata.normalize("NFKD", _URL_IN_TEXT_RE.sub(" ", text.lower()))
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return set(_WORD_RE.findall(folded))


def _list_evidence(body: str, entries: list[str]) -> float:
    """Quanto a lista explica as citacoes do corpo: para cada `[N]`, a fracao
    das palavras do titulo da entrada N que aparecem no texto logo antes do
    marcador. Marcador sem entrada N conta zero."""
    scores: list[float] = []
    for match in _BRACKET_NUMBER_RE.finditer(body):
        number = int(match.group(1))
        if not 1 <= number <= len(entries):
            scores.append(0.0)
            continue
        entry_tokens = _tokens(entries[number - 1])
        if not entry_tokens:
            continue
        window = _tokens(body[max(0, match.start() - _EVIDENCE_WINDOW) : match.start()])
        scores.append(len(entry_tokens & window) / len(entry_tokens))
    return sum(scores) / len(scores) if scores else 0.0


def _strip_prefix(entry: str) -> str:
    return _ENTRY_PREFIX_RE.sub("", entry, count=1).strip()


def resolve_reference_lists(text: str, report: NormalizationReport) -> str:
    """Com duas ou mais listas de fontes, escolhe a que o corpo cita e
    reescreve o fim do documento no formato canonico que o extrator le pelo
    caminho mais estavel:

        ## Referências
        [1] Titulo, URL          <- lista principal, numerada na ordem dela

        ⁂
        Titulo. URL              <- demais listas: secundarias (convencao do Perplexity)

    A escolha e por evidencia (`_list_evidence`), nunca por posicao: no
    Gemini a lista que o corpo segue ("Referências citadas") vem DEPOIS da
    lista pedida no prompt. Sem evidencia suficiente, fica a ultima lista e o
    relatorio avisa. Documento que ja tem `⁂`, ou uma lista so, ou corpo sem
    marcadores, sai intacto."""
    from .reference_extractor import _ASTERISM

    sections = _reference_list_sections(text)
    report.reference_lists = [ReferenceListInfo(heading=s.heading, entries=len(s.entries)) for s in sections]
    if not sections:
        return text
    if _ASTERISM in text:
        report.primary_list_choice = "asterism"
        return text
    if len(sections) == 1:
        report.primary_list_choice = "single"
        report.reference_lists[0].role = "primary"
        return text

    body = text[: sections[0].start]
    if not _BRACKET_NUMBER_RE.search(body):
        return text  # sem citacoes nao ha evidencia; o aviso sai em `normalize_answer`

    scores = [_list_evidence(body, s.entries) for s in sections]
    ranking = sorted(range(len(sections)), key=lambda i: scores[i], reverse=True)
    best, runner_up = ranking[0], ranking[1]
    if scores[best] > 0 and scores[best] >= _MIN_EVIDENCE_MARGIN * scores[runner_up]:
        primary, report.primary_list_choice = best, "evidence"
    else:
        primary, report.primary_list_choice = len(sections) - 1, "position"
        report.warnings.append(
            "lista principal escolhida pela posicao (a ultima): a evidencia nao distinguiu as "
            f"{len(sections)} listas — conferir manualmente"
        )
    for i, info in enumerate(report.reference_lists):
        info.role = "primary" if i == primary else "secondary"
        info.evidence_score = scores[i]

    gaps = sum(
        1 for a, b in zip(sections, sections[1:]) for line in text[a.end : b.start].splitlines() if line.strip()
    )
    report.dropped_list_lines = sum(s.dropped_lines for s in sections) + gaps

    canonical = [f"[{n}] {_strip_prefix(e)}" for n, e in enumerate(sections[primary].entries, 1)]
    secondary = [_strip_prefix(e) for i, s in enumerate(sections) if i != primary for e in s.entries]
    rebuilt = (
        body.rstrip() + "\n\n## Referências\n\n" + "\n".join(canonical)
        + f"\n\n{_ASTERISM}\n\n" + "\n".join(secondary) + "\n"
    )
    trailing = text[sections[-1].end :]
    if trailing.strip():
        rebuilt += "\n" + trailing.lstrip("\n")
    return rebuilt


def normalize_answer(text: str) -> tuple[str, NormalizationReport]:
    """Texto do loader -> markdown canonico + relatorio do que mudou."""
    from .reference_extractor import body_section_start

    report = NormalizationReport()
    text, report.tool_tokens_removed, report.stray_private_chars_removed = remove_tool_tokens(text)
    stats = MarkerStats()
    text = normalize_markers(text, stats)
    report.grouped_markers_expanded = stats.grouped_expanded
    report.superscripts_converted = stats.superscripts_converted
    report.exponents_preserved = stats.exponents_preserved
    text = resolve_reference_lists(text, report)

    cut = body_section_start(text)
    body = text if cut is None else text[:cut]
    report.body_markers = len(set(_BRACKET_NUMBER_RE.findall(body)))
    if report.body_markers == 0:
        report.warnings.append("corpo sem marcadores de citacao [N]: nenhuma afirmacao sera auditada")
    return text, report


def add_extraction_warnings(
    report: NormalizationReport, *, cited_markers: set[str], listed_markers: set[str]
) -> None:
    """Completa o relatorio depois da extracao: marcadores que o corpo cita
    e que nenhuma entrada da lista principal reivindica. Normalizar nao
    resolve isso (a lista exportada simplesmente nao tem a entrada); o aviso
    deixa explicito por que essas afirmacoes serao puladas."""
    missing = sorted(cited_markers - listed_markers, key=lambda m: int(m.strip("[]")))
    report.body_markers_without_entry = missing
    if missing:
        report.warnings.append(
            f"{len(missing)} de {len(cited_markers)} marcadores citados no corpo sem entrada na lista de "
            f"referencias: {', '.join(missing[:10])}" + (" ..." if len(missing) > 10 else "")
        )
