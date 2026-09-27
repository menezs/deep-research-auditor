"""Restaura a posicao dos marcadores de citacao sobrescritos que a conversao
PDF->Markdown emite fora da ordem de leitura.

Num PDF do Perplexity, `[1] [2] [3] [4]` e um objeto de linha PROPRIO, com
fonte menor, colocado a DIREITA da linha que ele segue:

    linha A  y 315.6-326.6  x  68-497   "...with the greatest protection when"
    linha B  y 331.4-342.3  x  68-233   "vaccination occurs before age 17."
    linha C  y 328.8-336.6  x 234-277   "[1] [2] [3] [4]"        <- fonte 7.9

A linha C pertence ao fim da linha B (comeca em x=234, logo depois do
x=233 onde B termina), mas seu topo e MAIS ALTO que o de B, porque
sobrescrito. Qualquer ordenacao por posicao vertical emite C antes de B:

    "...protection when [1] [2] [3] [4] vaccination occurs before age 17."

Isso joga o marcador para o meio da frase. O `AnswerChunker` ancora o texto
que vem ANTES do marcador, entao a afirmacao julgada perde justamente o
final — no exemplo, "vaccination occurs before age 17", que e a condicao que
da sentido ao dado. O `pymupdf4llm` 1.28 nao expoe opcao de ordenacao que
resolva isso, dai o reparo aqui.

A deteccao e geometrica (verdade do PDF); a aplicacao no Markdown ja
convertido e conservadora: sem casamento inequivoco, nao se move nada.
"""

from __future__ import annotations

import re

from ..logging_config import get_logger

logger = get_logger(__name__)

_MARKER_LINE_RE = re.compile(r"^\s*(?:\[\s*\d+\s*\]\s*)+$")
_NUMBER_RE = re.compile(r"\d+")
_ALNUM_RE = re.compile(r"[0-9A-Za-zÀ-ÿ]")
# Pontuacao que fecha a frase e deve ficar ANTES do marcador movido, como
# esta no PDF ("...age 17." + marcador).
_TRAILING_PUNCT = ".,;:!?)]}\"”’»"

# Sobreposicao vertical minima (pt) para considerar que o marcador esta na
# mesma faixa da linha — folga contra arredondamento do extrator.
_MIN_VERTICAL_OVERLAP = 2.0
# Um marcador sobrescrito e visivelmente menor que o corpo do texto.
_MAX_RELATIVE_SIZE = 0.9


def _line_text(line: dict) -> str:
    return "".join(span["text"] for span in line["spans"])


def _line_size(line: dict) -> float:
    return max((span["size"] for span in line["spans"]), default=0.0)


def misplaced_marker_runs(path) -> list[tuple[str, str]]:
    """Pares `(marcadores, texto da linha a que pertencem)` para os
    marcadores que a conversao vai emitir ANTES da linha correta.

    Qualquer falha de leitura do PDF devolve `[]`: o reparo e uma melhoria
    best-effort e nunca pode derrubar o carregamento da resposta."""
    try:
        import pymupdf
    except ImportError:
        return []
    try:
        doc = pymupdf.open(path)
    except Exception as exc:
        logger.warning("Nao foi possivel abrir %s para checar a ordem dos marcadores: %s", path, exc)
        return []

    pairs: list[tuple[str, str]] = []
    with doc:
        for page in doc:
            try:
                lines = [ln for block in page.get_text("dict")["blocks"] for ln in block.get("lines", [])]
            except Exception:  # pagina corrompida: ignora esta, nao o documento
                continue
            marker_lines = [ln for ln in lines if _MARKER_LINE_RE.match(_line_text(ln))]
            if not marker_lines:
                continue
            body_size = max((_line_size(ln) for ln in lines if ln not in marker_lines), default=0.0)

            for marker in marker_lines:
                if body_size and _line_size(marker) > body_size * _MAX_RELATIVE_SIZE:
                    continue  # mesmo tamanho do corpo: nao e sobrescrito
                mx0, my0, my1 = marker["bbox"][0], marker["bbox"][1], marker["bbox"][3]
                hosts = [
                    ln
                    for ln in lines
                    if ln is not marker
                    and not _MARKER_LINE_RE.match(_line_text(ln))
                    and min(my1, ln["bbox"][3]) - max(my0, ln["bbox"][1]) > _MIN_VERTICAL_OVERLAP
                    and ln["bbox"][2] <= mx0 + 1  # termina a esquerda do marcador
                ]
                if not hosts:
                    continue
                host = max(hosts, key=lambda ln: ln["bbox"][2])  # a mais proxima pela esquerda
                if host["bbox"][1] <= my0:
                    continue  # o marcador ja vem depois na ordenacao vertical
                pairs.append((_line_text(marker).strip(), _line_text(host).strip()))
    return pairs


def _alnum_key(text: str) -> str:
    return "".join(_ALNUM_RE.findall(text))


def _marker_run_pattern(markers: str) -> re.Pattern[str]:
    """Casa a sequencia de marcadores no Markdown, com as tags que o
    conversor põe em volta (`<sup><u>[1] [2]</u></sup>`).

    Nao absorve o espaco em volta: um `\\s*` nas pontas engoliria a quebra de
    paragrafo seguinte, e mover o marcador transformaria `\\n\\n- item` num
    ` - item` no meio da linha."""
    numbers = _NUMBER_RE.findall(markers)
    core = r"\s*".join(rf"\[\s*{n}\s*\]" for n in numbers)
    return re.compile(r"(?:<sup>\s*)?(?:<u>\s*)?" + core + r"(?:\s*</u>)?(?:\s*</sup>)?")


def _host_end_after(markdown: str, start: int, host: str) -> int | None:
    """Indice em `markdown`, a partir de `start`, onde o texto da linha
    hospedeira termina — ou `None` se o que vem depois do marcador nao for
    essa linha.

    Compara so os caracteres alfanumericos, na ordem, porque o conversor
    insere/remove espacos, tags e hifens livremente — mas exige casamento
    EXATO deles. Uma letra que o conversor perdeu (ligadura sem mapeamento
    ToUnicode: "before" virando "be?ore") desalinha a comparacao daquele
    ponto em diante, e nao ha como distinguir isso de "esta e outra linha".
    Nesse caso o marcador nao e movido; o `AnswerChunker` ainda recupera a
    afirmacao inteira pela cauda da frase, so nao corrige a posicao do
    marcador no texto de origem."""
    host_key = _alnum_key(host)
    if len(host_key) < 8:
        return None  # curto demais para identificar com seguranca

    matched = 0
    index = start
    while index < len(markdown) and matched < len(host_key):
        char = markdown[index]
        if _ALNUM_RE.match(char):
            if char != host_key[matched]:
                return None
            matched += 1
        index += 1

    if matched < len(host_key):
        return None
    # leva a pontuacao final da frase para antes do marcador, como no PDF
    while index < len(markdown) and markdown[index] in _TRAILING_PUNCT:
        index += 1
    return index


def repair_marker_order(markdown: str, pairs: list[tuple[str, str]]) -> str:
    """Move cada marcador de `pairs` para o fim da linha a que pertence.

    Conservador de proposito: o par e ignorado quando o marcador nao aparece
    exatamente uma vez seguido daquela linha (ambiguo), quando o texto que o
    segue nao e a linha esperada (a conversao pos o marcador em outro lugar,
    ou a colocou certa) ou quando a linha e curta demais para identificar.

    Nada e movido dentro da lista de fontes: ali o `[N]` vem ANTES do titulo
    da entrada por definicao do formato, e move-lo para depois quebraria o
    parser de referencias. A deteccao geometrica nao distingue corpo de lista
    — quem distingue e este corte, e ele usa `body_section_start` (o inicio
    mais cedo da regiao de fontes) em vez da ancora de leitura, que comeca
    depois de uma primeira lista quando ha duas."""
    from .reference_extractor import body_section_start  # ciclo: so em runtime

    start = body_section_start(markdown)
    body_limit = len(markdown) if start is None else start

    for markers, host in pairs:
        pattern = _marker_run_pattern(markers)
        moves = [
            (match, end)
            for match in pattern.finditer(markdown)
            if match.start() < body_limit
            and (end := _host_end_after(markdown, match.end(), host)) is not None
        ]
        if len(moves) != 1:
            if moves:
                logger.debug("ordem de marcador %s ambigua no Markdown (%d candidatos)", markers, len(moves))
            continue
        match, end = moves[0]
        run = match.group(0).strip()
        before, between, after = markdown[: match.start()], markdown[match.end() : end], markdown[end:]
        if before[-1:] in (" ", "\t") and between[:1] in (" ", "\t"):
            between = between.lstrip(" \t")  # nao deixa espaco duplo onde o marcador estava
        markdown = before + between + run + after
        logger.info("marcador %s movido para depois de %r", markers, host[-40:])
    return markdown
