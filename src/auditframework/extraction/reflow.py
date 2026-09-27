"""Desfaz as quebras de linha espurias que o `pymupdf4llm` insere no meio
de uma frase ou de um item de lista ao converter um PDF: cada linha VISUAL
do PDF vira uma linha do Markdown, as vezes com uma linha em branco e ate
um novo `- ` no meio de uma sentenca.

Sem isso, uma frase quebrada ao meio vira dois blocos para o
`AnswerChunker` (que separa por linha em branco), e a fronteira de sentenca
nao tem como fechar a frase — o trecho ancorado pela citacao sai cortado, e
a metade orfa aparece como "afirmacao sem citacao" que nao existe no
original.

Conservador por construcao: so junta a proxima linha na anterior quando a
anterior claramente NAO terminou (sem pontuacao final, sem fechar tag/
heading/tabela) e a proxima claramente CONTINUA (comeca em minuscula,
digito ou `(`, tolerando um marcador de lista espurio na frente). Prosa ja
correta, listas de verdade, cabecalhos e tabelas passam intactos.
"""

from __future__ import annotations

import re

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s")
_TABLE_RE = re.compile(r"^\s*\|| \| ")
_HR_RE = re.compile(r"^\s*([-*_])\1{2,}\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d{1,3}[.)])\s+")

# A linha anterior "fecha" se termina com pontuacao final, aspas de
# fechamento, `)` ou uma tag (`</u>`, `>`).
_CLOSES_RE = re.compile(r"[.!?:;)\]”\"’»>]\s*$")
# A proxima "continua" se, tirado um marcador de lista espurio, comeca em
# minuscula / digito / `(`.
_CONTINUES_RE = re.compile(r"^[0-9(a-zà-ÿçãõáéíóúâêôàäëïöü]")


def _is_structural(line: str) -> bool:
    return bool(
        _HEADING_RE.match(line)
        or _TABLE_RE.match(line)
        or _HR_RE.match(line)
        or _FENCE_RE.match(line)
    )


def _continues(previous: str, upcoming: str, blank_lines: int) -> bool:
    if blank_lines > 1:
        return False
    tail = previous.rstrip()
    if not tail or _is_structural(tail) or _CLOSES_RE.search(tail):
        return False
    if _is_structural(upcoming):
        return False
    body = _BULLET_RE.sub("", upcoming.strip())
    if body.startswith(("http://", "https://", "www.")):
        return False
    return bool(_CONTINUES_RE.match(body)) or tail.endswith("-")


def reflow_markdown(text: str) -> str:
    out: list[str] = []
    blank = 0
    for raw in text.split("\n"):
        line = raw.rstrip()
        if not line.strip():
            blank += 1
            continue
        if out and _continues(out[-1], line, blank):
            body = _BULLET_RE.sub("", line.strip())
            if out[-1].rstrip().endswith("-"):
                # hifen no ponto de quebra: junta sem espaco ("transparente-p" + "ara")
                out[-1] = out[-1].rstrip()[:-1] + body
            else:
                out[-1] = out[-1].rstrip() + " " + body
        else:
            if blank and out:
                out.append("")  # colapsa multiplas linhas em branco em uma
            out.append(line)
        blank = 0
    return "\n".join(out)
