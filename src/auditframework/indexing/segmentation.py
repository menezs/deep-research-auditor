"""Deteccao de fronteira de sentenca em portugues, por regras.

Um `.` so encerra uma afirmacao quando e de fato fim de frase. NAO contam
como fronteira:

* `.` dentro de parenteses, colchetes ou aspas — ``a UE (Reg. UE 2024/1689)
  classifica``;
* `.` entre digitos — ``art. 1.238``, ``R$ 50.000``, ``v2.5``, ``PL 2.338/2023``;
* `.` logo apos uma abreviacao conhecida — ``art.``, ``Reg.``, ``etc.``;
* `.` nao seguido de um inicio de frase plausivel (espaco + maiuscula/aspa/
  marcador de lista/digito, ou fim do bloco).

Sem dependencia de NLP pesada: o corpus alvo e prosa bem-formada gerada por
LLM, e o controle fino sobre abreviacoes juridicas importa mais que a
robustez estatistica generica. Substitui o regex ingenuo
(`[.!?]+\\s+[A-Z]`) que o `AnswerChunker` usava para contar frases e que
nao distinguia `art. 5o` de um fim de frase real.
"""

from __future__ import annotations

import re

_TERMINATORS = ".!?…"
_OPEN_CLOSE = {"(": ")", "[": "]", "{": "}"}
_QUOTES = {'"', "“", "”", "'", "«", "»"}

# Abreviacoes (sem o ponto final, minusculas). Um `.` logo apos qualquer
# uma delas nao e fim de frase.
_ABBREVIATIONS = {
    # juridico / referencias
    "art", "arts", "inc", "incs", "par", "pars", "cf", "op", "cit", "ibid",
    "apud", "ss", "seg", "segs", "cap", "caps", "ed", "eds", "vol", "vols",
    "n", "no", "nro", "num", "p", "pp", "pag", "pág", "págs", "pags",
    "reg", "dec", "res", "port", "súm", "sum",
    # tratamento
    "dr", "dra", "drs", "sr", "sra", "srs", "sras", "prof", "profa", "profs",
    "exmo", "exma",
    # gerais
    "etc", "ex", "vs", "obs", "fig", "figs", "tab", "tabs", "aprox", "ref",
    "refs", "i.e", "e.g",
    # meses
    "jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out",
    "nov", "dez",
    # medidas / enderecos
    "av", "r", "pça", "est", "km", "m", "cm", "mm", "kg", "g",
}

_WORD_BEFORE_RE = re.compile(r"([0-9A-Za-zÀ-ÿ.]+)$")
_SENTENCE_START_RE = re.compile(r'[0-9A-ZÀ-Þ"“«\'(\[]|[-*] ')


def _is_abbreviation(prefix: str) -> bool:
    word = prefix.rstrip(".").lower()
    if not word:
        return False
    if word in _ABBREVIATIONS:
        return True
    if len(word) == 1 and word.isalpha():
        return True  # inicial isolada: "J. Silva", "A. C. Souza"
    parts = word.split(".")
    return len(parts) > 1 and all(len(p) <= 2 and p.isalpha() for p in parts if p)


def sentence_boundaries(text: str) -> list[int]:
    """Indices onde comeca uma nova sentenca (posicao do primeiro caractere
    apos o terminador e os espacos que o seguem)."""
    boundaries: list[int] = []
    stack: list[str] = []
    quote_open = False
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if ch in _OPEN_CLOSE:
            stack.append(_OPEN_CLOSE[ch])
        elif stack and ch == stack[-1]:
            stack.pop()
        elif ch in _QUOTES:
            quote_open = not quote_open
        elif ch in _TERMINATORS:
            j = i
            while j + 1 < n and text[j + 1] in _TERMINATORS:
                j += 1  # colapsa "..." / "?!"
            end = _evaluate_terminator(text, i, j, inside=bool(stack) or quote_open)
            if end is not None:
                boundaries.append(end)
            i = j + 1
            continue
        i += 1
    return boundaries


def _evaluate_terminator(text: str, start: int, end: int, *, inside: bool) -> int | None:
    if inside:
        return None

    if text[start] == ".":
        previous = text[start - 1] if start > 0 else ""
        following = text[end + 1] if end + 1 < len(text) else ""
        if previous.isdigit() and following.isdigit():
            return None
        match = _WORD_BEFORE_RE.search(text[:start])
        if match and _is_abbreviation(match.group(1)):
            return None

    k = end + 1
    # pula fecha-parenteses/aspas colados ao terminador: `etc.).`, `cabo".`
    while k < len(text) and (text[k] in ")]}" or text[k] in _QUOTES):
        k += 1
    next_start = k
    while k < len(text) and text[k] in " \t\n\r":
        k += 1

    if k >= len(text):
        return next_start  # terminador no fim do bloco
    if _SENTENCE_START_RE.match(text, k):
        return k
    return None
