"""Deteccao de barreiras de acesso no conteudo baixado.

O status HTTP nao basta: anti-bots, paginas de erro servidas com 200,
avisos de "habilite o JavaScript" e intersticios de download chegam como
HTML valido, e o Playwright grava qualquer pagina com status 200. Sem esta
checagem, a tela "Checking your browser..." vira a evidencia contra a qual
o juiz avalia a afirmacao — um UNSUPPORTED causado pela coleta, nao pela
ferramenta auditada.

Os padroes so valem para documentos curtos (`MAX_BARRIER_WORDS`): uma tela
de bloqueio tem poucas dezenas de palavras, e um artigo longo que fala de
"captcha" ou "cookies" nao pode ser descartado por isso. Abaixo de
`MIN_CONTENT_WORDS` palavras, sem padrao nenhum, o documento tambem e
tratado como sem conteudo: nao ha o que sustentar uma afirmacao.

Padroes levantados nos 740 documentos dos runs de 18/10/2026."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BarrierKind = Literal[
    "anti_bot", "http_error", "requires_js_login_region", "interstitial", "empty_shell", "no_content"
]

MAX_BARRIER_WORDS = 300
MIN_CONTENT_WORDS = 25
_EVIDENCE_CHARS = 120

# Ordem importa: o primeiro grupo que casar define o tipo.
_PATTERNS: tuple[tuple[BarrierKind, tuple[str, ...]], ...] = (
    ("anti_bot", (
        "security service", "checking your browser", "verificando sua conex", "not a bot", "not for bots",
        "just a moment", "captcha", "attention required", "cf-browser-verification", "are you a robot",
        "unusual traffic", "verify that you're not a robot", "verify you are human", "incapsula",
        "request unsuccessful",
    )),
    ("http_error", (
        "403 forbidden", "error 403", "access denied", "access to this page is forbidden", "404 not found",
        "errors.edgesuite.net", "there was a problem providing the content",
    )),
    ("requires_js_login_region", (
        "javascript is disabled", "depende do javascript", "enable javascript to run", "necessário autenticar",
        "access limited to the u.s", "not available outside",
    )),
    ("interstitial", (
        "not redirected", "not automatically redirected", "now downloading", "continue to ", "click here if",
        "has been relocated", "utiliza cookies", "uses cookies", "we use cookies", "accept cookies",
    )),
    ("empty_shell", ("how youtube works", "match case limit results")),
)


@dataclass(frozen=True)
class Barrier:
    kind: BarrierKind
    evidence: str


def detect_barrier(markdown: str) -> Barrier | None:
    """A barreira que o conteudo baixado representa, ou `None` quando ele e
    o documento de fato."""
    text = " ".join(markdown.split())
    words = len(text.split())
    if words >= MAX_BARRIER_WORDS:
        return None
    lowered = text.lower()
    evidence = text[:_EVIDENCE_CHARS]
    for kind, patterns in _PATTERNS:
        if any(pattern in lowered for pattern in patterns):
            return Barrier(kind, evidence)
    if words < MIN_CONTENT_WORDS:
        return Barrier("no_content", evidence)
    return None
