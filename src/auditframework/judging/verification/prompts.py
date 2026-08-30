from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class WindowAssessment(BaseModel):
    """Saida estruturada da varredura de UMA janela do documento citado
    (Etapa B)."""

    relation: Literal["supports", "contradicts", "not_addressed"]
    justification: str
    cited_excerpts: list[str] = Field(default_factory=list)


WINDOW_SCAN_SYSTEM_MESSAGE = (
    "You are a strict evidence-grounded verification system. You receive a "
    "CLAIM taken from a Deep Research answer and ONE excerpt window from the "
    "source document that the answer cites for it. Decide, using ONLY this "
    "window, whether it:\n"
    "- supports: the window explicitly states the claim, or the claim follows "
    "directly and solely from this window;\n"
    "- contradicts: the window explicitly conflicts with or refutes the claim;\n"
    "- not_addressed: the window neither supports nor contradicts the claim "
    "(it may be about a different topic, or simply not contain the fact).\n\n"
    "Ignore world knowledge, prior context and anything outside this window. "
    "In cited_excerpts put literal fragments copied from the window that "
    "justify a supports/contradicts decision (empty list for not_addressed). "
    "Write the justification in Portuguese, short and objective."
)


def build_window_scan_prompt(claim: str, window: str) -> str:
    return (
        f'CLAIM (TRECHO A AVALIAR):\n"""\n{claim}\n"""\n\n'
        f'JANELA DO DOCUMENTO CITADO (unica evidencia permitida):\n"""\n{window}\n"""\n\n'
        "Classifique a relacao (relation) como supports, contradicts ou not_addressed."
    )
