from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class JudgeOutput(BaseModel):
    """Schema da saida estruturada exigida do LLM juiz."""

    verdict: Literal["supported", "unsupported", "contradicted"]
    justification: str
    cited_excerpts: list[str] = Field(default_factory=list)


JUDGE_SYSTEM_MESSAGE = (
    "Voce e um juiz factual rigoroso. Sua unica tarefa e avaliar se um "
    "TRECHO de uma resposta de ferramenta de Deep Research e sustentado "
    "pelo CONTEXTO fornecido. Use exclusivamente o CONTEXTO como fonte de "
    "evidencia — ignore completamente qualquer conhecimento previo do "
    "mundo que voce possua sobre o assunto. Sua tarefa nao e julgar se a "
    "afirmacao e verdadeira, e sim se ela e sustentada pelo CONTEXTO."
)


def build_judge_prompt(chunk_text: str, curated_context: str) -> str:
    return (
        f"TRECHO A AVALIAR:\n{chunk_text}\n\n"
        f"CONTEXTO (unica fonte de evidencia permitida):\n{curated_context}\n\n"
        "Classifique o TRECHO em uma das categorias:\n"
        "- supported: o CONTEXTO sustenta explicitamente a afirmacao (total ou na parte essencial dela).\n"
        "- unsupported: o CONTEXTO nao contem evidencia suficiente para confirmar nem negar a afirmacao, "
        "ou o sustenta apenas parcialmente.\n"
        "- contradicted: o CONTEXTO contradiz explicitamente a afirmacao.\n\n"
        "Em cited_excerpts, inclua trechos LITERAIS copiados do CONTEXTO que "
        "embasam seu veredito (lista vazia se o veredito for unsupported). "
        "Escreva a justificativa em portugues, de forma objetiva e curta."
    )
