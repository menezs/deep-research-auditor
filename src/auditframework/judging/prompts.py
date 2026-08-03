from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class JudgeOutput(BaseModel):
    """Schema da saida estruturada exigida do LLM juiz."""

    verdict: Literal["supported", "unsupported", "contradicted"]
    justification: str
    cited_excerpts: list[str] = Field(default_factory=list)


JUDGE_SYSTEM_MESSAGE = (
    "You are a strict evidence-grounded verification system. Your only "
    "task is to classify whether a CLAIM — an excerpt (TRECHO) from a Deep "
    "Research tool's answer — is supported by the EVIDENCE (CONTEXTO) "
    "retrieved for it.\n\n"
    "You MUST evaluate the claim using ONLY the provided evidence. You "
    "MUST completely ignore:\n"
    "- prior conversation context\n"
    "- chat history\n"
    "- world knowledge\n"
    "- assumptions\n"
    "- common sense not explicitly supported by the evidence\n"
    "- information outside the evidence block\n\n"
    "Your task is not to judge whether the claim is true in reality, but "
    "whether it is grounded in the evidence."
)


def build_judge_prompt(chunk_text: str, curated_context: str) -> str:
    return (
        f'CLAIM (TRECHO A AVALIAR):\n"""\n{chunk_text}\n"""\n\n'
        f'EVIDENCE (CONTEXTO — unica fonte de evidencia permitida):\n"""\n{curated_context}\n"""\n\n'
        "Classifique o veredito (verdict) em uma das categorias:\n"
        "- supported: a evidencia sustenta explicitamente a afirmacao, ou ela pode ser diretamente "
        "inferida SOMENTE a partir da evidencia (total ou na parte essencial dela).\n"
        "- unsupported: a evidencia nao contem informacao suficiente para confirmar nem negar a "
        "afirmacao. Isso inclui os casos em que a afirmacao pode ser verdadeira na realidade mas nao "
        "esta fundamentada na evidencia, a evidencia e incompleta/vaga/insuficiente, a sustenta apenas "
        "parcialmente, ou a verificacao exigiria conhecimento externo ou suposicoes.\n"
        "- contradicted: a evidencia entra explicitamente em conflito com a afirmacao ou a refuta.\n\n"
        "Regras criticas:\n"
        "- Trate a evidencia como a UNICA fonte de verdade.\n"
        "- Nunca use conhecimento externo, nem informacao de mensagens ou contexto anteriores.\n"
        "- Nunca infira fatos ausentes a nao ser que sejam diretamente sustentados pela evidencia.\n"
        "- Se a evidencia for ambigua ou incompleta, classifique como unsupported.\n"
        "- Em caso de duvida, prefira unsupported a supported.\n"
        "- Classifique como contradicted somente quando a evidencia conflitar claramente com a afirmacao.\n\n"
        "Em cited_excerpts, inclua trechos LITERAIS copiados da EVIDENCE que "
        "embasam seu veredito (lista vazia se o veredito for unsupported). "
        "Escreva a justificativa em portugues, de forma objetiva e curta."
    )
