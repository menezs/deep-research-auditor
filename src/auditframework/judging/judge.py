from __future__ import annotations

from ..common.llm_client import LLMClient
from ..logging_config import get_logger
from ..models import AnswerChunk, AuditResult, AuditVerdict, CuratedDocument
from .prompts import JUDGE_SYSTEM_MESSAGE, JudgeOutput, build_judge_prompt

logger = get_logger(__name__)


class Verifier:
    """Juiz LLM: classifica um AnswerChunk como suportado/nao
    suportado/contraditado pelo CuratedDocument correspondente, em apenas
    3 categorias de conteudo (AuditVerdict).

    Corrige o bug mais grave do audit_with_llm atual: saida do LLM nao
    parseavel nunca e coagida silenciosamente para CONTRADICTED — a
    excecao (`LLMParseError`) propaga para fora de `verify`, deixando o
    chunk pendente para uma proxima `audit resume`, em vez de virar um
    veredito de conteudo. `cited_excerpts` e um campo de primeira classe
    da saida, nao um campo ausente (hoje `passages` so guarda o documento
    inteiro)."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def verify(self, chunk: AnswerChunk, curated: CuratedDocument) -> AuditResult:
        if not curated.passages:
            logger.info("Chunk %s sem trechos recuperados — unsupported sem chamar o LLM", chunk.id)
            return AuditResult(
                answer_chunk_id=chunk.id,
                verdict=AuditVerdict.UNSUPPORTED,
                justification="Nenhum trecho de referencia foi recuperado para avaliar este chunk.",
                cited_excerpts=[],
                judge_model=self.llm_client.model,
            )

        prompt = build_judge_prompt(chunk.text, curated.assembled_context)
        output, usage = self.llm_client.complete_json(
            system_message=JUDGE_SYSTEM_MESSAGE,
            user_prompt=prompt,
            schema=JudgeOutput,
        )

        return AuditResult(
            answer_chunk_id=chunk.id,
            verdict=AuditVerdict(output.verdict),
            justification=output.justification,
            cited_excerpts=output.cited_excerpts,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_usd=usage.cost_usd,
            latency_ms=usage.latency_ms,
            judge_model=self.llm_client.model,
        )
