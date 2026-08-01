import pytest

from auditframework.common.errors import LLMParseError
from auditframework.common.llm_client import LLMUsage
from auditframework.judging.judge import Verifier
from auditframework.judging.prompts import JudgeOutput
from auditframework.models import AnswerChunk, AuditVerdict, CuratedDocument, RetrievedPassage


class FakeLLMClient:
    def __init__(self, response=None, error: Exception | None = None, usage: LLMUsage | None = None):
        self.model = "fake-model"
        self._response = response
        self._error = error
        self._usage = usage or LLMUsage(prompt_tokens=100, completion_tokens=20, latency_ms=42, cost_usd=0.0015)
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, *, system_message, user_prompt, schema):
        self.calls.append((system_message, user_prompt))
        if self._error is not None:
            raise self._error
        return self._response, self._usage


def _chunk(text: str = "O Brasil aprovou o Marco Civil da Internet em 2014.") -> AnswerChunk:
    return AnswerChunk(id="c1", answer_id="a1", position=0, text=text, cited_reference_ids=["refA"])


def _curated(with_passage: bool = True) -> CuratedDocument:
    passages = (
        [RetrievedPassage(reference_chunk_id="rc1", reference_id="refA", score=0.9, text="O Marco Civil...")]
        if with_passage
        else []
    )
    return CuratedDocument(
        answer_chunk_id="c1",
        passages=passages,
        assembled_context="[referencia=refA | score=0.900]\nO Marco Civil...",
    )


def test_no_passages_returns_unsupported_without_calling_llm():
    client = FakeLLMClient()
    verifier = Verifier(client)

    result = verifier.verify(_chunk(), _curated(with_passage=False))

    assert result.verdict == AuditVerdict.UNSUPPORTED
    assert client.calls == []
    assert result.prompt_tokens == 0
    assert result.cost_usd == 0.0


def test_supported_verdict_and_usage_are_propagated():
    output = JudgeOutput(
        verdict="supported",
        justification="O contexto confirma a afirmacao.",
        cited_excerpts=["O Marco Civil..."],
    )
    client = FakeLLMClient(response=output)
    verifier = Verifier(client)

    result = verifier.verify(_chunk(), _curated())

    assert result.verdict == AuditVerdict.SUPPORTED
    assert result.cited_excerpts == ["O Marco Civil..."]
    assert result.prompt_tokens == 100
    assert result.completion_tokens == 20
    assert result.latency_ms == 42
    assert result.cost_usd == 0.0015
    assert result.judge_model == "fake-model"
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "verdict_str,expected",
    [
        ("supported", AuditVerdict.SUPPORTED),
        ("unsupported", AuditVerdict.UNSUPPORTED),
        ("contradicted", AuditVerdict.CONTRADICTED),
    ],
)
def test_all_verdict_strings_map_to_correct_enum(verdict_str, expected):
    output = JudgeOutput(verdict=verdict_str, justification="...")
    client = FakeLLMClient(response=output)
    verifier = Verifier(client)

    result = verifier.verify(_chunk(), _curated())

    assert result.verdict == expected


def test_unparseable_llm_output_propagates_instead_of_becoming_a_verdict():
    """Regressao: o bug do audit_with_llm coagia qualquer rotulo
    desconhecido/saida malformada para CONTRADICTED silenciosamente. Com
    apenas 3 categorias de conteudo, uma falha de parsing nao vira um
    veredito — a excecao propaga e o chunk fica pendente para
    `audit resume`."""
    client = FakeLLMClient(error=LLMParseError("saida nao e JSON valido: 'blah'"))
    verifier = Verifier(client)

    with pytest.raises(LLMParseError, match="blah"):
        verifier.verify(_chunk(), _curated())
