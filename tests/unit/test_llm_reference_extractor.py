from auditframework.common.errors import LLMParseError
from auditframework.extraction.reference_extractor import (
    LLMReferenceExtractor,
    _ExtractedReference,
    _ExtractedReferenceList,
)


class FakeLLMClient:
    def __init__(self, response=None, error: Exception | None = None):
        self.model = "fake-model"
        self._response = response
        self._error = error

    def complete_json(self, *, system_message, user_prompt, schema):
        if self._error is not None:
            raise self._error
        return self._response, None


def test_extracts_references_from_llm_output():
    output = _ExtractedReferenceList(
        references=[
            _ExtractedReference(citation_markers=["[1]"], url="https://example.com/a", title="Artigo A"),
            _ExtractedReference(citation_markers=["[2]", "[3]"], url="https://example.com/b", title="Artigo B"),
        ]
    )
    extractor = LLMReferenceExtractor(FakeLLMClient(response=output))

    refs = extractor.extract("texto qualquer", source_answer_id="a1", tool_name="Gemini")

    assert len(refs) == 2
    b = next(r for r in refs if "example.com/b" in r.raw_url)
    assert b.citation_markers == ["[2]", "[3]"]
    assert b.title == "Artigo B"
    assert all(r.source_answer_id == "a1" and r.tool_name == "Gemini" for r in refs)


def test_duplicate_url_merges_citation_markers():
    output = _ExtractedReferenceList(
        references=[
            _ExtractedReference(citation_markers=["[1]"], url="https://example.com/a"),
            _ExtractedReference(citation_markers=["[5]"], url="https://example.com/a/"),
        ]
    )
    extractor = LLMReferenceExtractor(FakeLLMClient(response=output))

    refs = extractor.extract("texto", source_answer_id="a1", tool_name="ChatGPT")

    assert len(refs) == 1
    assert refs[0].citation_markers == ["[1]", "[5]"]


def test_malformed_llm_output_returns_empty_list_not_an_exception():
    extractor = LLMReferenceExtractor(FakeLLMClient(error=LLMParseError("nao e JSON")))

    refs = extractor.extract("texto", source_answer_id="a1", tool_name="ChatGPT")

    assert refs == []
