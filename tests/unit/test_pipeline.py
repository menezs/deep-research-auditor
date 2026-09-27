from __future__ import annotations

import json
from pathlib import Path

import pytest

from auditframework.common.errors import ConfigurationError
from auditframework.config import Settings
from auditframework.pipeline import (
    ExtractionStage,
    IndexingStage,
    IngestionStage,
    JudgingStage,
    Pipeline,
    ReportingStage,
    RunContext,
    _load_answer_chunks,
    _strip_reference_section,
    build_pipeline,
    load_run_context,
    save_run_meta,
)


def _settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", model_cache_dir=tmp_path / "model_cache")


class _RecordingStage:
    def __init__(self, name: str):
        self.name = name
        self.calls = 0

    def run(self, ctx: RunContext) -> None:
        self.calls += 1


def test_save_run_meta_creates_run_dir_if_missing(tmp_path: Path):
    settings = _settings(tmp_path)
    ctx = RunContext(run_id="run-1", settings=settings, answer_path=tmp_path / "answer.md", tool_name="ChatGPT")

    assert not ctx.run_dir.exists()
    save_run_meta(ctx)

    assert (ctx.run_dir / "run_meta.json").exists()


def test_load_run_context_round_trips_meta(tmp_path: Path):
    settings = _settings(tmp_path)
    answer_path = tmp_path / "answer.md"
    original = RunContext(run_id="run-1", settings=settings, answer_path=answer_path, tool_name="Gemini")
    save_run_meta(original)

    loaded = load_run_context("run-1", settings)

    assert loaded.answer_path == answer_path
    assert loaded.tool_name == "Gemini"
    assert loaded.started_at == original.started_at


def test_load_run_context_raises_for_unknown_run_id(tmp_path: Path):
    settings = _settings(tmp_path)
    with pytest.raises(ConfigurationError):
        load_run_context("nunca-existiu", settings)


def test_pipeline_skips_stages_already_completed_on_disk(tmp_path: Path):
    settings = _settings(tmp_path)
    ctx = RunContext(run_id="run-1", settings=settings, answer_path=tmp_path / "answer.md", tool_name="ChatGPT")
    ctx.run_dir.mkdir(parents=True)
    (ctx.run_dir / "state.json").write_text('{"stages_completed": ["stage-a"]}', encoding="utf-8")

    stage_a, stage_b = _RecordingStage("stage-a"), _RecordingStage("stage-b")
    pipeline = Pipeline(settings)
    pipeline.add_stage(stage_a)
    pipeline.add_stage(stage_b)

    pipeline.run(ctx)

    assert stage_a.calls == 0
    assert stage_b.calls == 1
    assert ctx.stages_completed == ["stage-a", "stage-b"]


def test_pipeline_persists_stage_completion_incrementally(tmp_path: Path):
    settings = _settings(tmp_path)
    ctx = RunContext(run_id="run-1", settings=settings, answer_path=tmp_path / "answer.md", tool_name="ChatGPT")
    pipeline = Pipeline(settings)
    pipeline.add_stage(_RecordingStage("stage-a"))
    pipeline.add_stage(_RecordingStage("stage-b"))

    pipeline.run(ctx)

    from auditframework.pipeline import _load_stage_state

    assert _load_stage_state(ctx.run_dir) == ["stage-a", "stage-b"]


class _FakeEmbedder:
    def __init__(self, model_name: str, cache_folder):
        self.model_name = model_name
        self.cache_folder = cache_folder
        self.dimension = 4


class _FakeReranker:
    def __init__(self, model_name: str, cache_folder):
        self.model_name = model_name
        self.cache_folder = cache_folder


class _FakeLLMClient:
    model = "fake-model"


def test_build_pipeline_wires_all_five_stages_in_order(tmp_path: Path, monkeypatch):
    import auditframework.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "BGEEmbedder", _FakeEmbedder)
    monkeypatch.setattr(pipeline_module, "Reranker", _FakeReranker)
    monkeypatch.setattr(pipeline_module, "create_llm_client", lambda settings: _FakeLLMClient())

    settings = _settings(tmp_path)
    pipeline = build_pipeline(settings)

    stage_types = [type(stage) for stage in pipeline._stages]
    assert stage_types == [ExtractionStage, IngestionStage, IndexingStage, JudgingStage, ReportingStage]


def test_build_pipeline_passes_configured_model_names_to_adapters(tmp_path: Path, monkeypatch):
    import auditframework.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "BGEEmbedder", _FakeEmbedder)
    monkeypatch.setattr(pipeline_module, "Reranker", _FakeReranker)
    monkeypatch.setattr(pipeline_module, "create_llm_client", lambda settings: _FakeLLMClient())

    settings = _settings(tmp_path)
    pipeline = build_pipeline(settings)

    indexing_stage = pipeline._stages[2]
    assert indexing_stage.embedder.model_name == settings.embedding_model
    judging_stage = pipeline._stages[3]
    assert judging_stage.embedder.model_name == settings.embedding_model
    assert judging_stage.reranker.model_name == settings.reranker_model
    assert judging_stage.top_k == settings.retrieval_top_k
    assert judging_stage.rerank_top_k == settings.rerank_top_k


def test_build_pipeline_wires_five_stages_ending_in_judging_and_reporting(tmp_path: Path, monkeypatch):
    import auditframework.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "BGEEmbedder", _FakeEmbedder)
    monkeypatch.setattr(pipeline_module, "Reranker", _FakeReranker)
    monkeypatch.setattr(pipeline_module, "create_llm_client", lambda settings: _FakeLLMClient())

    settings = _settings(tmp_path)
    stages = build_pipeline(settings)._stages
    assert [s.name for s in stages] == ["extraction", "ingestion", "indexing", "judging", "reporting"]


class TestStripReferenceSection:
    def test_removes_heading_and_everything_after_it(self):
        text = "Corpo da resposta [1].\n\n## Referências\n\n[1] Titulo\nhttps://example.com"
        assert _strip_reference_section(text) == "Corpo da resposta [1]."

    def test_is_case_insensitive_and_accepts_variants(self):
        text = "Corpo.\n\n## References\n\n[1] Title\nhttps://example.com"
        assert _strip_reference_section(text) == "Corpo."

    def test_text_without_a_reference_heading_is_returned_unchanged(self):
        text = "Um paragrafo qualquer sem lista de fontes."
        assert _strip_reference_section(text) == text


class TestStripAsterismList:
    def test_cuts_at_the_perplexity_asterism_when_there_is_no_heading(self):
        text = "Corpo [1].\n\n⁂\n\n1. <u>https://a.com</u>\n2. <u>https://b.com</u>"
        assert _strip_reference_section(text) == "Corpo [1]."

    def test_cuts_at_the_heading_even_when_a_later_asterism_exists(self):
        """Corta no cabeçalho, não na última âncora: quando as duas existem,
        o Perplexity emite `References` (a numeração que o corpo cita) e
        depois `⁂` (todas as fontes consultadas), e a lista do meio ficava
        no corpo — 59 entradas de lista viraram afirmações julgadas no
        corpus. O preço é este caso: um cabeçalho de fontes *com* marcação
        markdown seguido de prosa perde essa prosa. Não ocorre em nenhum dos
        27 arquivos reais; o caso que ocorre (título de seção do corpo como
        `Fontes oficiais`) chega sem marcação e é barrado pela densidade."""
        text = "## Referências\n\nCorpo [1].\n\n⁂\n\n1. https://a.com"
        assert _strip_reference_section(text) == ""


class TestSmallModelHeuristic:
    def test_flags_small_models(self):
        from auditframework.pipeline import _looks_like_small_model

        for m in (
            "google/gemma-4-e4b", "qwen2.5-3b", "phi-3-mini", "llama-3.2-1b", "mistral-7b",
            "deepseek-v4-flash", "gemini-2.5-flash-lite", "qwen3-4b-fast", "deepseek-r1-distill-llama-8b",
        ):
            assert _looks_like_small_model(m) is True, m

    def test_does_not_flag_large_models(self):
        from auditframework.pipeline import _looks_like_small_model

        for m in ("openai/gpt-oss-20b", "llama-3.1-8b", "claude-sonnet-5", "qwen2.5-32b"):
            assert _looks_like_small_model(m) is False, m


class TestLoadingChunksFromOlderRuns:
    """Runs gravadas antes de os chunks sem citação deixarem de existir
    ainda têm esses registros em `answer_chunks.json`."""

    def _write(self, run_dir: Path, records: list[dict]) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "answer_chunks.json").write_text(json.dumps(records), encoding="utf-8")

    def test_uncited_chunks_from_an_old_run_are_dropped(self, tmp_path: Path):
        self._write(tmp_path, [
            {"id": "a1-0", "answer_id": "a1", "position": 0, "text": "Prosa sem fonte.",
             "cited_reference_ids": [], "is_uncited_claim": True},
            {"id": "a1-1", "answer_id": "a1", "position": 1, "text": "Afirmação citada.",
             "cited_reference_ids": ["r1"], "is_uncited_claim": False},
        ])
        chunks = _load_answer_chunks(tmp_path)
        assert [c.id for c in chunks] == ["a1-1"]

    def test_chunk_whose_marker_has_no_entry_is_kept(self, tmp_path: Path):
        """Tem citação, só não tem entrada na lista — é auditável como
        SKIPPED com motivo próprio, não é um trecho sem citação."""
        self._write(tmp_path, [
            {"id": "a1-0", "answer_id": "a1", "position": 0, "text": "Cita [9], que não está na lista.",
             "cited_reference_ids": [], "is_uncited_claim": False},
        ])
        assert [c.id for c in _load_answer_chunks(tmp_path)] == ["a1-0"]
