from __future__ import annotations

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


def test_build_pipeline_defaults_to_citation_scoped_retrieval(tmp_path: Path, monkeypatch):
    import auditframework.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "BGEEmbedder", _FakeEmbedder)
    monkeypatch.setattr(pipeline_module, "Reranker", _FakeReranker)
    monkeypatch.setattr(pipeline_module, "create_llm_client", lambda settings: _FakeLLMClient())

    settings = _settings(tmp_path)
    assert build_pipeline(settings)._stages[3].full_corpus_mode is False
    assert build_pipeline(settings, full_corpus_mode=True)._stages[3].full_corpus_mode is True


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
