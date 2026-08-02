from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from auditframework import cli as cli_module
from auditframework.config import Settings
from auditframework.pipeline import Pipeline, RunContext, save_run_meta

runner = CliRunner()


class _FakeStage:
    name = "fake"

    def run(self, ctx: RunContext) -> None:
        ctx.run_dir.mkdir(parents=True, exist_ok=True)
        (ctx.run_dir / "report.md").write_text("# Relatorio fake", encoding="utf-8")


def _fake_build_pipeline(settings: Settings, *, full_corpus_mode: bool = False) -> Pipeline:
    pipeline = Pipeline(settings)
    pipeline.add_stage(_FakeStage())
    return pipeline


def _settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", model_cache_dir=tmp_path / "model_cache")


def test_run_command_executes_pipeline_and_reports_output_path(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)
    monkeypatch.setattr(cli_module, "build_pipeline", _fake_build_pipeline)

    answer_file = tmp_path / "resposta.md"
    answer_file.write_text("Uma resposta qualquer.", encoding="utf-8")

    result = runner.invoke(cli_module.app, ["run", str(answer_file), "--tool", "Gemini"])

    assert result.exit_code == 0, result.output
    assert "Relatorio gerado em" in result.output


def test_run_command_full_corpus_flag_is_persisted_and_passed_to_build_pipeline(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)

    captured: dict[str, bool] = {}

    def _spying_build_pipeline(settings: Settings, *, full_corpus_mode: bool = False) -> Pipeline:
        captured["full_corpus_mode"] = full_corpus_mode
        return _fake_build_pipeline(settings, full_corpus_mode=full_corpus_mode)

    monkeypatch.setattr(cli_module, "build_pipeline", _spying_build_pipeline)

    answer_file = tmp_path / "resposta.md"
    answer_file.write_text("Uma resposta qualquer.", encoding="utf-8")

    result = runner.invoke(cli_module.app, ["run", str(answer_file), "--full-corpus"])

    assert result.exit_code == 0, result.output
    assert captured["full_corpus_mode"] is True

    import json

    run_id = cli_module.make_run_id(answer_file)
    meta = json.loads((settings.run_dir(run_id) / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["full_corpus_mode"] is True


def test_resume_command_reuses_full_corpus_mode_from_run_meta(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)

    captured: dict[str, bool] = {}

    def _spying_build_pipeline(settings: Settings, *, full_corpus_mode: bool = False) -> Pipeline:
        captured["full_corpus_mode"] = full_corpus_mode
        return _fake_build_pipeline(settings, full_corpus_mode=full_corpus_mode)

    monkeypatch.setattr(cli_module, "build_pipeline", _spying_build_pipeline)

    answer_file = tmp_path / "resposta.md"
    answer_file.write_text("Uma resposta qualquer.", encoding="utf-8")
    ctx = RunContext(
        run_id="run-1", settings=settings, answer_path=answer_file, tool_name="ChatGPT", full_corpus_mode=True
    )
    save_run_meta(ctx)

    result = runner.invoke(cli_module.app, ["resume", "run-1"])

    assert result.exit_code == 0, result.output
    assert captured["full_corpus_mode"] is True


def test_run_command_fails_for_missing_answer_file(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)

    result = runner.invoke(cli_module.app, ["run", str(tmp_path / "nao-existe.md")])

    assert result.exit_code != 0


def test_resume_command_continues_an_existing_run(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)
    monkeypatch.setattr(cli_module, "build_pipeline", _fake_build_pipeline)

    answer_file = tmp_path / "resposta.md"
    answer_file.write_text("Uma resposta qualquer.", encoding="utf-8")
    ctx = RunContext(run_id="run-1", settings=settings, answer_path=answer_file, tool_name="ChatGPT")
    save_run_meta(ctx)

    result = runner.invoke(cli_module.app, ["resume", "run-1"])

    assert result.exit_code == 0, result.output
    assert (ctx.run_dir / "report.md").exists()


def test_resume_command_fails_clearly_for_unknown_run_id(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)

    result = runner.invoke(cli_module.app, ["resume", "nunca-existiu"])

    assert result.exit_code == 1
    assert "nao encontrada" in result.output


def test_report_command_prints_existing_report(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)
    run_dir = settings.run_dir("run-1")
    run_dir.mkdir(parents=True)
    (run_dir / "report.md").write_text("# Relatorio ja pronto", encoding="utf-8")

    result = runner.invoke(cli_module.app, ["report", "run-1"])

    assert result.exit_code == 0
    assert "Relatorio ja pronto" in result.output


def test_report_command_fails_clearly_when_report_does_not_exist_yet(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)

    result = runner.invoke(cli_module.app, ["report", "run-1"])

    assert result.exit_code == 1
    assert "audit resume run-1" in result.output


def _seed_run(settings: Settings, run_id: str, tool_name: str, verdicts: list[str]) -> None:
    from auditframework.models import AuditResult, AuditVerdict

    answer_path = settings.data_dir / f"{run_id}.md"
    answer_path.parent.mkdir(parents=True, exist_ok=True)
    answer_path.write_text("resposta", encoding="utf-8")
    ctx = RunContext(run_id=run_id, settings=settings, answer_path=answer_path, tool_name=tool_name)
    save_run_meta(ctx)
    results_path = ctx.run_dir / "audit_results.jsonl"
    with results_path.open("w", encoding="utf-8") as fh:
        for i, verdict in enumerate(verdicts):
            result = AuditResult(
                answer_chunk_id=f"{run_id}-{i}", verdict=AuditVerdict(verdict), justification="x", judge_model="m"
            )
            fh.write(result.model_dump_json() + "\n")


def test_compare_command_renders_a_table_for_each_run(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)
    _seed_run(settings, "run-a", "ChatGPT", ["supported", "unsupported"])
    _seed_run(settings, "run-b", "Gemini", ["supported", "supported"])

    result = runner.invoke(cli_module.app, ["compare", "run-a", "run-b"])

    assert result.exit_code == 0, result.output
    assert "ChatGPT" in result.output
    assert "Gemini" in result.output


def test_compare_command_fails_clearly_for_unknown_run_id(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)

    result = runner.invoke(cli_module.app, ["compare", "nunca-existiu"])

    assert result.exit_code == 1
    assert "nao encontrada" in result.output
