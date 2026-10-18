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


def _fake_build_pipeline(settings: Settings) -> Pipeline:
    pipeline = Pipeline(settings)
    pipeline.add_stage(_FakeStage())
    return pipeline


class _ExplodingStage:
    name = "explode"

    def run(self, ctx: RunContext) -> None:
        raise RuntimeError("falha simulada no estagio")


def _exploding_build_pipeline(settings: Settings) -> Pipeline:
    pipeline = Pipeline(settings)
    pipeline.add_stage(_ExplodingStage())
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


def test_run_command_rejects_the_removed_full_corpus_flag(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)
    monkeypatch.setattr(cli_module, "build_pipeline", _fake_build_pipeline)

    answer_file = tmp_path / "resposta.md"
    answer_file.write_text("Uma resposta qualquer.", encoding="utf-8")

    result = runner.invoke(cli_module.app, ["run", str(answer_file), "--full-corpus"])

    assert result.exit_code != 0
    assert "full-corpus" in result.output or "No such option" in result.output


def test_run_command_persists_run_meta_without_full_corpus_mode(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)
    monkeypatch.setattr(cli_module, "build_pipeline", _fake_build_pipeline)

    answer_file = tmp_path / "resposta.md"
    answer_file.write_text("Uma resposta qualquer.", encoding="utf-8")

    result = runner.invoke(cli_module.app, ["run", str(answer_file)])
    assert result.exit_code == 0, result.output

    import json

    run_id = cli_module.make_run_id(answer_file)
    meta = json.loads((settings.run_dir(run_id) / "run_meta.json").read_text(encoding="utf-8"))
    assert "full_corpus_mode" not in meta
    assert set(meta) == {"answer_path", "tool_name", "started_at"}


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


def test_cli_reclassify_sem_barreira_nao_invalida(tmp_path, monkeypatch):
    from auditframework.ingestion.registry import ReferenceRegistry
    from auditframework.models import Document, Reference, ReferenceStatus

    settings = _settings(tmp_path)
    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)
    answer_file = tmp_path / "resposta.md"
    answer_file.write_text("Uma resposta qualquer.", encoding="utf-8")
    ctx = RunContext(run_id="run-1", settings=settings, answer_path=answer_file, tool_name="ChatGPT")
    save_run_meta(ctx)
    registry = ReferenceRegistry(ctx.run_dir)
    text = "Texto real e longo o bastante sobre o assunto citado. " * 5
    registry.save_references([Reference(
        id="r1", raw_url="https://example.com", normalized_url="https://example.com",
        status=ReferenceStatus.DOWNLOADED, source_answer_id="a", tool_name="ChatGPT",
    )])
    registry.save_document(
        Document(reference_id="r1", markdown_path=Path("documents/r1.md"), content_hash="x",
                 fetch_method="requests", word_count=len(text.split())),
        text,
    )
    (ctx.run_dir / "index").mkdir()

    result = runner.invoke(cli_module.app, ["reclassify", "run-1"])

    assert result.exit_code == 0, result.output
    assert "0 referencia(s)" in result.output
    assert (ctx.run_dir / "index").exists()


def test_run_command_writes_log_txt_in_run_dir(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)
    monkeypatch.setattr(cli_module, "build_pipeline", _fake_build_pipeline)
    answer_file = tmp_path / "resposta.md"
    answer_file.write_text("Uma resposta qualquer.", encoding="utf-8")

    result = runner.invoke(cli_module.app, ["run", str(answer_file), "--tool", "Gemini"])

    assert result.exit_code == 0, result.output
    run_id = cli_module.make_run_id(answer_file)
    log = (settings.run_dir(run_id) / "log.txt").read_text(encoding="utf-8")
    assert f"Iniciando run {run_id}" in log
    assert "Relatorio gerado em" in log


def test_resume_command_appends_to_existing_log(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)
    monkeypatch.setattr(cli_module, "build_pipeline", _fake_build_pipeline)
    answer_file = tmp_path / "resposta.md"
    answer_file.write_text("Uma resposta qualquer.", encoding="utf-8")
    ctx = RunContext(run_id="run-1", settings=settings, answer_path=answer_file, tool_name="ChatGPT")
    save_run_meta(ctx)
    (ctx.run_dir / "log.txt").write_text("log da execucao anterior\n", encoding="utf-8")

    result = runner.invoke(cli_module.app, ["resume", "run-1"])

    assert result.exit_code == 0, result.output
    log = (ctx.run_dir / "log.txt").read_text(encoding="utf-8")
    assert log.startswith("log da execucao anterior")
    assert "Retomando run run-1" in log


def test_resume_unknown_run_id_does_not_create_run_dir(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)

    runner.invoke(cli_module.app, ["resume", "nunca-existiu"])

    assert not settings.run_dir("nunca-existiu").exists()


def test_run_failure_traceback_is_recorded_in_log(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)
    monkeypatch.setattr(cli_module, "build_pipeline", _exploding_build_pipeline)
    answer_file = tmp_path / "resposta.md"
    answer_file.write_text("Uma resposta qualquer.", encoding="utf-8")

    result = runner.invoke(cli_module.app, ["run", str(answer_file)])

    assert result.exit_code != 0
    run_id = cli_module.make_run_id(answer_file)
    log = (settings.run_dir(run_id) / "log.txt").read_text(encoding="utf-8")
    assert "Traceback" in log
    assert "falha simulada no estagio" in log


def test_cli_reclassify_records_its_output_in_log(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)
    answer_file = tmp_path / "resposta.md"
    answer_file.write_text("Uma resposta qualquer.", encoding="utf-8")
    ctx = RunContext(run_id="run-1", settings=settings, answer_path=answer_file, tool_name="ChatGPT")
    save_run_meta(ctx)

    result = runner.invoke(cli_module.app, ["reclassify", "run-1", "--dry-run"])

    assert result.exit_code == 0, result.output
    log = (ctx.run_dir / "log.txt").read_text(encoding="utf-8")
    assert "run-1: 0 referencia(s) com barreira de acesso" in log
