from __future__ import annotations

from pathlib import Path

import typer

from .common import make_run_id
from .common.errors import ConfigurationError
from .config import Settings, get_settings
from .logging_config import (
    attach_run_log,
    detach_run_log,
    get_logger,
    record_exception,
    record_terminal_output,
    setup_logging,
)
from .ingestion.registry import ReferenceRegistry
from .ingestion.service import reclassify_documents
from .pipeline import (
    RunContext,
    build_pipeline,
    invalidate_after_ingestion,
    load_audit_results,
    load_run_context,
    save_run_meta,
)
from .reporting import aggregate_tool_stats, render_tool_comparison_markdown

app = typer.Typer(
    help=(
        "Framework para auditar se respostas de ferramentas de Deep "
        "Research sao suportadas pelas referencias que citam."
    )
)


def _echo(message: str) -> None:
    """`typer.echo` que tambem fica registrado no `log.txt` da run."""
    typer.echo(message)
    record_terminal_output(message)


def _run_pipeline(settings: Settings, ctx: RunContext) -> None:
    logger = get_logger(__name__)
    try:
        build_pipeline(settings).run(ctx)
    except Exception:
        record_exception(f"Run {ctx.run_id} interrompida por erro")
        raise
    logger.info("Run %s concluida", ctx.run_id)
    _echo(f"Relatorio gerado em {ctx.run_dir / 'report.md'}")


@app.command()
def run(
    answer_file: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Arquivo de resposta (md/pdf/docx)"
    ),
    tool_name: str = typer.Option(
        "unknown", "--tool", help="Nome da ferramenta de Deep Research (ChatGPT/Gemini/Perplexity/...)"
    ),
) -> None:
    """Executa o pipeline completo (extraction -> ingestion -> indexing ->
    judging -> reporting) para um unico arquivo de resposta."""
    settings = get_settings()
    setup_logging(settings)
    logger = get_logger(__name__)

    run_id = make_run_id(answer_file)
    ctx = RunContext(run_id=run_id, settings=settings, answer_path=answer_file, tool_name=tool_name)
    attach_run_log(ctx.run_dir)
    try:
        logger.info("Iniciando run %s para %s (ferramenta=%s)", run_id, answer_file, tool_name)
        save_run_meta(ctx)
        _run_pipeline(settings, ctx)
    finally:
        detach_run_log()


@app.command()
def resume(run_id: str = typer.Argument(..., help="Identificador de uma execucao anterior")) -> None:
    """Retoma uma execucao anterior a partir do ultimo estagio concluido,
    sem reprocessar estagios ja persistidos em `data/runs/<run_id>/`."""
    settings = get_settings()
    setup_logging(settings)
    logger = get_logger(__name__)

    try:
        ctx = load_run_context(run_id, settings)
    except ConfigurationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    attach_run_log(ctx.run_dir)
    try:
        logger.info("Retomando run %s", run_id)
        _run_pipeline(settings, ctx)
    finally:
        detach_run_log()


@app.command()
def reclassify(
    run_id: str = typer.Argument(..., help="Identificador de uma execucao anterior"),
    dry_run: bool = typer.Option(False, "--dry-run", help="So lista as barreiras, sem alterar nada"),
) -> None:
    """Procura barreiras de acesso (anti-bot, erro, login, intersticio) nos
    documentos ja baixados de uma execucao e tira essas fontes da analise,
    sem baixar nada de novo. Os artefatos de indexacao e julgamento ficam
    invalidos e sao movidos para `invalidado-<data>/`; depois rode
    `audit resume`."""
    settings = get_settings()
    setup_logging(settings)
    try:
        ctx = load_run_context(run_id, settings)
    except ConfigurationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    attach_run_log(ctx.run_dir)
    try:
        record_terminal_output(f"audit reclassify {run_id}{' --dry-run' if dry_run else ''}")
        blocked = reclassify_documents(ReferenceRegistry(ctx.run_dir), dry_run=dry_run)
        _echo(f"{run_id}: {len(blocked)} referencia(s) com barreira de acesso")
        for ref in blocked:
            _echo(f"  {ref.raw_url} — {ref.barrier_kind} — {ref.error_message}")
        if dry_run or not blocked:
            return
        backup = invalidate_after_ingestion(ctx.run_dir)
        if backup is not None:
            _echo(f"Artefatos de indexacao/julgamento movidos para {backup}")
            _echo(f"Rode: audit resume {run_id}")
    finally:
        detach_run_log()


@app.command()
def report(run_id: str = typer.Argument(..., help="Identificador de uma execucao anterior")) -> None:
    """Reimprime o relatorio ja gerado de uma execucao concluida."""
    settings = get_settings()
    setup_logging(settings)

    report_path = settings.run_dir(run_id) / "report.md"
    if not report_path.exists():
        typer.echo(
            f"Relatorio ainda nao existe para a run {run_id!r}. Rode `audit resume {run_id}` primeiro.", err=True
        )
        raise typer.Exit(code=1)
    typer.echo(report_path.read_text(encoding="utf-8"))


@app.command()
def compare(
    run_ids: list[str] = typer.Argument(..., help="Dois ou mais run_id para comparar lado a lado"),
) -> None:
    """Compara estatisticas de multiplas execucoes ja julgadas (ex: a
    mesma resposta auditada por ferramentas ou LLMs juizes diferentes)."""
    settings = get_settings()
    setup_logging(settings)

    stats = []
    for run_id in run_ids:
        try:
            ctx = load_run_context(run_id, settings)
        except ConfigurationError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        results = load_audit_results(ctx.run_dir / "audit_results.jsonl")
        stats.append(aggregate_tool_stats(ctx.tool_name, results))

    typer.echo(render_tool_comparison_markdown(stats))


if __name__ == "__main__":
    app()
