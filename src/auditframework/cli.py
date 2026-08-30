from __future__ import annotations

from pathlib import Path

import typer

from .common import make_run_id
from .common.errors import ConfigurationError
from .config import get_settings
from .logging_config import get_logger, setup_logging
from .pipeline import Pipeline, RunContext, build_pipeline, load_audit_results, load_run_context, save_run_meta
from .reporting import aggregate_tool_stats, render_tool_comparison_markdown

app = typer.Typer(
    help=(
        "Framework para auditar se respostas de ferramentas de Deep "
        "Research sao suportadas pelas referencias que citam."
    )
)


def _run_pipeline(pipeline: Pipeline, ctx: RunContext) -> None:
    logger = get_logger(__name__)
    pipeline.run(ctx)
    logger.info("Run %s concluida", ctx.run_id)
    typer.echo(f"Relatorio gerado em {ctx.run_dir / 'report.md'}")


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
    logger.info("Iniciando run %s para %s (ferramenta=%s)", run_id, answer_file, tool_name)

    ctx = RunContext(run_id=run_id, settings=settings, answer_path=answer_file, tool_name=tool_name)
    save_run_meta(ctx)

    _run_pipeline(build_pipeline(settings), ctx)


@app.command()
def resume(run_id: str = typer.Argument(..., help="Identificador de uma execucao anterior")) -> None:
    """Retoma uma execucao anterior a partir do ultimo estagio concluido,
    sem reprocessar estagios ja persistidos em `data/runs/<run_id>/`."""
    settings = get_settings()
    setup_logging(settings)
    logger = get_logger(__name__)
    logger.info("Retomando run %s", run_id)

    try:
        ctx = load_run_context(run_id, settings)
    except ConfigurationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    _run_pipeline(build_pipeline(settings), ctx)


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
