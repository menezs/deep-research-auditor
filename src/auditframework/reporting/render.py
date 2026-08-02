from __future__ import annotations

from ..models import AnswerChunk, AuditResult, AuditVerdict, Reference, Report, ToolStats

_VERDICT_LABELS: dict[AuditVerdict, str] = {
    AuditVerdict.SUPPORTED: "SUPPORTED",
    AuditVerdict.UNSUPPORTED: "UNSUPPORTED",
    AuditVerdict.CONTRADICTED: "CONTRADICTED",
}

_MAX_EXAMPLES_PER_VERDICT = 3
_EXCERPT_LEN = 220


def _excerpt(text: str, length: int = _EXCERPT_LEN) -> str:
    text = " ".join(text.split())
    return text if len(text) <= length else text[: length - 1].rstrip() + "…"


class ReportRenderer:
    """Builder do texto final de um `Report`.

    O `Report` em si ja carrega toda a agregacao numerica; esta classe so
    monta a representacao (Markdown/JSON), na estrutura observada nos
    relatorios escritos a mao no audit_with_llm original
    (`relatorio_execucao_*.md`): metadados, distribuicao de veredito,
    custo, ranking de referencias, referencias mortas/inacessiveis e
    exemplos representativos por veredito.

    `chunks`/`references`/`results` sao opcionais e usados apenas para a
    secao de exemplos — sem eles o relatorio ainda e completo, so sem essa
    secao (util para gerar um relatorio so a partir de um `Report` ja
    persistido, sem precisar recarregar todos os resultados brutos)."""

    def __init__(
        self,
        report: Report,
        *,
        chunks: list[AnswerChunk] | None = None,
        references: list[Reference] | None = None,
        results: list[AuditResult] | None = None,
    ) -> None:
        self.report = report
        self._chunk_by_id = {c.id: c for c in (chunks or [])}
        self._reference_by_id = {r.id: r for r in (references or [])}
        self._results = results or []

    def to_markdown(self) -> str:
        sections = [
            self._render_header(),
            self._render_metadata(),
            self._render_distribution(),
            self._render_cost(),
            self._render_reference_ranking(),
            self._render_dead_and_inaccessible(),
        ]
        if self._results:
            sections.append(self._render_examples())
        sections.append(self._render_skipped_chunks())
        return "\n\n---\n\n".join(section for section in sections if section)

    def to_json(self) -> str:
        return self.report.model_dump_json(indent=2)

    def _render_header(self) -> str:
        report = self.report
        return f"# Relatório de Auditoria — {report.tool_name} ({report.answer_id})"

    def _render_metadata(self) -> str:
        report = self.report
        rows = [
            ("Run ID", report.run_id),
            ("Ferramenta", report.tool_name),
            ("Gerado em", report.generated_at.isoformat()),
            ("Total de chunks", str(report.total_chunks)),
            ("Tempo de processamento", f"{report.processing_time_seconds:.1f}s"),
        ]
        table = "\n".join(f"| **{label}** | {value} |" for label, value in rows)
        return "## 1. Metadados da Execução\n\n| Campo | Valor |\n|---|---|\n" + table

    def _render_distribution(self) -> str:
        report = self.report
        pct_skipped = (report.count_skipped / report.total_chunks * 100.0) if report.total_chunks else 0.0
        rows = [
            ("SUPPORTED", report.count_supported, report.pct_supported),
            ("UNSUPPORTED", report.count_unsupported, report.pct_unsupported),
            ("CONTRADICTED", report.count_contradicted, report.pct_contradicted),
        ]
        table = "\n".join(f"| **{label}** | {count} | {pct:.1f}% |" for label, count, pct in rows)
        total_chunks = report.count_supported + report.count_unsupported + report.count_contradicted
        total_pct = report.pct_supported + report.pct_unsupported + report.pct_contradicted
        if report.count_skipped:
            table += f"\n| **SKIPPED** | {report.count_skipped} | {pct_skipped:.1f}% |"
            total_chunks += report.count_skipped
            total_pct += pct_skipped
        table += f"\n| **TOTAL** | {total_chunks} | {total_pct:.1f}% |"
        return "## 2. Distribuição de Vereditos\n\n| Veredito | Chunks | Percentual |\n|---|---|---|\n" + table

    def _render_cost(self) -> str:
        report = self.report
        total_requests = report.count_supported + report.count_unsupported + report.count_contradicted
        avg_tokens = report.total_tokens / total_requests if total_requests else 0.0
        avg_cost = report.total_cost_usd / total_requests if total_requests else 0.0
        rows = [
            ("Custo total estimado", f"US$ {report.total_cost_usd:.4f}"),
            ("Total de tokens", str(report.total_tokens)),
            ("Média de tokens por requisição", f"{avg_tokens:.1f}"),
            ("Média de custo por requisição", f"US$ {avg_cost:.4f}"),
        ]
        table = "\n".join(f"| **{label}** | {value} |" for label, value in rows)
        return "## 3. Custo e Uso de Tokens\n\n| Métrica | Valor |\n|---|---|\n" + table

    def _render_reference_ranking(self) -> str:
        stats = self.report.reference_stats
        if not stats:
            return "## 4. Análise por Referência\n\nNenhuma referência citada nos chunks avaliados."
        header = (
            "## 4. Análise por Referência\n\n"
            "| Referência | Status | Citações | SUPPORTED | UNSUPPORTED | CONTRADICTED |\n"
            "|---|---|---|---|---|---|\n"
        )
        rows = "\n".join(
            f"| [{self._reference_label(s.reference_id, s.url)}]({s.url}) | {s.status.value} | {s.times_cited} | "
            f"{s.supported_count} | {s.unsupported_count} | {s.contradicted_count} |"
            for s in stats
        )
        return header + rows

    def _reference_label(self, reference_id: str, fallback_url: str) -> str:
        reference = self._reference_by_id.get(reference_id)
        if reference is not None and reference.citation_markers:
            return " ".join(reference.citation_markers)
        return fallback_url

    def _render_dead_and_inaccessible(self) -> str:
        report = self.report
        if not report.dead_references and not report.inaccessible_references:
            return ""
        lines = ["## 5. Referências Mortas e Inacessíveis"]
        if report.dead_references:
            lines.append("\n### Mortas (HTTP 404)\n")
            lines.append("\n".join(f"- {r.raw_url} — {r.error_message or 'sem detalhes'}" for r in report.dead_references))
        if report.inaccessible_references:
            lines.append("\n### Inacessíveis (403/timeout/SSL)\n")
            lines.append(
                "\n".join(f"- {r.raw_url} — {r.error_message or 'sem detalhes'}" for r in report.inaccessible_references)
            )
        return "\n".join(lines)

    def _render_skipped_chunks(self) -> str:
        skipped = self.report.skipped_chunks
        if not skipped:
            return ""
        lines = ["## 7. Chunks Não Auditados"]
        lines.append("\n".join(f"- `{s.answer_chunk_id}` — {s.reason}" for s in skipped))
        return "\n\n".join(lines)

    def _render_examples(self) -> str:
        by_verdict: dict[AuditVerdict, list[AuditResult]] = {v: [] for v in AuditVerdict}
        for result in self._results:
            by_verdict[result.verdict].append(result)

        blocks = ["## 6. Exemplos Representativos por Veredito"]
        for verdict, label in _VERDICT_LABELS.items():
            examples = by_verdict.get(verdict, [])[:_MAX_EXAMPLES_PER_VERDICT]
            if not examples:
                continue
            blocks.append(f"\n### {label}\n")
            for result in examples:
                blocks.append(self._render_example(result))
        return "\n".join(blocks)

    def _render_example(self, result: AuditResult) -> str:
        chunk = self._chunk_by_id.get(result.answer_chunk_id)
        refs = []
        if chunk is not None:
            for ref_id in chunk.cited_reference_ids:
                reference = self._reference_by_id.get(ref_id)
                if reference is not None:
                    refs.append(", ".join(reference.citation_markers) or reference.raw_url)
        ref_label = " ".join(refs) if refs else "(sem referência resolvida)"
        chunk_excerpt = _excerpt(chunk.text) if chunk is not None else "(chunk indisponível)"
        return (
            f"- **Chunk `{result.answer_chunk_id}`** {ref_label}\n"
            f"  - Trecho: {chunk_excerpt}\n"
            f"  - Justificativa: {_excerpt(result.justification)}"
        )


def render_markdown(
    report: Report,
    *,
    chunks: list[AnswerChunk] | None = None,
    references: list[Reference] | None = None,
    results: list[AuditResult] | None = None,
) -> str:
    return ReportRenderer(report, chunks=chunks, references=references, results=results).to_markdown()


def render_json(report: Report) -> str:
    return ReportRenderer(report).to_json()


def render_tool_comparison_markdown(tool_stats: list[ToolStats]) -> str:
    """Tabela comparativa entre ferramentas (ChatGPT/Gemini/Perplexity),
    equivalente ao que hoje e escrito a mao em `relatorio_comparativo_*.md`."""
    if not tool_stats:
        return "## Comparação entre Ferramentas\n\nNenhum dado disponível."
    header = (
        "## Comparação entre Ferramentas\n\n"
        "| Ferramenta | Chunks | SUPPORTED | UNSUPPORTED | CONTRADICTED |\n"
        "|---|---|---|---|---|\n"
    )
    rows = "\n".join(
        f"| {s.tool_name} | {s.total_chunks} | {s.pct_supported:.1f}% | {s.pct_unsupported:.1f}% | "
        f"{s.pct_contradicted:.1f}% |"
        for s in tool_stats
    )
    return header + rows
