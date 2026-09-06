from __future__ import annotations

from dataclasses import dataclass

from ..models import AnswerChunk, AuditResult, AuditVerdict, Reference, Report, ToolStats


@dataclass
class ReportSection:
    """Uma secao do relatorio, sem numero — o numero e atribuido por
    `ReportRenderer.to_markdown` na ordem em que as secoes NAO vazias
    aparecem, para que uma secao condicional ausente (ex: sem referencias
    mortas) nao deixe um buraco na numeracao (1, 2, 3, 4, 7...)."""

    title: str
    body: str

    def render(self, number: int) -> str:
        return f"## {number}. {self.title}\n\n{self.body}"

_VERDICT_LABELS: dict[AuditVerdict, str] = {
    AuditVerdict.SUPPORTED: "SUPPORTED",
    AuditVerdict.UNSUPPORTED: "UNSUPPORTED",
    AuditVerdict.CONTRADICTED: "CONTRADICTED",
}

_RELATION_LABELS: dict[str, str] = {
    "supports": "sustenta",
    "partial": "parcial",
    "absent": "ausente",
    "contradicts": "contradiz",
}

_MAX_EXAMPLES_PER_VERDICT = 3
_EXCERPT_LEN = 220


def _excerpt(text: str, length: int = _EXCERPT_LEN) -> str:
    text = " ".join(text.split())
    return text if len(text) <= length else text[: length - 1].rstrip() + "…"


def _compress_positions(positions: list[int]) -> str:
    """`[1, 2, 3, 5, 8, 9]` -> `#1–3, #5, #8–9`."""
    nums = sorted(set(positions))
    if not nums:
        return ""
    ranges: list[str] = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        ranges.append(f"#{start}" if start == prev else f"#{start}–{prev}")
        start = prev = n
    ranges.append(f"#{start}" if start == prev else f"#{start}–{prev}")
    return ", ".join(ranges)


def _format_duration(seconds: float) -> str:
    """Segundos -> `dias:horas:min:seg` (`D:HH:MM:SS`, dias sem zero à
    esquerda)."""
    total = int(round(max(seconds, 0.0)))
    days, rem = divmod(total, 86_400)
    hours, rem = divmod(rem, 3_600)
    minutes, secs = divmod(rem, 60)
    return f"{days}:{hours:02d}:{minutes:02d}:{secs:02d}"


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
        candidates = [
            self._render_metadata(),
            self._render_distribution(),
            self._render_verification(),
            self._render_citation_issues(),
            self._render_cost(),
            self._render_reference_ranking(),
            self._render_content_verification(),
            self._render_dead_and_inaccessible(),
            self._render_examples() if self._results else None,
            self._render_uncited_claims(),
            self._render_potentially_unsourced(),
            self._render_skipped_chunks(),
        ]
        parts = [self._render_header()]
        number = 0
        for section in candidates:
            if section is None:
                continue
            number += 1
            parts.append(section.render(number))
        return "\n\n---\n\n".join(parts)

    def to_json(self) -> str:
        return self.report.model_dump_json(indent=2)

    def _render_header(self) -> str:
        report = self.report
        header = f"# Relatório de Auditoria — {report.tool_name} ({report.answer_id})"
        if report.answer_path:
            header += f"\n\n**Arquivo auditado:** `{report.answer_path}`"
        return header

    def _render_metadata(self) -> ReportSection:
        report = self.report
        rows = [
            ("Run ID", report.run_id),
            ("Ferramenta", report.tool_name),
            ("Gerado em", report.generated_at.isoformat()),
            ("Total de chunks", str(report.total_chunks)),
            ("Tempo de processamento", _format_duration(report.processing_time_seconds)),
        ]
        judge = report.judge_config
        if judge is not None:
            rows.append(("Modelo do juiz LLM", judge.model))
            rows.append(("Provider do juiz LLM", judge.provider))
            rows.append(("Temperature", str(judge.temperature)))
            rows.append(("Max retries", str(judge.max_retries)))
            rows.append(("Retry delay", f"{judge.retry_delay}s"))
            if judge.base_url:
                rows.append(("Base URL", judge.base_url))
        table = "\n".join(f"| **{label}** | {value} |" for label, value in rows)
        body = "| Campo | Valor |\n|---|---|\n" + table
        source_block = self._render_source_info()
        if source_block:
            body += "\n\n" + source_block
        return ReportSection("Metadados da Execução", body)

    def _render_source_info(self) -> str:
        si = self.report.source_info
        if si is None:
            return ""
        na = "— (ausente)"
        kb = f"{si.size_bytes / 1024:.1f} KB"
        rows = [("Formato / tamanho", f"{si.format} · {kb}")]
        if si.page_count is not None:
            rows.append(("Páginas", str(si.page_count)))
        if si.pdf_creator or si.pdf_producer:
            rows.append(("Criador / Produtor", f"{si.pdf_creator or '—'} / {si.pdf_producer or '—'}"))
        rows.append(("Título / Autor", f"{si.pdf_title or na} / {si.pdf_author or na}"))
        if si.created or si.modified:
            rows.append(("Criação / Modificação", f"{si.created or '—'} / {si.modified or '—'}"))
        if si.encrypted:
            rows.append(("Criptografia", "sim"))
        rows.append(
            ("Citações no corpo", f"{si.citation_markers_total} marcadores, {si.citation_markers_distinct} distintos")
        )
        rows.append(
            ("Referências listadas", f"{si.references_listed} ({si.references_never_cited} nunca citadas)")
        )
        table = "**Arquivo de origem**\n\n| Campo | Valor |\n|---|---|\n" + "\n".join(
            f"| {label} | {value} |" for label, value in rows
        )
        notes = []
        if si.browser_print:
            notes.append(
                "PDF gerado por impressão de navegador (Skia/PDF + Chromium) — típico de captura de "
                "resposta de Deep Research; sem metadados de autoria confiáveis."
            )
        if si.citation_markers_distinct > si.references_listed:
            notes.append(
                f"{si.citation_markers_distinct} números de citação distintos no corpo para apenas "
                f"{si.references_listed} referências listadas — há marcadores `[N]` sem entrada correspondente."
            )
        for note in notes:
            table += f"\n\n> {note}"
        return table

    def _render_distribution(self) -> ReportSection:
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
        body = "| Veredito | Chunks | Percentual |\n|---|---|---|\n" + table
        if report.count_partially_supported:
            body += (
                f"\n\n> {report.count_partially_supported} de {report.count_supported} SUPPORTED têm "
                "aspectos da afirmação não cobertos pela evidência (\"parcialmente suportada\") — ver Exemplos."
            )
        if report.count_uncited_claims:
            body += (
                f"\n\n> {report.count_uncited_claims} de {report.count_claim_chunks} trechos com afirmação "
                "factual não têm nenhuma citação no documento — ver *Afirmações sem Citação*."
            )
        return ReportSection("Distribuição de Vereditos", body)

    def _render_verification(self) -> ReportSection | None:
        """Efeito da cascata de verificação aplicada aos vereditos
        UNSUPPORTED iniciais (etapas A → B → C)."""
        report = self.report
        if not report.verification_ran:
            return None
        initial_unsupported = (
            report.count_unsupported + report.count_reclassified_by_verification
        )
        lines = [
            f"`{initial_unsupported}` chunks tiveram veredito **UNSUPPORTED** no julgamento inicial.",
            "",
            f"- `{report.count_unsupported_confirmed}` **confirmados** — sobreviveram à varredura do "
            "documento citado inteiro (Etapa B).",
            f"- `{report.count_reclassified_by_verification}` **reclassificados** pela cascata para "
            "SUPPORTED/CONTRADICTED.",
            f"- `{report.mis_cited_reference_count}` **provável erro de citação** — não sustentados pela "
            "fonte citada, mas corroborados por outra referência baixada (Etapa C).",
        ]
        if report.verification_stage_counts:
            by_stage = ", ".join(
                f"{stage} = {count}" for stage, count in sorted(report.verification_stage_counts.items())
            )
            lines.append("")
            lines.append(f"Reclassificações por etapa: {by_stage}.")
        if report.citation_issues:
            lines.append("")
            lines.append(
                "Os prováveis erros de citação estão detalhados na seção seguinte."
            )
        return ReportSection("Verificação de UNSUPPORTED", "\n".join(lines))

    def _render_citation_issues(self) -> ReportSection | None:
        """Detalhe da Etapa C: chunks que a fonte citada não sustenta, mas
        cuja afirmação aparece (ou é contradita) em outra referência
        baixada."""
        issues = self.report.citation_issues
        if not issues:
            return None
        lines = [
            "Trechos com veredito **UNSUPPORTED** — a **fonte citada não os sustenta** — mas "
            "cuja afirmação **foi encontrada em outra referência já baixada** durante a "
            "checagem no corpus inteiro (Etapa C). A Etapa C **nunca** reclassifica para "
            "SUPPORTED (a auditoria é sobre a fonte citada), então o veredito continua "
            "UNSUPPORTED; o cenário mais provável, porém, é **citação trocada** — o fato é "
            "real, a fonte apontada é que está errada. `⚠` marca quando outra fonte "
            "**contradiz** o trecho.",
            "",
            "| Trecho | Cita | Corroborado por | Contradito por |",
            "|---|---|---|---|",
        ]
        for issue in issues:
            claim = f"`#{issue.position}` {_excerpt(issue.claim_excerpt, 120)}".replace("|", "\\|")
            contra = f"{issue.contradicting_markers} ⚠" if issue.contradicting_markers else "—"
            lines.append(
                f"| {claim} | {issue.cited_markers} | {issue.corroborating_markers or '—'} | {contra} |"
            )
        return ReportSection("Prováveis Erros de Citação", "\n".join(lines))

    def _render_cost(self) -> ReportSection:
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
        body = "| Métrica | Valor |\n|---|---|\n" + table
        if not report.cost_tracked:
            judge = report.judge_config
            provider = judge.provider if judge is not None else "desconhecido"
            model = judge.model if judge is not None else "desconhecido"
            body = (
                f"> ⚠️ **Custo não contabilizado** — o provider `{provider}` (modelo `{model}`) não tem "
                "tabela de preços neste framework. Os valores abaixo somam apenas chamadas com preço "
                "conhecido (Anthropic) ou custo local zero, e devem ser lidos como um piso, não como o "
                "custo real da execução.\n\n"
            ) + body
        return ReportSection("Custo e Uso de Tokens", body)

    def _render_reference_ranking(self) -> ReportSection:
        report = self.report
        stats = report.reference_stats
        if not stats:
            return ReportSection("Análise por Referência", "Nenhuma referência citada nos chunks avaliados.")
        total = len(stats)
        uncited = sum(1 for s in stats if s.times_cited == 0)
        pct_uncited = (uncited / total) * 100.0
        summary = f"Referências sem nenhuma citação: {uncited}/{total} ({pct_uncited:.1f}%)\n"
        if report.uncredited_reference_count:
            summary += (
                f"Referências citadas mas que nunca sustentaram (nem parcialmente) nenhuma afirmação: "
                f"{report.uncredited_reference_count}.\n"
            )
        summary += "\n"
        summary += (
            "> **Citada** = quantos trechos citam a fonte. As colunas seguintes abrem esse "
            "total e **somam Citada**: **Sustenta / Parcial** (a fonte sustenta a afirmação "
            "inteira / em parte); **Não sustenta †** (a fonte foi verificada e não trata a "
            "afirmação, a contradiz, ou o juiz não a detalhou); **Não auditada** (o trecho "
            "que a cita foi pulado, ou a fonte não pôde ser baixada — não houve verificação).\n\n"
        )
        header = (
            "| Referência | Status | Citada | Sustenta | Parcial | Não sustenta † | Não auditada |\n"
            "|---|---|---|---|---|---|---|\n"
        )
        rows_out: list[str] = []
        for s in stats:
            label = f"[{self._reference_label(s.reference_id, s.url)}]({s.url})"
            if s.times_cited == 0:
                rows_out.append(f"| {label} | {s.status.value} | 0 · nunca citada | — | — | — | — |")
                continue
            unsupportive = (
                len(s.absent_positions) + len(s.contradicts_positions) + len(s.unrated_positions)
            )
            rows_out.append(
                f"| {label} | {s.status.value} | {s.times_cited} | "
                f"{len(s.supports_positions)} | {len(s.partial_positions)} | {unsupportive} | "
                f"{len(s.not_audited_positions)} |"
            )
        return ReportSection("Análise por Referência", summary + header + "\n".join(rows_out))

    def _render_content_verification(self) -> ReportSection | None:
        cited = [s for s in self.report.reference_stats if s.times_cited > 0]
        rated = [
            s for s in cited
            if s.supports_positions or s.partial_positions or s.absent_positions or s.contradicts_positions
        ]
        if not rated:
            return None
        lines = [
            "Uma linha por **referência citada**: quais trechos do documento (`#N` = posição "
            "no texto) ela sustenta, sustenta só em parte, ou não sustenta. `⚡` = a fonte "
            "**contradiz** o trecho; `s/ aval.` = o juiz não avaliou aquela fonte no trecho; "
            "`pulado` = o trecho não foi auditado. O `#N` antes do trecho da fonte indica de "
            "qual trecho da resposta ele é evidência.",
            "",
            "| Referência | Sustenta | Parcial | Não sustenta | Trecho representativo da fonte |",
            "|---|---|---|---|---|",
        ]
        for s in rated:
            label = f"[{self._reference_label(s.reference_id, s.url)}]({s.url})"
            sustenta = _compress_positions(s.supports_positions) or "—"
            parcial = _compress_positions(s.partial_positions) or "—"
            nao_parts: list[str] = []
            if s.absent_positions:
                nao_parts.append(_compress_positions(s.absent_positions))
            if s.contradicts_positions:
                nao_parts.append(_compress_positions(s.contradicts_positions) + " ⚡")
            if s.unrated_positions:
                nao_parts.append(_compress_positions(s.unrated_positions) + " (s/ aval.)")
            if s.not_audited_positions:
                nao_parts.append(_compress_positions(s.not_audited_positions) + " (pulado)")
            nao = ", ".join(nao_parts) or "—"
            if s.key_excerpt:
                excerpt = _excerpt(s.key_excerpt, 160).replace("|", "\\|")
                if s.key_excerpt_position is not None:
                    excerpt = f"`#{s.key_excerpt_position}` {excerpt}"
            else:
                excerpt = "—"
            lines.append(f"| {label} | {sustenta} | {parcial} | {nao} | {excerpt} |")
        lines.append("")
        lines.append("> Detalhe de cada trecho (justificativa, aspectos não cobertos): ver *Exemplos*.")
        return ReportSection("Tabela de Verificação por Fonte", "\n".join(lines))

    def _reference_label(self, reference_id: str, fallback_url: str) -> str:
        reference = self._reference_by_id.get(reference_id)
        if reference is not None and reference.citation_markers:
            return " ".join(reference.citation_markers)
        return fallback_url

    def _markers_for(self, reference_ids: list[str]) -> str:
        labels = [self._reference_label(rid, rid) for rid in reference_ids]
        return " ".join(labels) if labels else "(nenhuma)"

    def _render_dead_and_inaccessible(self) -> ReportSection | None:
        report = self.report
        if not report.dead_references and not report.inaccessible_references:
            return None
        lines: list[str] = []
        if report.dead_references:
            lines.append("### Mortas (HTTP 404)\n")
            lines.append("\n".join(f"- {r.raw_url} — {r.error_message or 'sem detalhes'}" for r in report.dead_references))
        if report.inaccessible_references:
            lines.append("\n### Inacessíveis (403/timeout/SSL)\n")
            lines.append(
                "\n".join(f"- {r.raw_url} — {r.error_message or 'sem detalhes'}" for r in report.inaccessible_references)
            )
        return ReportSection("Referências Mortas e Inacessíveis", "\n".join(lines))

    _SKIP_REASON_LABELS = {
        "sem_citacao": "não citam nenhuma referência",
        "ref_sem_conteudo": "citam referência não baixada/inacessível",
        "outro": "outro motivo",
    }

    def _render_skipped_chunks(self) -> ReportSection | None:
        report = self.report
        skipped = report.skipped_chunks
        if not skipped:
            return None
        lines: list[str] = []
        if report.skipped_reason_counts:
            summary = "; ".join(
                f"{count} {self._SKIP_REASON_LABELS.get(key, key)}"
                for key, count in sorted(report.skipped_reason_counts.items())
            )
            lines.append(f"Motivos: {summary}.")
            lines.append("")
        lines.append("\n".join(f"- `{s.answer_chunk_id}` — {s.reason}" for s in skipped))
        return ReportSection("Chunks Não Auditados", "\n".join(lines))

    def _render_uncited_claims(self) -> ReportSection | None:
        items = self.report.uncited_claims
        if not items:
            return None
        lines = [
            f"**{self.report.count_uncited_claims} de {self.report.count_claim_chunks}** trechos com "
            "afirmação factual não têm nenhuma citação no documento. O marcador `[N]` no fim de um "
            "parágrafo sustenta aquele parágrafo; parágrafos anteriores sem marcador próprio entram aqui "
            "— as afirmações abaixo não têm fonte associada:",
            "",
        ]
        shown = items[:12]
        for c in shown:
            lines.append(f"- `{c.answer_chunk_id}` — {c.excerpt}")
        if len(items) > len(shown):
            lines.append(f"- … e mais {len(items) - len(shown)} trecho(s) — ver `report.json`.")
        return ReportSection("Afirmações sem Citação", "\n".join(lines))

    def _render_potentially_unsourced(self) -> ReportSection | None:
        chunks = self.report.potentially_unsourced_chunks
        if not chunks:
            return None
        lines = [
            "Chunks **citados** com várias frases em que só a última está diretamente ancorada pela "
            "citação — as frases anteriores do mesmo parágrafo podem não estar cobertas pela fonte:",
            "",
        ]
        for c in chunks:
            lines.append(
                f"- `{c.answer_chunk_id}` — {c.sentence_count} frases, cita {self._markers_for(c.cited_reference_ids)}"
            )
            lines.append(f"  - {c.excerpt}")
        return ReportSection("Chunks Citados com Várias Afirmações", "\n".join(lines))

    def _render_examples(self) -> ReportSection:
        by_verdict: dict[AuditVerdict, list[AuditResult]] = {v: [] for v in AuditVerdict}
        for result in self._results:
            by_verdict[result.verdict].append(result)

        blocks: list[str] = []
        for verdict, label in _VERDICT_LABELS.items():
            examples = by_verdict.get(verdict, [])[:_MAX_EXAMPLES_PER_VERDICT]
            if not examples:
                continue
            blocks.append(f"### {label}\n")
            for result in examples:
                blocks.append(self._render_example(result))
        return ReportSection("Exemplos Representativos por Veredito", "\n".join(blocks))

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
        lines = [
            f"- **Chunk `{result.answer_chunk_id}`** {ref_label}",
            f"  - Trecho: {chunk_excerpt}",
            f"  - Justificativa: {_excerpt(result.justification)}",
        ]
        if result.supporting_reference_ids:
            lines.append(f"  - Suporte veio de: {self._markers_for(result.supporting_reference_ids)}")
        if result.cited_excerpts:
            lines.append(f"  - Evidência: “{_excerpt(result.cited_excerpts[0])}”")
        if result.unsupported_aspects:
            lines.append(f"  - Não coberto pela evidência: {'; '.join(a for a in result.unsupported_aspects[:3])}")
        if len(result.per_reference) > 1 or (result.per_reference and chunk and len(chunk.cited_reference_ids) > 1):
            lines.append("  - Verificação por fonte:")
            lines.append("")
            lines.append("    | Fonte | Relação | Trecho |")
            lines.append("    |---|---|---|")
            for pr in result.per_reference:
                marker = self._reference_label(pr.reference_id, pr.reference_id)
                excerpt = _excerpt(pr.excerpt, 120).replace("|", "\\|") if pr.excerpt else "—"
                lines.append(f"    | {marker} | {_RELATION_LABELS.get(pr.relation, pr.relation)} | {excerpt} |")
            lines.append("")
        verification = self._verification_note(result)
        if verification:
            lines.append(f"  - Verificação: {verification}")
        return "\n".join(lines)

    def _verification_note(self, result: AuditResult) -> str:
        if not result.verification_trail:
            return ""
        path = " → ".join(step.stage for step in result.verification_trail)
        if result.verdict == AuditVerdict.UNSUPPORTED:
            note = "UNSUPPORTED confirmado" if result.unsupported_confirmed else "UNSUPPORTED (não confirmado)"
            if result.corroborated_by_other_reference:
                note += (
                    "; afirmação corroborada por outra fonte não citada: "
                    f"{self._markers_for(result.corroborated_by_other_reference)} "
                    "(provável citação trocada)"
                )
            if result.contradicted_by_other_reference:
                note += (
                    "; afirmação contradita por outra fonte: "
                    f"{self._markers_for(result.contradicted_by_other_reference)}"
                )
        else:
            baseline = next(
                (s.note for s in result.verification_trail if s.stage == "baseline"), ""
            )
            note = f"reclassificado para {result.verdict.value.upper()} na etapa `{result.verification_stage}`"
            if baseline:
                note += f' (baseline dizia: "{_excerpt(baseline, 140)}")'
        return f"{note} — cascata: {path}"


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
