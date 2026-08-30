from __future__ import annotations

import json

import pytest

from auditframework.common.pricing import is_cost_tracked
from auditframework.models import (
    AnswerChunk,
    AuditResult,
    AuditVerdict,
    JudgeConfig,
    Reference,
    ReferenceStatus,
    SkippedChunk,
    VerificationStep,
)
from auditframework.reporting import (
    aggregate_report,
    aggregate_tool_stats,
    build_reference_stats,
    render_json,
    render_markdown,
    render_tool_comparison_markdown,
    summarize_cost,
)
from auditframework.reporting.render import _format_duration


def _reference(ref_id: str, *, status: ReferenceStatus = ReferenceStatus.DOWNLOADED, **overrides) -> Reference:
    defaults = dict(
        id=ref_id,
        citation_markers=[f"[{ref_id}]"],
        raw_url=f"https://example.com/{ref_id}",
        normalized_url=f"https://example.com/{ref_id}",
        status=status,
        source_answer_id="answer-1",
        tool_name="ChatGPT",
    )
    defaults.update(overrides)
    return Reference(**defaults)


def _chunk(chunk_id: str, cited: list[str], text: str = "texto do chunk") -> AnswerChunk:
    return AnswerChunk(id=chunk_id, answer_id="answer-1", position=0, text=text, cited_reference_ids=cited)


def _result(chunk_id: str, verdict: AuditVerdict, **overrides) -> AuditResult:
    defaults = dict(
        answer_chunk_id=chunk_id,
        verdict=verdict,
        justification="justificativa qualquer",
        judge_model="local/gpt-oss-20b",
    )
    defaults.update(overrides)
    return AuditResult(**defaults)


class TestSummarizeCost:
    def test_sums_cost_and_tokens_across_results(self):
        results = [
            _result("c1", AuditVerdict.SUPPORTED, cost_usd=0.01, prompt_tokens=100, completion_tokens=20),
            _result("c2", AuditVerdict.UNSUPPORTED, cost_usd=0.02, prompt_tokens=200, completion_tokens=40),
        ]
        summary = summarize_cost(results)
        assert summary.total_cost_usd == 0.03
        assert summary.total_prompt_tokens == 300
        assert summary.total_completion_tokens == 60
        assert summary.total_tokens == 360

    def test_empty_results_yields_zeroed_summary(self):
        summary = summarize_cost([])
        assert summary.total_cost_usd == 0.0
        assert summary.total_tokens == 0


class TestBuildReferenceStats:
    def test_counts_citations_and_verdicts_per_reference(self):
        references = [_reference("r1"), _reference("r2")]
        chunks = [_chunk("c1", ["r1"]), _chunk("c2", ["r1"]), _chunk("c3", ["r2"])]
        results = [
            _result("c1", AuditVerdict.SUPPORTED),
            _result("c2", AuditVerdict.UNSUPPORTED),
            _result("c3", AuditVerdict.CONTRADICTED),
        ]

        stats = build_reference_stats(chunks, references, results)
        stats_by_id = {s.reference_id: s for s in stats}

        assert stats_by_id["r1"].times_cited == 2
        assert stats_by_id["r1"].supported_count == 1
        assert stats_by_id["r1"].unsupported_count == 1
        assert stats_by_id["r2"].times_cited == 1
        assert stats_by_id["r2"].contradicted_count == 1

    def test_sorted_by_times_cited_descending(self):
        references = [_reference("r1"), _reference("r2")]
        chunks = [_chunk("c1", ["r2"]), _chunk("c2", ["r1"]), _chunk("c3", ["r1"])]
        results = [_result(c.id, AuditVerdict.SUPPORTED) for c in chunks]

        stats = build_reference_stats(chunks, references, results)

        assert [s.reference_id for s in stats] == ["r1", "r2"]

    def test_reference_cited_but_not_in_reference_list_is_skipped_not_crashed(self):
        chunks = [_chunk("c1", ["ghost-ref"])]
        results = [_result("c1", AuditVerdict.SUPPORTED)]

        stats = build_reference_stats(chunks, references=[], results=results)

        assert stats == []

    def test_chunk_without_matching_result_is_still_counted_as_cited(self):
        references = [_reference("r1")]
        chunks = [_chunk("c1", ["r1"])]

        stats = build_reference_stats(chunks, references, results=[])

        assert stats[0].times_cited == 1
        assert stats[0].supported_count == 0

    def test_reference_never_cited_by_any_chunk_still_appears_zeroed(self):
        """Uma referencia extraida da resposta mas nunca citada por nenhum
        chunk (ex: link listado no rodape mas sem chunk correspondente)
        deve continuar visivel na secao 4 do relatorio, com times_cited=0,
        em vez de desaparecer silenciosamente."""
        references = [_reference("r1"), _reference("r2")]
        chunks = [_chunk("c1", ["r1"])]
        results = [_result("c1", AuditVerdict.SUPPORTED)]

        stats = build_reference_stats(chunks, references, results)
        stats_by_id = {s.reference_id: s for s in stats}

        assert set(stats_by_id) == {"r1", "r2"}
        assert stats_by_id["r2"].times_cited == 0
        assert stats_by_id["r2"].supported_count == 0
        assert stats_by_id["r2"].unsupported_count == 0
        assert stats_by_id["r2"].contradicted_count == 0


class TestAggregateReport:
    def test_percentages_and_totals_are_computed_correctly(self):
        references = [_reference("r1"), _reference("r2", status=ReferenceStatus.DEAD)]
        chunks = [_chunk("c1", ["r1"]), _chunk("c2", ["r1"]), _chunk("c3", [])]
        results = [
            _result("c1", AuditVerdict.SUPPORTED, cost_usd=0.01, prompt_tokens=10, completion_tokens=5),
            _result("c2", AuditVerdict.UNSUPPORTED, cost_usd=0.02, prompt_tokens=20, completion_tokens=10),
            _result("c3", AuditVerdict.CONTRADICTED),
        ]

        report = aggregate_report(
            run_id="run-1",
            answer_id="answer-1",
            tool_name="ChatGPT",
            chunks=chunks,
            references=references,
            results=results,
            processing_time_seconds=12.5,
        )

        assert report.total_chunks == 3
        assert report.pct_supported == pytest.approx(100 / 3)
        assert report.pct_unsupported == pytest.approx(100 / 3)
        assert report.pct_contradicted == pytest.approx(100 / 3)
        assert report.total_cost_usd == pytest.approx(0.03)
        assert report.total_tokens == 45
        assert report.processing_time_seconds == 12.5
        assert [r.id for r in report.dead_references] == ["r2"]
        assert report.inaccessible_references == []

    def test_empty_results_do_not_raise_division_by_zero(self):
        report = aggregate_report(
            run_id="run-1", answer_id="answer-1", tool_name="ChatGPT", chunks=[], references=[], results=[]
        )
        assert report.pct_supported == 0.0
        assert report.total_chunks == 0

    def test_skipped_chunks_are_counted_and_percentages_account_for_them(self):
        chunks = [_chunk("c1", []), _chunk("c2", []), _chunk("c3", [])]
        results = [_result("c1", AuditVerdict.SUPPORTED)]
        skipped = [SkippedChunk(answer_chunk_id="c2", reason="referencia nao baixada")]

        report = aggregate_report(
            run_id="run-1",
            answer_id="answer-1",
            tool_name="ChatGPT",
            chunks=chunks,
            references=[],
            results=results,
            skipped=skipped,
        )

        assert report.count_supported == 1
        assert report.count_skipped == 1
        assert report.skipped_chunks == skipped
        assert report.total_chunks == 3
        assert report.pct_supported == pytest.approx(100 / 3)


class TestAggregateToolStats:
    def test_computes_percentages_for_a_single_tool(self):
        results = [
            _result("c1", AuditVerdict.SUPPORTED),
            _result("c2", AuditVerdict.SUPPORTED),
            _result("c3", AuditVerdict.UNSUPPORTED),
            _result("c4", AuditVerdict.CONTRADICTED),
        ]
        stats = aggregate_tool_stats("ChatGPT", results)
        assert stats.tool_name == "ChatGPT"
        assert stats.total_chunks == 4
        assert stats.pct_supported == 50.0
        assert stats.pct_unsupported == 25.0
        assert stats.pct_contradicted == 25.0


class TestRender:
    def _sample_report(self):
        references = [_reference("r1"), _reference("r2", status=ReferenceStatus.DEAD, error_message="HTTP 404")]
        chunks = [_chunk("c1", ["r1"], text="O Marco Civil da Internet estabelece [1] princípios."), _chunk("c2", ["r2"])]
        results = [
            _result("c1", AuditVerdict.SUPPORTED, justification="A referencia confirma o principio citado."),
            _result("c2", AuditVerdict.UNSUPPORTED, justification="Referencia morta, sem evidencia."),
        ]
        report = aggregate_report(
            run_id="run-1",
            answer_id="answer-1",
            tool_name="ChatGPT",
            chunks=chunks,
            references=references,
            results=results,
            processing_time_seconds=3.2,
        )
        return report, chunks, references, results

    def test_markdown_contains_key_sections(self):
        report, chunks, references, results = self._sample_report()
        markdown = render_markdown(report, chunks=chunks, references=references, results=results)

        assert "# Relatório de Auditoria" in markdown
        assert "## 1. Metadados da Execução" in markdown
        assert "## 2. Distribuição de Vereditos" in markdown
        assert "## 3. Custo e Uso de Tokens" in markdown
        assert "## 4. Análise por Referência" in markdown
        assert "## 5. Referências Mortas e Inacessíveis" in markdown
        assert "## 6. Exemplos Representativos por Veredito" in markdown
        assert "HTTP 404" in markdown

    def test_reference_table_uses_citation_marker_not_raw_url_as_label(self):
        report, chunks, references, results = self._sample_report()
        markdown = render_markdown(report, chunks=chunks, references=references, results=results)
        assert "| [[r1]](https://example.com/r1) |" in markdown

    def test_distribution_table_includes_chunk_counts_and_total(self):
        report, chunks, references, results = self._sample_report()
        markdown = render_markdown(report, chunks=chunks, references=references, results=results)

        assert "| Veredito | Chunks | Percentual |" in markdown
        # _sample_report tem 1 SUPPORTED e 1 UNSUPPORTED, 0 CONTRADICTED
        assert "| **SUPPORTED** | 1 | 50.0% |" in markdown
        assert "| **UNSUPPORTED** | 1 | 50.0% |" in markdown
        assert "| **CONTRADICTED** | 0 | 0.0% |" in markdown
        assert "| **TOTAL** | 2 | 100.0% |" in markdown

    def test_cost_table_includes_averages_per_request(self):
        chunks = [_chunk("c1", ["r1"]), _chunk("c2", ["r1"])]
        results = [
            _result("c1", AuditVerdict.SUPPORTED, cost_usd=0.02, prompt_tokens=100, completion_tokens=20),
            _result("c2", AuditVerdict.UNSUPPORTED, cost_usd=0.04, prompt_tokens=300, completion_tokens=60),
        ]
        report = aggregate_report(
            run_id="run-1", answer_id="answer-1", tool_name="ChatGPT",
            chunks=chunks, references=[], results=results,
        )
        markdown = render_markdown(report, chunks=chunks, results=results)

        # total: 0.06 USD, 480 tokens, 2 requisicoes -> media 0.03 USD / 240 tokens
        assert "| **Média de tokens por requisição** | 240.0 |" in markdown
        assert "| **Média de custo por requisição** | US$ 0.0300 |" in markdown

    def test_cost_table_averages_are_zero_with_no_judged_chunks(self):
        report = aggregate_report(
            run_id="run-1", answer_id="answer-1", tool_name="ChatGPT",
            chunks=[], references=[], results=[],
        )
        markdown = render_markdown(report)

        assert "| **Média de tokens por requisição** | 0.0 |" in markdown
        assert "| **Média de custo por requisição** | US$ 0.0000 |" in markdown

    def test_distribution_table_has_no_skipped_row_when_nothing_was_skipped(self):
        report, chunks, references, results = self._sample_report()
        markdown = render_markdown(report, chunks=chunks, references=references, results=results)
        assert "SKIPPED" not in markdown
        assert "Chunks Não Auditados" not in markdown

    def test_distribution_table_and_section_show_skipped_chunks(self):
        chunks = [_chunk("c1", ["r1"]), _chunk("c2", [])]
        results = [_result("c1", AuditVerdict.SUPPORTED)]
        skipped = [SkippedChunk(answer_chunk_id="c2", reason="chunk nao cita nenhuma referencia")]
        report = aggregate_report(
            run_id="run-1", answer_id="answer-1", tool_name="ChatGPT",
            chunks=chunks, references=[_reference("r1")], results=results, skipped=skipped,
        )
        markdown = render_markdown(report, chunks=chunks, results=results)

        assert "| **SKIPPED** | 1 | 50.0% |" in markdown
        assert "| **TOTAL** | 2 | 100.0% |" in markdown
        # sem referencias mortas nesse cenario, entao a secao 5 (mortas/inacessiveis)
        # nao existe e "Chunks Não Auditados" e numerada sem deixar buraco.
        assert "## 6. Chunks Não Auditados" in markdown
        assert "`c2` — chunk nao cita nenhuma referencia" in markdown

    def test_section_numbering_has_no_gaps_when_conditional_sections_are_absent(self):
        # so as 4 secoes sempre presentes: metadados, distribuicao, custo, referencias.
        chunks = [_chunk("c1", ["r1"])]
        results = [_result("c1", AuditVerdict.SUPPORTED)]
        report = aggregate_report(
            run_id="run-1", answer_id="answer-1", tool_name="ChatGPT",
            chunks=chunks, references=[_reference("r1")], results=results,
        )
        markdown = render_markdown(report)  # sem results -> sem secao de exemplos

        assert "## 1. Metadados da Execução" in markdown
        assert "## 2. Distribuição de Vereditos" in markdown
        assert "## 3. Custo e Uso de Tokens" in markdown
        assert "## 4. Análise por Referência" in markdown
        assert "## 5." not in markdown
        assert "Referências Mortas e Inacessíveis" not in markdown
        assert "Chunks Não Auditados" not in markdown

    def test_markdown_without_raw_results_skips_examples_section(self):
        report, chunks, references, _results = self._sample_report()
        markdown = render_markdown(report, chunks=chunks, references=references)
        assert "Exemplos Representativos" not in markdown

    def test_json_round_trips_report_fields(self):
        report, *_ = self._sample_report()
        raw = render_json(report)
        parsed = json.loads(raw)
        assert parsed["run_id"] == "run-1"
        assert parsed["tool_name"] == "ChatGPT"

    def test_tool_comparison_table_lists_every_tool(self):
        stats = [aggregate_tool_stats("ChatGPT", [_result("c1", AuditVerdict.SUPPORTED)]),
                 aggregate_tool_stats("Gemini", [_result("c1", AuditVerdict.UNSUPPORTED)])]
        table = render_tool_comparison_markdown(stats)
        assert "ChatGPT" in table
        assert "Gemini" in table

    def test_empty_tool_comparison_does_not_crash(self):
        assert "Nenhum dado" in render_tool_comparison_markdown([])

    def test_processing_time_is_rendered_as_days_hours_min_sec(self):
        report = aggregate_report(
            run_id="run-1", answer_id="answer-1", tool_name="ChatGPT",
            chunks=[], references=[], results=[],
            processing_time_seconds=90_061.4,  # 1d 01:01:01
        )
        markdown = render_markdown(report)
        assert "| **Tempo de processamento** | 1:01:01:01 |" in markdown

    def test_header_includes_audited_file_path_when_present(self):
        report = aggregate_report(
            run_id="run-1", answer_id="answer-1", tool_name="ChatGPT",
            answer_path="answers/gemini_direito.pdf", chunks=[], references=[], results=[],
        )
        markdown = render_markdown(report)
        assert "**Arquivo auditado:** `answers/gemini_direito.pdf`" in markdown

    def test_header_omits_file_path_when_absent(self):
        report = aggregate_report(
            run_id="run-1", answer_id="answer-1", tool_name="ChatGPT",
            chunks=[], references=[], results=[],
        )
        assert "Arquivo auditado" not in render_markdown(report)


class TestFormatDuration:
    def test_sub_minute(self):
        assert _format_duration(3.2) == "0:00:00:03"

    def test_hours_and_minutes(self):
        assert _format_duration(3_600 + 12 * 60 + 5) == "0:01:12:05"

    def test_multiple_days(self):
        assert _format_duration(2 * 86_400 + 3 * 3_600) == "2:03:00:00"

    def test_negative_clamped_to_zero(self):
        assert _format_duration(-5) == "0:00:00:00"


def _judge_config(provider: str, model: str) -> JudgeConfig:
    return JudgeConfig(provider=provider, model=model, temperature=0.0, max_retries=3, retry_delay=2.0)


class TestIsCostTracked:
    def test_local_and_ollama_are_tracked(self):
        assert is_cost_tracked("local", "openai/gpt-oss-20b") is True
        assert is_cost_tracked("ollama", "qualquer-modelo") is True

    def test_anthropic_tracked_only_for_known_model(self):
        assert is_cost_tracked("anthropic", "claude-sonnet-5") is True
        assert is_cost_tracked("anthropic", "claude-modelo-inexistente") is False

    def test_openai_and_unknown_providers_are_not_tracked(self):
        assert is_cost_tracked("openai", "gpt-4o") is False
        assert is_cost_tracked("outro", "x") is False


class TestCostTrackedInReport:
    def _report(self, judge_config: JudgeConfig | None):
        chunks = [_chunk("c1", ["r1"])]
        results = [_result("c1", AuditVerdict.SUPPORTED, cost_usd=0.0, prompt_tokens=100, completion_tokens=20)]
        return aggregate_report(
            run_id="run-1", answer_id="answer-1", tool_name="ChatGPT",
            chunks=chunks, references=[_reference("r1")], results=results, judge_config=judge_config,
        )

    def test_report_defaults_to_cost_tracked_without_judge_config(self):
        assert self._report(None).cost_tracked is True

    def test_report_flags_untracked_cost_for_openai(self):
        report = self._report(_judge_config("openai", "gpt-4o"))
        assert report.cost_tracked is False

    def test_report_keeps_cost_tracked_for_known_anthropic_model(self):
        report = self._report(_judge_config("anthropic", "claude-sonnet-5"))
        assert report.cost_tracked is True

    def test_markdown_shows_warning_when_cost_not_tracked(self):
        report = self._report(_judge_config("openai", "gpt-4o"))
        markdown = render_markdown(report)
        assert "Custo não contabilizado" in markdown
        assert "`openai`" in markdown
        assert "`gpt-4o`" in markdown

    def test_markdown_has_no_warning_when_cost_tracked(self):
        report = self._report(_judge_config("anthropic", "claude-sonnet-5"))
        markdown = render_markdown(report)
        assert "Custo não contabilizado" not in markdown


def _verified_result(chunk_id, verdict, *, stage="baseline", confirmed=False, corroborated=(), trail_stages=("baseline",)):
    return AuditResult(
        answer_chunk_id=chunk_id, verdict=verdict, justification="j", judge_model="m",
        verification_stage=stage, unsupported_confirmed=confirmed,
        corroborated_by_other_reference=list(corroborated),
        verification_trail=[VerificationStep(stage=s, verdict=verdict, note="n") for s in trail_stages],
    )


class TestVerificationSection:
    def _report(self, results):
        chunks = [_chunk(r.answer_chunk_id, ["r1"]) for r in results]
        return aggregate_report(
            run_id="run-1", answer_id="answer-1", tool_name="ChatGPT",
            chunks=chunks, references=[_reference("r1")], results=results,
        )

    def test_no_section_when_verification_never_ran(self):
        report = self._report([_result("c1", AuditVerdict.SUPPORTED)])
        assert report.verification_ran is False
        assert "Verificação de UNSUPPORTED" not in render_markdown(report)

    def test_counts_confirmed_reclassified_and_mis_cited(self):
        results = [
            _verified_result("c1", AuditVerdict.UNSUPPORTED, confirmed=True,
                             trail_stages=("baseline", "context_expansion", "full_doc_scan", "cross_reference")),
            _verified_result("c2", AuditVerdict.SUPPORTED, stage="context_expansion",
                             trail_stages=("baseline", "context_expansion")),
            _verified_result("c3", AuditVerdict.UNSUPPORTED, confirmed=True, corroborated=("r9",),
                             trail_stages=("baseline", "context_expansion", "full_doc_scan", "cross_reference")),
        ]
        report = self._report(results)
        assert report.verification_ran is True
        assert report.count_unsupported_confirmed == 2
        assert report.count_reclassified_by_verification == 1
        assert report.mis_cited_reference_count == 1
        assert report.verification_stage_counts == {"context_expansion": 1}

        markdown = render_markdown(report)
        assert "## 3. Verificação de UNSUPPORTED" in markdown
        assert "confirmados" in markdown and "reclassificados" in markdown
        assert "provável erro de citação" in markdown
