from __future__ import annotations

import io
import json
import logging

import pytest
from rich.console import Console
from rich.logging import RichHandler

from auditframework import logging_config
from auditframework.config import Settings
from auditframework.logging_config import (
    RUN_LOG_FILENAME,
    RUN_LOG_RECORDS_FILENAME,
    _JsonFormatter,
    attach_run_log,
    detach_run_log,
    record_terminal_output,
    setup_logging,
    stage_banner,
    stage_done,
    stage_skipped,
)


@pytest.fixture(autouse=True)
def _reset_logging_state():
    """`setup_logging` e um singleton de modulo (`_CONFIGURED`) — sem
    resetar entre testes, so o primeiro `setup_logging` de toda a suite
    teria efeito."""
    logging_config._CONFIGURED = False
    logging_config._PRETTY = False
    yield
    detach_run_log()
    logging_config._CONFIGURED = False
    logging_config._PRETTY = False
    logging.getLogger().handlers.clear()


class TestSetupLogging:
    def test_text_format_installs_rich_handler(self):
        setup_logging(Settings(log_format="text"))

        handlers = logging.getLogger().handlers
        assert len(handlers) == 1
        assert isinstance(handlers[0], RichHandler)

    def test_json_format_installs_plain_handler_with_json_formatter(self):
        """Regressao: o modo pretty (RichHandler) nao pode vazar para o
        modo json — esse formato e consumido por maquina/agregador de
        log, onde decoracao visual quebraria o parsing."""
        setup_logging(Settings(log_format="json"))

        handlers = logging.getLogger().handlers
        assert len(handlers) == 1
        assert not isinstance(handlers[0], RichHandler)
        assert isinstance(handlers[0].formatter, _JsonFormatter)

    def test_second_call_is_a_noop(self):
        setup_logging(Settings(log_format="text"))
        setup_logging(Settings(log_format="json"))

        # a segunda chamada nao teve efeito: handler continua sendo o Rich
        # instalado pela primeira, nao o StreamHandler+_JsonFormatter
        assert isinstance(logging.getLogger().handlers[0], RichHandler)


class TestStageHelpers:
    def _fake_console(self) -> tuple[Console, io.StringIO]:
        buffer = io.StringIO()
        return Console(file=buffer, color_system=None, width=100), buffer

    def test_stage_banner_pretty_mode_writes_stage_name(self, monkeypatch):
        console, buffer = self._fake_console()
        monkeypatch.setattr(logging_config, "console", console)
        logging_config._PRETTY = True

        stage_banner("judging", 4, 5)

        output = buffer.getvalue()
        assert "judging" in output
        assert "4/5" in output

    def test_stage_done_pretty_mode_writes_elapsed_time(self, monkeypatch):
        console, buffer = self._fake_console()
        monkeypatch.setattr(logging_config, "console", console)
        logging_config._PRETTY = True

        stage_done("indexing", 12.3)

        output = buffer.getvalue()
        assert "indexing" in output
        assert "12.3" in output

    def test_stage_skipped_pretty_mode_writes_stage_name(self, monkeypatch):
        console, buffer = self._fake_console()
        monkeypatch.setattr(logging_config, "console", console)
        logging_config._PRETTY = True

        stage_skipped("reporting")

        assert "reporting" in buffer.getvalue()

    def test_stage_helpers_fallback_to_plain_logging_outside_pretty_mode(self, caplog):
        """Fora do modo pretty (ex: log_format=json), as funcoes de banner
        nao devem desenhar nada com Rich — so logar uma linha comum, para
        nao poluir um stream que precisa continuar sendo JSON valido."""
        logging_config._PRETTY = False

        with caplog.at_level(logging.INFO):
            stage_banner("judging", 1, 1)
            stage_done("judging", 1.0)
            stage_skipped("judging")

        messages = [record.getMessage() for record in caplog.records]
        assert any("judging" in m for m in messages)
        assert len(messages) == 3

    def test_unknown_stage_name_does_not_crash(self, monkeypatch):
        console, buffer = self._fake_console()
        monkeypatch.setattr(logging_config, "console", console)
        logging_config._PRETTY = True

        stage_banner("estagio-futuro-desconhecido", 1, 1)

        assert "estagio-futuro-desconhecido" in buffer.getvalue()


class TestJsonFormatterUnaffectedByStageHelpers:
    def test_json_lines_remain_parseable_with_pretty_helpers_disabled(self, caplog):
        """Roteiro end-to-end simplificado do modo json: cada linha
        efetivamente emitida pelo handler configurado precisa continuar
        sendo um JSON valido."""
        setup_logging(Settings(log_format="json"))
        logger = logging.getLogger("auditframework.pipeline")

        formatter = _JsonFormatter()
        record = logger.makeRecord(logger.name, logging.INFO, __file__, 0, "Executando estagio 1/1: extraction", (), None)
        line = formatter.format(record)

        parsed = json.loads(line)
        assert parsed["message"] == "Executando estagio 1/1: extraction"
        assert parsed["level"] == "INFO"


class TestRunLog:
    """`log.txt` na pasta da run: tudo o que o terminal mostra durante o
    pipeline, em texto puro, para conferir a execucao depois."""

    def _fake_console(self) -> tuple[Console, io.StringIO]:
        buffer = io.StringIO()
        return Console(file=buffer, color_system=None, width=100), buffer

    def _read(self, run_dir) -> str:
        return (run_dir / RUN_LOG_FILENAME).read_text(encoding="utf-8")

    def test_attach_writes_log_records_to_log_txt_in_run_dir(self, tmp_path):
        setup_logging(Settings(log_format="text"))
        run_dir = tmp_path / "run-1"

        path = attach_run_log(run_dir)
        logging.getLogger("auditframework.pipeline").info("Extraidas %d referencias", 7)
        detach_run_log()

        assert path == run_dir / "log.txt"
        content = self._read(run_dir)
        assert "INFO" in content
        assert "auditframework.pipeline: Extraidas 7 referencias" in content

    def test_attach_appends_so_resume_keeps_previous_log(self, tmp_path):
        setup_logging(Settings(log_format="text"))
        logger = logging.getLogger("auditframework.cli")

        attach_run_log(tmp_path)
        logger.info("primeira execucao")
        detach_run_log()
        attach_run_log(tmp_path)
        logger.info("retomada")
        detach_run_log()

        content = self._read(tmp_path)
        assert content.index("primeira execucao") < content.index("retomada")

    def test_detach_stops_writing_to_file(self, tmp_path):
        setup_logging(Settings(log_format="text"))
        logger = logging.getLogger("auditframework.cli")

        attach_run_log(tmp_path)
        detach_run_log()
        logger.info("depois de desanexar")

        assert "depois de desanexar" not in self._read(tmp_path)

    def test_attach_twice_does_not_duplicate_lines(self, tmp_path):
        setup_logging(Settings(log_format="text"))

        attach_run_log(tmp_path)
        attach_run_log(tmp_path)
        logging.getLogger("auditframework.cli").info("linha unica")
        detach_run_log()

        assert self._read(tmp_path).count("linha unica") == 1

    def test_pretty_stage_helpers_are_recorded_as_plain_text(self, tmp_path, monkeypatch):
        """No modo pretty os banners vao direto para o console Rich, sem
        passar pelo logging — mesmo assim precisam aparecer no log.txt."""
        console, _ = self._fake_console()
        monkeypatch.setattr(logging_config, "console", console)
        setup_logging(Settings(log_format="text"))

        attach_run_log(tmp_path)
        stage_banner("judging", 4, 5)
        stage_done("judging", 12.3)
        stage_skipped("reporting")
        detach_run_log()

        content = self._read(tmp_path)
        assert "Estágio 4/5 · judging" in content
        assert "judging concluído em 12.3s" in content
        assert "Pulando estágio já concluído: reporting" in content
        assert "[bold" not in content

    def test_plain_stage_helpers_are_recorded_once(self, tmp_path):
        setup_logging(Settings(log_format="json"))

        attach_run_log(tmp_path)
        stage_banner("judging", 1, 1)
        detach_run_log()

        assert self._read(tmp_path).count("Executando estagio 1/1: judging") == 1

    def test_record_terminal_output_goes_only_to_file(self, tmp_path, capsys):
        setup_logging(Settings(log_format="json"))

        attach_run_log(tmp_path)
        record_terminal_output("Relatorio gerado em /x/report.md")
        detach_run_log()

        assert "Relatorio gerado em /x/report.md" in self._read(tmp_path)
        assert "Relatorio gerado" not in capsys.readouterr().out

    def test_record_terminal_output_without_attached_log_is_noop(self):
        record_terminal_output("nada acontece")


class TestRunLogMirrorsTerminal:
    """Tudo o que vai para stdout/stderr durante a run (print, warnings,
    barras de progresso de bibliotecas) tambem entra no log.txt."""

    def _read(self, run_dir) -> str:
        return (run_dir / RUN_LOG_FILENAME).read_text(encoding="utf-8")

    def test_print_to_stdout_is_recorded(self, tmp_path):
        setup_logging(Settings(log_format="text"))

        attach_run_log(tmp_path)
        print("saida direta de uma biblioteca")
        detach_run_log()

        assert "saida direta de uma biblioteca" in self._read(tmp_path)

    def test_python_warning_on_stderr_is_recorded(self, tmp_path):
        import warnings

        setup_logging(Settings(log_format="text"))

        attach_run_log(tmp_path)
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn("argumento obsoleto", FutureWarning)
        detach_run_log()

        assert "FutureWarning: argumento obsoleto" in self._read(tmp_path)

    def test_progress_bar_is_recorded_once_in_its_final_state(self, tmp_path):
        import sys

        setup_logging(Settings(log_format="text"))

        attach_run_log(tmp_path)
        for pct in (10, 50, 100):
            sys.stderr.write(f"\rJulgando chunks: {pct}%")
        sys.stderr.write("\n")
        detach_run_log()

        content = self._read(tmp_path)
        assert content.count("Julgando chunks") == 1
        assert "Julgando chunks: 100%" in content

    def test_interrupted_progress_bar_keeps_its_last_state(self, tmp_path):
        import sys

        setup_logging(Settings(log_format="text"))

        attach_run_log(tmp_path)
        sys.stderr.write("\rJulgando chunks: 10%")
        sys.stderr.write("\rJulgando chunks: 68%")
        detach_run_log()

        assert "Julgando chunks: 68%" in self._read(tmp_path)

    def test_ansi_color_codes_are_stripped(self, tmp_path):
        import sys

        setup_logging(Settings(log_format="text"))

        attach_run_log(tmp_path)
        sys.stdout.write("\x1b[31mtexto vermelho\x1b[0m\n")
        detach_run_log()

        content = self._read(tmp_path)
        assert "texto vermelho" in content
        assert "\x1b" not in content

    def test_log_records_are_not_duplicated_by_the_mirror(self, tmp_path):
        setup_logging(Settings(log_format="text"))

        attach_run_log(tmp_path)
        logging.getLogger("auditframework.pipeline").warning("aviso unico")
        stage_banner("judging", 4, 5)
        detach_run_log()

        content = self._read(tmp_path)
        assert content.count("aviso unico") == 1
        assert content.count("judging") == 1

    def test_detach_restores_stdout_and_stderr(self, tmp_path):
        import sys

        out, err = sys.stdout, sys.stderr
        attach_run_log(tmp_path)
        detach_run_log()

        assert sys.stdout is out
        assert sys.stderr is err


class TestRunLogFormats:
    """`log.txt` quebra as linhas longas para leitura (largura 120, recuo
    alinhado com o inicio da mensagem); `log_registros.txt` tem uma linha
    por registro, para busca e script."""

    LONG = "inicio " + "palavra " * 60 + "fim"

    def _read(self, run_dir, name) -> str:
        return (run_dir / name).read_text(encoding="utf-8")

    def _log(self, tmp_path, emit):
        setup_logging(Settings(log_format="text"))
        attach_run_log(tmp_path)
        emit(logging.getLogger("auditframework.pipeline"))
        detach_run_log()
        return self._read(tmp_path, RUN_LOG_FILENAME), self._read(tmp_path, RUN_LOG_RECORDS_FILENAME)

    def test_records_file_lives_next_to_log_txt(self):
        assert RUN_LOG_RECORDS_FILENAME == "log_registros.txt"

    def test_long_message_wraps_in_log_txt_with_aligned_indent(self, tmp_path):
        legivel, _ = self._log(tmp_path, lambda log: log.warning(self.LONG))

        lines = legivel.splitlines()
        assert len(lines) > 1
        assert all(len(line) <= 120 for line in lines)
        indent = " " * len("2026-10-18 14:25:16,821 WARNING  ")
        assert all(line.startswith(indent) and line[len(indent)] != " " for line in lines[1:])
        assert " ".join(line.strip() for line in lines).endswith(self.LONG)

    def test_long_message_is_one_line_in_records_file(self, tmp_path):
        _, registros = self._log(tmp_path, lambda log: log.warning(self.LONG))

        lines = registros.splitlines()
        assert len(lines) == 1
        assert lines[0].endswith(f"auditframework.pipeline: {self.LONG}")

    def test_multiline_message_is_escaped_in_records_and_indented_in_log_txt(self, tmp_path):
        legivel, registros = self._log(tmp_path, lambda log: log.warning('resposta: {\n  "verdict": "x"\n}'))

        assert registros.splitlines() == [registros.rstrip("\n")]
        assert 'resposta: {\\n  "verdict": "x"\\n}' in registros
        indent = " " * len("2026-10-18 14:25:16,821 WARNING  ")
        assert legivel.splitlines()[1] == indent + '  "verdict": "x"'

    def test_traceback_is_one_line_in_records_file(self, tmp_path):
        def emit(log):
            try:
                raise RuntimeError("falha simulada")
            except RuntimeError:
                log.exception("estagio quebrou")

        legivel, registros = self._log(tmp_path, emit)

        assert len(registros.splitlines()) == 1
        assert "Traceback" in registros and "falha simulada" in registros
        assert "Traceback" in legivel and "falha simulada" in legivel

    def test_long_url_is_not_split_in_log_txt(self, tmp_path):
        url = "https://exemplo.org/" + "a" * 150
        legivel, _ = self._log(tmp_path, lambda log: log.warning("Barreira de acesso em %s", url))

        assert url in legivel

    def test_terminal_output_goes_to_both_files(self, tmp_path):
        setup_logging(Settings(log_format="text"))
        attach_run_log(tmp_path)
        print("saida direta")
        detach_run_log()

        assert "saida direta" in self._read(tmp_path, RUN_LOG_FILENAME)
        assert "saida direta" in self._read(tmp_path, RUN_LOG_RECORDS_FILENAME)
