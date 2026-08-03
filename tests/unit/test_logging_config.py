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
    _JsonFormatter,
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
