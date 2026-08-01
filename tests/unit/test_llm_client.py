from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from auditframework.common.errors import ConfigurationError, LLMParseError, LLMProviderError
from auditframework.common.llm_client import (
    AnthropicClient,
    OpenAICompatibleClient,
    create_llm_client,
    parse_json_object,
)
from auditframework.common.pricing import cost_usd


class _Schema(BaseModel):
    verdict: str
    justification: str


def _settings(**overrides) -> SimpleNamespace:
    base = dict(
        llm_provider="local",
        llm_model="openai/gpt-oss-20b",
        openai_api_key=None,
        llm_base_url="http://localhost:1234/v1/",
        anthropic_api_key=None,
        llm_temperature=0.0,
        llm_max_retries=1,
        llm_retry_delay=0.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestParseJsonObject:
    def test_parses_plain_json(self):
        assert parse_json_object('{"a": 1}') == {"a": 1}

    def test_parses_json_wrapped_in_markdown_fence(self):
        raw = 'Aqui esta a resposta:\n```json\n{"a": 1}\n```\nFim.'
        assert parse_json_object(raw) == {"a": 1}

    def test_parses_bare_object_with_surrounding_prose(self):
        raw = 'A resposta e {"a": 1} conforme solicitado.'
        assert parse_json_object(raw) == {"a": 1}

    def test_invalid_json_raises_llm_parse_error(self):
        with pytest.raises(LLMParseError):
            parse_json_object("isso nao e json nenhum")


class TestCostUsd:
    def test_local_provider_is_always_free(self):
        assert cost_usd("local", "qualquer-modelo", 1_000_000, 1_000_000) == 0.0

    def test_anthropic_known_model_computes_correctly(self):
        cost = cost_usd("anthropic", "claude-sonnet-5", 1_000_000, 1_000_000)
        assert cost == pytest.approx(3.00 + 15.00)

    def test_anthropic_unknown_model_returns_zero(self):
        assert cost_usd("anthropic", "modelo-desconhecido", 1000, 1000) == 0.0

    def test_openai_has_no_builtin_table(self):
        assert cost_usd("openai", "gpt-4", 1000, 1000) == 0.0


class TestCreateLlmClient:
    def test_local_provider_returns_openai_compatible_client(self):
        client = create_llm_client(_settings(llm_provider="local"))
        assert isinstance(client, OpenAICompatibleClient)
        assert client.model == "openai/gpt-oss-20b"

    def test_configured_temperature_and_retry_settings_are_passed_through(self):
        client = create_llm_client(_settings(llm_provider="local", llm_temperature=0.2, llm_max_retries=5, llm_retry_delay=3.0))
        assert client.temperature == 0.2
        assert client.max_retries == 5
        assert client.retry_delay == 3.0

    def test_anthropic_without_api_key_raises_configuration_error(self):
        with pytest.raises(ConfigurationError):
            create_llm_client(_settings(llm_provider="anthropic", anthropic_api_key=None))

    def test_anthropic_with_api_key_returns_anthropic_client(self):
        client = create_llm_client(
            _settings(llm_provider="anthropic", anthropic_api_key="sk-ant-test", llm_model="claude-sonnet-5")
        )
        assert isinstance(client, AnthropicClient)
        assert client.model == "claude-sonnet-5"

    def test_unknown_provider_raises_configuration_error(self):
        with pytest.raises(ConfigurationError):
            create_llm_client(_settings(llm_provider="bogus"))


class TestOpenAICompatibleClient:
    def test_complete_json_parses_response_and_computes_usage(self):
        client = OpenAICompatibleClient(model="m", provider="local", api_key=None, base_url="http://x")
        fake_response = MagicMock()
        fake_response.choices[0].message.content = '{"verdict": "supported", "justification": "ok"}'
        fake_response.usage.prompt_tokens = 50
        fake_response.usage.completion_tokens = 10
        client._client.chat.completions.create = MagicMock(return_value=fake_response)

        instance, usage = client.complete_json(system_message="sys", user_prompt="user", schema=_Schema)

        assert instance.verdict == "supported"
        assert usage.prompt_tokens == 50
        assert usage.completion_tokens == 10
        assert usage.cost_usd == 0.0  # provider local

    def test_configured_temperature_is_forwarded_to_the_sdk_call(self):
        client = OpenAICompatibleClient(model="m", provider="local", api_key=None, base_url="http://x", temperature=0.7)
        fake_response = MagicMock()
        fake_response.choices[0].message.content = '{"verdict": "supported", "justification": "ok"}'
        create_mock = MagicMock(return_value=fake_response)
        client._client.chat.completions.create = create_mock

        client.complete_json(system_message="sys", user_prompt="user", schema=_Schema)

        assert create_mock.call_args.kwargs["temperature"] == 0.7

    def test_default_connect_timeout_is_more_generous_than_the_sdk_default(self):
        # o SDK da OpenAI usa por padrao connect=5.0s, curto demais para um
        # servidor local single-worker ainda ocupado com a chamada anterior
        # (bug real observado contra um LM Studio ocupado)
        client = OpenAICompatibleClient(model="m", provider="local", api_key=None, base_url="http://x")
        assert client._client._client.timeout.connect > 5.0

    def test_provider_exception_becomes_llm_provider_error(self):
        client = OpenAICompatibleClient(model="m", provider="local", api_key=None, base_url="http://x")
        client._client.chat.completions.create = MagicMock(side_effect=RuntimeError("conexao recusada"))

        with pytest.raises(LLMProviderError):
            client.complete_json(system_message="sys", user_prompt="user", schema=_Schema)

    def test_invalid_schema_becomes_llm_parse_error(self):
        client = OpenAICompatibleClient(model="m", provider="local", api_key=None, base_url="http://x")
        fake_response = MagicMock()
        fake_response.choices[0].message.content = '{"campo_errado": true}'
        client._client.chat.completions.create = MagicMock(return_value=fake_response)

        with pytest.raises(LLMParseError):
            client.complete_json(system_message="sys", user_prompt="user", schema=_Schema)

    def test_transient_provider_error_is_retried_and_eventually_succeeds(self):
        sleeps: list[float] = []
        client = OpenAICompatibleClient(
            model="m", provider="local", api_key=None, base_url="http://x",
            max_retries=3, retry_delay=1.5, sleep=sleeps.append,
        )
        fake_response = MagicMock()
        fake_response.choices[0].message.content = '{"verdict": "supported", "justification": "ok"}'
        client._client.chat.completions.create = MagicMock(
            side_effect=[RuntimeError("timeout"), RuntimeError("timeout"), fake_response]
        )

        instance, _usage = client.complete_json(system_message="sys", user_prompt="user", schema=_Schema)

        assert instance.verdict == "supported"
        assert sleeps == [1.5, 1.5]  # duas esperas entre as tres tentativas

    def test_provider_error_still_raises_after_exhausting_retries(self):
        sleeps: list[float] = []
        client = OpenAICompatibleClient(
            model="m", provider="local", api_key=None, base_url="http://x",
            max_retries=3, retry_delay=0.0, sleep=sleeps.append,
        )
        client._client.chat.completions.create = MagicMock(side_effect=RuntimeError("fora do ar"))

        with pytest.raises(LLMProviderError):
            client.complete_json(system_message="sys", user_prompt="user", schema=_Schema)

        assert client._client.chat.completions.create.call_count == 3
        assert sleeps == [0.0, 0.0]

    def test_parse_error_is_never_retried(self):
        sleeps: list[float] = []
        client = OpenAICompatibleClient(
            model="m", provider="local", api_key=None, base_url="http://x",
            max_retries=3, retry_delay=1.0, sleep=sleeps.append,
        )
        fake_response = MagicMock()
        fake_response.choices[0].message.content = '{"campo_errado": true}'
        client._client.chat.completions.create = MagicMock(return_value=fake_response)

        with pytest.raises(LLMParseError):
            client.complete_json(system_message="sys", user_prompt="user", schema=_Schema)

        assert client._client.chat.completions.create.call_count == 1
        assert sleeps == []

    def test_model_rejecting_temperature_retries_without_it(self):
        """Regressao: modelos de raciocinio recentes (ex: familia
        o1/o3/gpt-5) rejeitam qualquer `temperature` != default com um
        BadRequestError contendo "'temperature' does not support 0" --
        deve reconhecer isso pelo nome do parametro na mensagem de erro e
        retentar sem `temperature`, sem qualquer logica especifica de
        modelo/provider."""
        client = OpenAICompatibleClient(model="gpt-5.5", provider="openai", api_key="sk-test", base_url=None)
        fake_response = MagicMock()
        fake_response.choices[0].message.content = '{"verdict": "supported", "justification": "ok"}'
        create_mock = MagicMock(
            side_effect=[
                RuntimeError("Unsupported value: 'temperature' does not support 0 with this model. Only the default (1) value is supported."),
                fake_response,
            ]
        )
        client._client.chat.completions.create = create_mock

        instance, _usage = client.complete_json(system_message="sys", user_prompt="user", schema=_Schema)

        assert instance.verdict == "supported"
        assert create_mock.call_count == 2
        assert "temperature" not in create_mock.call_args_list[1].kwargs
        assert "response_format" in create_mock.call_args_list[1].kwargs

    def test_rejected_param_is_not_retried_on_subsequent_calls(self):
        """Regressao: um `LLMClient` e construido uma unica vez por run e
        reusado para todos os chunks (`build_pipeline`) — depois que um
        parametro e identificado como rejeitado, as chamadas seguintes
        (chunks seguintes) devem sair direto sem ele, sem repetir a
        chamada fadada a falhar a cada vez."""
        client = OpenAICompatibleClient(model="gpt-5.5", provider="openai", api_key="sk-test", base_url=None)
        fake_response = MagicMock()
        fake_response.choices[0].message.content = '{"verdict": "supported", "justification": "ok"}'
        create_mock = MagicMock(
            side_effect=[
                RuntimeError("Unsupported value: 'temperature' does not support 0 with this model."),
                fake_response,  # chunk 1: aprende que 'temperature' e rejeitado
                fake_response,  # chunk 2: deve ir direto, sem repetir o erro
            ]
        )
        client._client.chat.completions.create = create_mock

        client.complete_json(system_message="sys", user_prompt="chunk 1", schema=_Schema)
        client.complete_json(system_message="sys", user_prompt="chunk 2", schema=_Schema)

        assert create_mock.call_count == 3  # 2 do chunk 1 (falha + sucesso) + 1 do chunk 2
        assert "temperature" not in create_mock.call_args_list[2].kwargs

    def test_model_rejecting_both_response_format_and_temperature_retries_without_either(self):
        client = OpenAICompatibleClient(model="gpt-5.5", provider="openai", api_key="sk-test", base_url=None)
        fake_response = MagicMock()
        fake_response.choices[0].message.content = '{"verdict": "supported", "justification": "ok"}'
        create_mock = MagicMock(
            side_effect=[
                RuntimeError("Unrecognized request argument supplied: response_format"),
                RuntimeError("Unsupported value: 'temperature' does not support 0 with this model."),
                fake_response,
            ]
        )
        client._client.chat.completions.create = create_mock

        instance, _usage = client.complete_json(system_message="sys", user_prompt="user", schema=_Schema)

        assert instance.verdict == "supported"
        assert create_mock.call_count == 3
        assert "response_format" not in create_mock.call_args_list[2].kwargs
        assert "temperature" not in create_mock.call_args_list[2].kwargs

    def test_error_unrelated_to_any_optional_param_is_not_retried_as_fallback(self):
        client = OpenAICompatibleClient(model="m", provider="local", api_key=None, base_url="http://x")
        create_mock = MagicMock(side_effect=RuntimeError("modelo desconhecido"))
        client._client.chat.completions.create = create_mock

        with pytest.raises(LLMProviderError):
            client.complete_json(system_message="sys", user_prompt="user", schema=_Schema)

        assert create_mock.call_count == 1


class TestAnthropicClient:
    def test_complete_json_parses_response_and_computes_usage(self):
        client = AnthropicClient(model="claude-sonnet-5", api_key="sk-ant-test")
        fake_response = MagicMock()
        fake_response.parsed_output = _Schema(verdict="contradicted", justification="ok")
        fake_response.usage.input_tokens = 1_000_000
        fake_response.usage.output_tokens = 1_000_000
        client._client.messages.parse = MagicMock(return_value=fake_response)

        instance, usage = client.complete_json(system_message="sys", user_prompt="user", schema=_Schema)

        assert instance.verdict == "contradicted"
        assert usage.cost_usd == pytest.approx(3.00 + 15.00)

    def test_configured_temperature_is_forwarded_to_the_sdk_call(self):
        client = AnthropicClient(model="claude-sonnet-5", api_key="sk-ant-test", temperature=0.5)
        fake_response = MagicMock()
        fake_response.parsed_output = _Schema(verdict="supported", justification="ok")
        parse_mock = MagicMock(return_value=fake_response)
        client._client.messages.parse = parse_mock

        client.complete_json(system_message="sys", user_prompt="user", schema=_Schema)

        assert parse_mock.call_args.kwargs["temperature"] == 0.5

    def test_none_parsed_output_becomes_llm_parse_error(self):
        client = AnthropicClient(model="claude-sonnet-5", api_key="sk-ant-test")
        fake_response = MagicMock()
        fake_response.parsed_output = None
        client._client.messages.parse = MagicMock(return_value=fake_response)

        with pytest.raises(LLMParseError):
            client.complete_json(system_message="sys", user_prompt="user", schema=_Schema)

    def test_provider_exception_becomes_llm_provider_error(self):
        client = AnthropicClient(model="claude-sonnet-5", api_key="sk-ant-test")
        client._client.messages.parse = MagicMock(side_effect=RuntimeError("rate limited"))

        with pytest.raises(LLMProviderError):
            client.complete_json(system_message="sys", user_prompt="user", schema=_Schema)

    def test_transient_provider_error_is_retried_and_eventually_succeeds(self):
        sleeps: list[float] = []
        client = AnthropicClient(model="claude-sonnet-5", api_key="sk-ant-test", max_retries=3, retry_delay=2.0, sleep=sleeps.append)
        fake_response = MagicMock()
        fake_response.parsed_output = _Schema(verdict="supported", justification="ok")
        fake_response.usage.input_tokens = 10
        fake_response.usage.output_tokens = 5
        client._client.messages.parse = MagicMock(side_effect=[RuntimeError("rate limited"), fake_response])

        instance, _usage = client.complete_json(system_message="sys", user_prompt="user", schema=_Schema)

        assert instance.verdict == "supported"
        assert sleeps == [2.0]

    def test_parse_error_is_never_retried(self):
        sleeps: list[float] = []
        client = AnthropicClient(model="claude-sonnet-5", api_key="sk-ant-test", max_retries=3, retry_delay=1.0, sleep=sleeps.append)
        fake_response = MagicMock()
        fake_response.parsed_output = None
        client._client.messages.parse = MagicMock(return_value=fake_response)

        with pytest.raises(LLMParseError):
            client.complete_json(system_message="sys", user_prompt="user", schema=_Schema)

        assert client._client.messages.parse.call_count == 1
        assert sleeps == []

    def test_model_rejecting_temperature_retries_without_it(self):
        client = AnthropicClient(model="claude-futuro", api_key="sk-ant-test")
        fake_response = MagicMock()
        fake_response.parsed_output = _Schema(verdict="supported", justification="ok")
        parse_mock = MagicMock(
            side_effect=[
                RuntimeError("Unsupported value: 'temperature' does not support 0 with this model."),
                fake_response,
            ]
        )
        client._client.messages.parse = parse_mock

        instance, _usage = client.complete_json(system_message="sys", user_prompt="user", schema=_Schema)

        assert instance.verdict == "supported"
        assert parse_mock.call_count == 2
        assert "temperature" not in parse_mock.call_args_list[1].kwargs
