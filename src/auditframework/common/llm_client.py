from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Callable, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from ..logging_config import get_logger
from .errors import ConfigurationError, LLMParseError, LLMProviderError
from .pricing import cost_usd

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class LLMUsage:
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    cost_usd: float


class LLMClient(Protocol):
    """Contrato compartilhado por qualquer estagio que precise de um LLM
    (extraction, para respostas cuja lista de fontes nao segue o formato
    regex-friendly, e judging, para o veredito). Substitui as duas
    implementacoes de "LLM service" hoje duplicadas e incompativeis entre
    si no CorpusForge e no audit_with_llm."""

    model: str

    def complete_json(self, *, system_message: str, user_prompt: str, schema: type[T]) -> tuple[T, LLMUsage]: ...


def parse_json_object(raw_text: str) -> dict:
    """Extrai um objeto JSON de uma resposta de LLM, tolerando fences de
    markdown (```json ... ```) ou texto extra ao redor do JSON — mais
    robusto que o `JsonOutputParser` do LangChain (fonte de falhas reais
    documentadas no audit_with_llm)."""
    text = raw_text.strip()
    for candidate in (text, _extract_fenced(text), _extract_bare_object(text)):
        if candidate is None:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise LLMParseError(f"resposta do LLM nao e JSON valido: {raw_text!r}")


def _extract_fenced(text: str) -> str | None:
    match = _JSON_FENCE_RE.search(text)
    return match.group(1) if match else None


def _extract_bare_object(text: str) -> str | None:
    match = _BARE_OBJECT_RE.search(text)
    return match.group(0) if match else None


def _json_schema_response_format(schema: type[BaseModel]) -> dict:
    json_schema = dict(schema.model_json_schema())
    json_schema["additionalProperties"] = False
    json_schema["required"] = list(json_schema.get("properties", {}).keys())
    return {
        "type": "json_schema",
        "json_schema": {"name": schema.__name__, "schema": json_schema, "strict": True},
    }


_OPTIONAL_CHAT_PARAMS: tuple[str, ...] = ("response_format", "temperature")


class OpenAICompatibleClient:
    """Adapter para qualquer endpoint compativel com a API de chat da
    OpenAI — cobre tanto servidores locais (LM Studio/Ollama expondo
    `/v1/chat/completions`) quanto a OpenAI real, via `base_url`.

    Servidores e modelos variam no que aceitam como parametro opcional:
    servidores locais (ex: LM Studio) as vezes rejeitam `response_format`
    inteiramente; modelos de raciocinio mais recentes (ex: familia
    o1/o3/gpt-5) rejeitam qualquer `temperature` diferente do default
    (`"Unsupported value: 'temperature' does not support 0..."`). Em vez
    de tratar cada caso/modelo especificamente, `_create_chat_completion`
    tenta a chamada completa e, se o provider rejeitar um parametro
    opcional especifico (identificado pelo nome dele aparecendo na
    mensagem de erro), remove so esse parametro e tenta de novo — ate
    esgotar os parametros opcionais conhecidos (`_OPTIONAL_CHAT_PARAMS`).
    Sem `response_format`, `parse_json_object` ja tolera fences de
    markdown e texto ao redor do JSON na resposta em texto livre.

    Um parametro rejeitado uma vez fica em `self._rejected_params` pelo
    resto da vida desta instancia (um `LLMClient` e construido uma unica
    vez por run em `build_pipeline()` e reusado para todos os chunks) —
    as chamadas seguintes ja saem sem ele, em vez de repetir a mesma
    chamada fadada a falhar (e o retry correspondente) a cada chunk.

    Falhas de rede/provider (`LLMProviderError`) sao retentadas ate
    `max_retries` vezes com espera de `retry_delay` segundos entre
    tentativas — cobre blips transitorios (timeout, conexao recusada
    momentaneamente). `LLMParseError` (saida que nao bate com o schema)
    nunca e retentada aqui: com `temperature` baixa repetir a mesma
    chamada tende a repetir o mesmo erro, e o chamador (`Verifier`) deixa
    essa excecao propagar em vez de coagi-la para um veredito de
    conteudo. Se o provider continuar indisponivel apos esgotar as
    tentativas, o erro tambem se propaga (fail fast) em vez de coagir todo
    chunk restante para um veredito — `audit resume` retoma exatamente dos
    chunks ainda nao julgados.

    O SDK da OpenAI usa por padrao um timeout de CONEXAO de apenas 5s
    (`httpx.Timeout(connect=5.0, read=600, ...)`) — curto demais para um
    servidor local single-worker (LM Studio/Ollama) que pode estar ainda
    processando a chamada anterior quando a proxima conexao e aberta,
    fazendo o handshake TCP/HTTP estourar o timeout antes mesmo da
    geracao comecar (bug real observado em producao contra um servidor
    LM Studio ocupado). Por isso o timeout aqui e uniforme e generoso."""

    def __init__(
        self,
        *,
        model: str,
        provider: str,
        api_key: str | None,
        base_url: str | None,
        temperature: float = 0.0,
        max_retries: int = 1,
        retry_delay: float = 0.0,
        timeout: float = 600.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key or "not-needed", base_url=base_url, timeout=timeout)
        self.model = model
        self._provider = provider
        self.temperature = temperature
        self.max_retries = max(1, max_retries)
        self.retry_delay = retry_delay
        self._sleep = sleep
        self._rejected_params: set[str] = set()

    def complete_json(self, *, system_message: str, user_prompt: str, schema: type[T]) -> tuple[T, LLMUsage]:
        attempt = 0
        while True:
            attempt += 1
            try:
                return self._complete_once(system_message, user_prompt, schema)
            except LLMProviderError:
                if attempt >= self.max_retries:
                    raise
                logger.warning(
                    "Chamada ao provider LLM falhou (tentativa %d/%d); tentando novamente em %.1fs",
                    attempt,
                    self.max_retries,
                    self.retry_delay,
                )
                self._sleep(self.retry_delay)

    def _create_chat_completion(self, kwargs: dict):
        remaining = {k: v for k, v in kwargs.items() if k not in self._rejected_params}
        droppable = [p for p in _OPTIONAL_CHAT_PARAMS if p in remaining]
        while True:
            try:
                return self._client.chat.completions.create(**remaining)
            except Exception as exc:  # biblioteca externa: superficie de erro ampla
                error_text = str(exc)
                rejected = next((p for p in droppable if p in error_text), None)
                if rejected is None:
                    raise LLMProviderError(error_text) from exc
                logger.info(
                    "Provider rejeitou o parametro '%s' para o modelo %s; sera omitido nas "
                    "proximas chamadas desta execucao: %s",
                    rejected,
                    self.model,
                    error_text,
                )
                self._rejected_params.add(rejected)
                droppable.remove(rejected)
                del remaining[rejected]

    def _complete_once(self, system_message: str, user_prompt: str, schema: type[T]) -> tuple[T, LLMUsage]:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ]
        kwargs: dict = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": _json_schema_response_format(schema),
            "messages": messages,
        }
        start = time.monotonic()
        response = self._create_chat_completion(kwargs)
        latency_ms = int((time.monotonic() - start) * 1000)

        raw_text = response.choices[0].message.content or ""
        parsed = parse_json_object(raw_text)
        try:
            instance = schema.model_validate(parsed)
        except ValidationError as exc:
            raise LLMParseError(f"saida nao corresponde ao schema esperado: {raw_text!r}") from exc

        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        return instance, LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd(self._provider, self.model, prompt_tokens, completion_tokens),
        )


_OPTIONAL_MESSAGE_PARAMS: tuple[str, ...] = ("temperature",)


class AnthropicClient:
    """Adapter para a API da Anthropic, usando `messages.parse()` com
    `output_format` para saida estruturada nativa — mais confiavel que
    JSON mode baseado em texto, sem depender do provider suportar
    `response_format` corretamente. Assim como `OpenAICompatibleClient`,
    se o modelo rejeitar `temperature` diferente do default (comportamento
    de alguns modelos de raciocinio), `_create_message` remove esse
    parametro e tenta de novo em vez de falhar direto — e memoriza isso
    em `self._rejected_params` pelo resto da execucao, para nao repetir a
    mesma chamada fadada a falhar a cada chunk.

    Assim como `OpenAICompatibleClient`, retenta `LLMProviderError` ate
    `max_retries` vezes (blips transitorios); nao retenta `LLMParseError`
    nem coage uma falha persistente do provider para um veredito de
    conteudo chunk a chunk — ela se propaga, e `audit resume` retoma de
    onde parou."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        max_retries: int = 1,
        retry_delay: float = 0.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self._max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max(1, max_retries)
        self.retry_delay = retry_delay
        self._sleep = sleep
        self._rejected_params: set[str] = set()

    def complete_json(self, *, system_message: str, user_prompt: str, schema: type[T]) -> tuple[T, LLMUsage]:
        attempt = 0
        while True:
            attempt += 1
            try:
                return self._complete_once(system_message, user_prompt, schema)
            except LLMProviderError:
                if attempt >= self.max_retries:
                    raise
                logger.warning(
                    "Chamada ao provider LLM falhou (tentativa %d/%d); tentando novamente em %.1fs",
                    attempt,
                    self.max_retries,
                    self.retry_delay,
                )
                self._sleep(self.retry_delay)

    def _create_message(self, kwargs: dict):
        remaining = {k: v for k, v in kwargs.items() if k not in self._rejected_params}
        droppable = [p for p in _OPTIONAL_MESSAGE_PARAMS if p in remaining]
        while True:
            try:
                return self._client.messages.parse(**remaining)
            except Exception as exc:  # biblioteca externa: superficie de erro ampla
                error_text = str(exc)
                rejected = next((p for p in droppable if p in error_text), None)
                if rejected is None:
                    raise LLMProviderError(error_text) from exc
                logger.info(
                    "Provider rejeitou o parametro '%s' para o modelo %s; sera omitido nas "
                    "proximas chamadas desta execucao: %s",
                    rejected,
                    self.model,
                    error_text,
                )
                self._rejected_params.add(rejected)
                droppable.remove(rejected)
                del remaining[rejected]

    def _complete_once(self, system_message: str, user_prompt: str, schema: type[T]) -> tuple[T, LLMUsage]:
        start = time.monotonic()
        response = self._create_message(
            {
                "model": self.model,
                "max_tokens": self._max_tokens,
                "temperature": self.temperature,
                "system": system_message,
                "messages": [{"role": "user", "content": user_prompt}],
                "output_format": schema,
            }
        )
        latency_ms = int((time.monotonic() - start) * 1000)

        if response.parsed_output is None:
            raise LLMParseError("Claude nao retornou saida estruturada valida para o schema esperado")

        usage = response.usage
        prompt_tokens = usage.input_tokens
        completion_tokens = usage.output_tokens
        return response.parsed_output, LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd("anthropic", self.model, prompt_tokens, completion_tokens),
        )


def create_llm_client(settings) -> LLMClient:  # settings: auditframework.config.Settings
    """Factory que resolve o adapter correto a partir do `Settings`."""
    provider = settings.llm_provider
    if provider in ("local", "ollama", "openai"):
        return OpenAICompatibleClient(
            model=settings.llm_model,
            provider=provider,
            api_key=settings.openai_api_key,
            base_url=settings.llm_base_url,
            temperature=settings.llm_temperature,
            max_retries=settings.llm_max_retries,
            retry_delay=settings.llm_retry_delay,
        )
    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ConfigurationError("llm_provider=anthropic exige ANTHROPIC_API_KEY configurada")
        return AnthropicClient(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
            temperature=settings.llm_temperature,
            max_retries=settings.llm_max_retries,
            retry_delay=settings.llm_retry_delay,
        )
    raise ConfigurationError(f"llm_provider desconhecido: {provider!r}")
