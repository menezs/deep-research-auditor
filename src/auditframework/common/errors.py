from __future__ import annotations


class AuditFrameworkError(Exception):
    """Base de toda excecao tipada do pipeline.

    Substitui o casamento de substring em texto de erro livre usado hoje
    no CorpusForge (`"429" in error_message`) para decidir o que e
    retryable — cada situacao de falha vira um tipo explicito."""


class ConfigurationError(AuditFrameworkError):
    """Configuracao invalida ou incompleta (ex: provider LLM selecionado
    sem a API key correspondente)."""


class ExtractionError(AuditFrameworkError):
    """Falha ao extrair referencias/citacoes de um arquivo de resposta."""


class FetchError(AuditFrameworkError):
    """Base para falhas ao baixar o conteudo de uma Reference."""


class DeadReferenceError(FetchError):
    """A referencia nao existe mais (ex: HTTP 404). Nao deve ser reprocessada
    automaticamente — precisa aparecer no relatorio como referencia morta."""


class InaccessibleReferenceError(FetchError):
    """A referencia existe mas nao pode ser acessada apos esgotar as
    estrategias de fetch e retries (403, timeout, SSL, rate limit
    persistente)."""


class LLMError(AuditFrameworkError):
    """Base para falhas de chamadas a um LLM — compartilhada por qualquer
    estagio que use `common.llm_client.LLMClient` (extraction e judging)."""


class LLMParseError(LLMError):
    """A saida do LLM nao pode ser parseada/validada contra o schema
    esperado. `Verifier.verify` deixa a excecao propagar; quem a captura
    e `JudgingStage` (pipeline.py), que pula o chunk com um aviso em vez
    de derrubar o run inteiro — o chunk fica pendente para uma proxima
    `audit resume`. Nunca vira uma coacao silenciosa para CONTRADICTED
    nem um AuditVerdict a parte."""


class LLMProviderError(LLMError):
    """A chamada ao provider de LLM falhou (rede, autenticacao, rate
    limit nao recuperado)."""
