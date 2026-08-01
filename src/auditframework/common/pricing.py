from __future__ import annotations

# USD por 1 milhao de tokens. Fonte: platform.claude.com/docs/en/pricing
# (consultado em 2026-07-29) — revisar periodicamente, precos mudam.
_ANTHROPIC_PRICING_PER_MILLION: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

_LOCAL_PROVIDERS = {"local", "ollama"}


def cost_usd(provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estima o custo em USD de uma chamada de LLM.

    Provedores locais (LM Studio/Ollama) nao tem custo por token. Para
    OpenAI/outros nao ha tabela embutida aqui (precos mudam com frequencia
    e nao sao o foco deste framework) — retorna 0.0 nesse caso, deixando
    claro no relatorio que o custo daquele provider nao foi contabilizado,
    em vez de estimar um valor potencialmente errado."""
    if provider in _LOCAL_PROVIDERS:
        return 0.0
    if provider == "anthropic":
        pricing = _ANTHROPIC_PRICING_PER_MILLION.get(model)
        if pricing is None:
            return 0.0
        input_price, output_price = pricing
        return (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000
    return 0.0
