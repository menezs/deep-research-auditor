from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuracao unica do pipeline, substituindo os tres esquemas de
    .env incompativeis hoje existentes em CorpusForge/syntex/audit_with_llm."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Diretorios
    data_dir: Path = Path("./data")

    # Logging
    log_level: str = "INFO"
    log_format: Literal["text", "json"] = "text"

    # LLM (extracao de referencias + juiz)
    llm_provider: Literal["local", "openai", "anthropic", "ollama"] = "local"
    llm_base_url: str | None = "http://localhost:1234/v1/"
    llm_model: str = "openai/gpt-oss-20b"
    llm_temperature: float = 0.0
    llm_max_retries: int = 3
    llm_retry_delay: float = 2.0
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    # Embeddings / reranking / vetor store
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    model_cache_dir: Path = Path("./model_cache")
    retrieval_top_k: int = 50
    rerank_top_k: int = 20

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id


def get_settings() -> Settings:
    """Ponto unico de construcao de Settings, para permitir troca por
    injecao de dependencia (ex: em testes) sem tocar nos consumidores."""
    return Settings()
