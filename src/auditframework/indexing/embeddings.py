from __future__ import annotations

from pathlib import Path
from typing import Protocol


class Embedder(Protocol):
    dimension: int

    def encode(self, texts: list[str]) -> list[list[float]]: ...


class BGEEmbedder:
    """Adapter sobre `sentence-transformers` para o modelo BGE-M3 —
    trocavel por qualquer outro `Embedder` sem alterar `Retriever`."""

    def __init__(self, model_name: str, cache_folder: Path):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, cache_folder=str(cache_folder))
        self.dimension = self._model.get_embedding_dimension()

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
