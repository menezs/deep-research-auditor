from __future__ import annotations

from pathlib import Path

from ..models import ReferenceChunk


class Reranker:
    """Adapter sobre `sentence-transformers.CrossEncoder` (BGE-Reranker),
    usado para reordenar por precisao os candidatos recuperados pelo
    FAISS (que otimiza recall)."""

    def __init__(self, model_name: str, cache_folder: Path):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name, cache_folder=str(cache_folder))

    def rerank(
        self, query: str, candidates: list[tuple[ReferenceChunk, float]], top_k: int
    ) -> list[tuple[ReferenceChunk, float]]:
        if not candidates:
            return []
        pairs = [[query, chunk.text] for chunk, _ in candidates]
        scores = self._model.predict(pairs)
        reranked = sorted(zip((c for c, _ in candidates), scores), key=lambda item: item[1], reverse=True)
        return [(chunk, float(score)) for chunk, score in reranked[:top_k]]
