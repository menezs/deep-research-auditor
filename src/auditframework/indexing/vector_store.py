from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np

from ..models import ReferenceChunk


class VectorStore(Protocol):
    def add(self, chunks: list[ReferenceChunk], embeddings: list[list[float]]) -> None: ...
    def search(
        self, query_embedding: list[float], top_k: int, *, allowed_ids: set[int] | None = None
    ) -> list[tuple[ReferenceChunk, float]]: ...
    def embedding_ids_for_references(self, reference_ids: list[str]) -> set[int]: ...
    def save(self, index_path: Path, chunks_path: Path) -> None: ...


class FaissVectorStore:
    """Adapter sobre `faiss.IndexFlatIP`, usando `IndexIDMap` para que o
    id de cada vetor seja o `ReferenceChunk.embedding_id` explicito, e
    nao a posicao de insercao implicita — o syntex acoplava a ordem
    interna do FAISS ao id do SQLite, o que quebraria silenciosamente em
    qualquer insercao fora de ordem ou seletiva (como a busca escopada
    por referencia que este `Retriever` precisa fazer).

    `IndexIDMap.reconstruct` nao e suportado pelo binding do faiss-cpu
    para este tipo de indice (`reconstruct not implemented`), entao a
    busca escopada reconstrói os vetores via o indice `Flat` interno
    (`self.index.index`), usando um mapa id->posicao mantido a parte."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))
        self._chunks: dict[int, ReferenceChunk] = {}
        self._position_of_id: dict[int, int] = {}

    def add(self, chunks: list[ReferenceChunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        ids = np.array([c.embedding_id for c in chunks], dtype="int64")
        vectors = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(vectors)

        start_position = self.index.ntotal
        self.index.add_with_ids(vectors, ids)
        for offset, chunk in enumerate(chunks):
            self._chunks[chunk.embedding_id] = chunk
            self._position_of_id[chunk.embedding_id] = start_position + offset

    def embedding_ids_for_references(self, reference_ids: list[str]) -> set[int]:
        wanted = set(reference_ids)
        return {eid for eid, chunk in self._chunks.items() if chunk.reference_id in wanted}

    def search(
        self, query_embedding: list[float], top_k: int, *, allowed_ids: set[int] | None = None
    ) -> list[tuple[ReferenceChunk, float]]:
        query = np.array([query_embedding], dtype="float32")
        faiss.normalize_L2(query)

        if allowed_ids is not None:
            return self._search_scoped(query, top_k, allowed_ids)

        distances, ids = self.index.search(query, top_k)
        return self._collect(distances[0], ids[0])

    def _search_scoped(
        self, query: np.ndarray, top_k: int, allowed_ids: set[int]
    ) -> list[tuple[ReferenceChunk, float]]:
        """Busca exata restrita a um subconjunto de chunks (tipicamente
        os de uma unica Reference citada). Como `IndexFlatIP` ja e busca
        exata por forca bruta, escopar a busca e simplesmente reconstruir
        os vetores do subconjunto (sem duplicar armazenamento de
        embeddings) e buscar num indice efemero contendo so esse
        subconjunto."""
        id_list = sorted(allowed_ids)
        if not id_list:
            return []
        vectors = np.array(
            [self.index.index.reconstruct(self._position_of_id[i]) for i in id_list], dtype="float32"
        )
        sub_index = faiss.IndexFlatIP(self.dimension)
        sub_index.add(vectors)
        distances, positions = sub_index.search(query, min(top_k, len(id_list)))
        results = []
        for score, pos in zip(distances[0], positions[0]):
            if pos == -1:
                continue
            results.append((self._chunks[id_list[int(pos)]], float(score)))
        return results

    def _collect(self, distances: np.ndarray, ids: np.ndarray) -> list[tuple[ReferenceChunk, float]]:
        results = []
        for score, id_ in zip(distances, ids):
            if id_ == -1:
                continue
            chunk = self._chunks.get(int(id_))
            if chunk is not None:
                results.append((chunk, float(score)))
        return results

    def save(self, index_path: Path, chunks_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        payload = [json.loads(chunk.model_dump_json()) for chunk in self._chunks.values()]
        chunks_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, dimension: int, index_path: Path, chunks_path: Path) -> "FaissVectorStore":
        store = cls(dimension)
        store.index = faiss.read_index(str(index_path))
        ids = faiss.vector_to_array(store.index.id_map)
        store._position_of_id = {int(id_): position for position, id_ in enumerate(ids)}
        for item in json.loads(chunks_path.read_text(encoding="utf-8")):
            chunk = ReferenceChunk.model_validate(item)
            store._chunks[chunk.embedding_id] = chunk
        return store
