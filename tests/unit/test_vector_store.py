from pathlib import Path

import pytest

pytest.importorskip("faiss")

from auditframework.indexing.vector_store import FaissVectorStore
from auditframework.models import ReferenceChunk


def _chunk(embedding_id: int, reference_id: str, text: str = "texto") -> ReferenceChunk:
    return ReferenceChunk(
        id=f"{reference_id}-{embedding_id}",
        reference_id=reference_id,
        section=None,
        text=text,
        token_count=1,
        embedding_id=embedding_id,
    )


def test_search_returns_nearest_by_cosine_similarity():
    store = FaissVectorStore(dimension=3)
    chunks = [_chunk(0, "refA"), _chunk(1, "refB")]
    embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    store.add(chunks, embeddings)

    results = store.search([1.0, 0.0, 0.0], top_k=1)

    assert results[0][0].reference_id == "refA"


def test_scoped_search_only_returns_allowed_ids():
    store = FaissVectorStore(dimension=3)
    chunks = [_chunk(0, "refA"), _chunk(1, "refB")]
    embeddings = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    store.add(chunks, embeddings)

    # a query mais proxima globalmente e refB, mas escopamos para refA
    allowed = store.embedding_ids_for_references(["refA"])
    results = store.search([1.0, 0.0, 0.0], top_k=5, allowed_ids=allowed)

    assert len(results) == 1
    assert results[0][0].reference_id == "refA"


def test_embedding_ids_for_references_filters_by_reference():
    store = FaissVectorStore(dimension=2)
    chunks = [_chunk(0, "refA"), _chunk(1, "refA"), _chunk(2, "refB")]
    embeddings = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]
    store.add(chunks, embeddings)

    assert store.embedding_ids_for_references(["refA"]) == {0, 1}
    assert store.embedding_ids_for_references(["refB"]) == {2}
    assert store.embedding_ids_for_references(["refC"]) == set()


def test_save_and_load_roundtrip(tmp_path: Path):
    store = FaissVectorStore(dimension=3)
    chunks = [_chunk(0, "refA", text="conteudo A")]
    store.add(chunks, [[1.0, 0.0, 0.0]])

    index_path = tmp_path / "faiss.index"
    chunks_path = tmp_path / "chunks.json"
    store.save(index_path, chunks_path)

    reloaded = FaissVectorStore.load(dimension=3, index_path=index_path, chunks_path=chunks_path)
    results = reloaded.search([1.0, 0.0, 0.0], top_k=1)

    assert results[0][0].text == "conteudo A"
