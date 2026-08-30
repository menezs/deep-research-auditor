import pytest

pytest.importorskip("faiss")

from auditframework.indexing.retriever import Retriever
from auditframework.indexing.vector_store import FaissVectorStore
from auditframework.models import AnswerChunk, ReferenceChunk


class FakeEmbedder:
    def __init__(self, vectors: dict[str, list[float]]):
        self.dimension = len(next(iter(vectors.values())))
        self._vectors = vectors

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors[t] for t in texts]


def _chunk(embedding_id: int, reference_id: str, text: str) -> ReferenceChunk:
    return ReferenceChunk(
        id=f"{reference_id}-{embedding_id}",
        reference_id=reference_id,
        section=None,
        text=text,
        token_count=1,
        embedding_id=embedding_id,
    )


def _build_store() -> FaissVectorStore:
    """Corpus deliberadamente adversarial: a query do chunk que cita
    refA e, em termos de similaridade de cosseno pura, mais proxima do
    chunk de refB do que de qualquer chunk da propria refA — simulando o
    caso real em que dois temas parecidos (mas distintos) sao discutidos
    por referencias diferentes."""
    store = FaissVectorStore(dimension=3)
    chunks = [
        _chunk(0, "refA", "conteudo de A1"),
        _chunk(1, "refA", "conteudo de A2"),
        _chunk(2, "refB", "conteudo de B1"),
    ]
    embeddings = [
        [0.0, 1.0, 0.0],  # A1
        [0.0, 0.9, 0.1],  # A2
        [1.0, 0.0, 0.0],  # B1 - mais proximo da query abaixo do que qualquer chunk de A
    ]
    store.add(chunks, embeddings)
    return store


def test_retrieval_is_scoped_to_the_cited_reference_even_when_another_reference_scores_higher():
    store = _build_store()
    embedder = FakeEmbedder({"pergunta sobre A": [0.99, 0.01, 0.0]})
    retriever = Retriever(embedder, store, reranker=None, top_k=5, rerank_top_k=5)

    chunk = AnswerChunk(id="c1", answer_id="a1", position=0, text="pergunta sobre A", cited_reference_ids=["refA"])
    curated = retriever.retrieve(chunk)

    assert curated.skip_reason is None
    assert all(p.reference_id == "refA" for p in curated.passages)


def test_unscoped_search_would_have_returned_the_wrong_reference():
    """Sanity check da configuracao do teste acima: confirma que, sem
    escopo, o resultado top-1 seria de fato refB (provando que o teste
    anterior de fato exercita a correcao do bug, nao um cenario trivial)."""
    store = _build_store()
    results = store.search([0.99, 0.01, 0.0], top_k=1)
    assert results[0][0].reference_id == "refB"


def test_no_cited_reference_is_skipped_in_citation_scoped_mode():
    store = _build_store()
    embedder = FakeEmbedder({"pergunta sem citacao": [1.0, 0.0, 0.0]})
    retriever = Retriever(embedder, store, reranker=None, top_k=5, rerank_top_k=5)

    chunk = AnswerChunk(id="c1", answer_id="a1", position=0, text="pergunta sem citacao", cited_reference_ids=[])
    curated = retriever.retrieve(chunk)

    assert curated.skip_reason is not None
    assert curated.passages == []


def test_cited_reference_with_no_indexed_chunks_is_skipped_in_citation_scoped_mode():
    store = _build_store()
    embedder = FakeEmbedder({"pergunta sobre C": [1.0, 0.0, 0.0]})
    retriever = Retriever(embedder, store, reranker=None, top_k=5, rerank_top_k=5)

    # refC nao tem nenhum chunk indexado (ex: referencia morta/inacessivel)
    chunk = AnswerChunk(id="c1", answer_id="a1", position=0, text="pergunta sobre C", cited_reference_ids=["refC"])
    curated = retriever.retrieve(chunk)

    assert curated.skip_reason is not None
    assert "refC" in curated.skip_reason
    assert curated.passages == []


def test_retrieve_whole_corpus_ignores_citation_scope_and_never_skips():
    store = _build_store()
    embedder = FakeEmbedder({"pergunta sem citacao": [1.0, 0.0, 0.0]})
    retriever = Retriever(embedder, store, reranker=None, top_k=5, rerank_top_k=5)

    chunk = AnswerChunk(id="c1", answer_id="a1", position=0, text="pergunta sem citacao", cited_reference_ids=[])
    curated = retriever.retrieve_whole_corpus(chunk)

    assert curated.skip_reason is None
    assert curated.passages[0].reference_id == "refB"


def test_retrieve_whole_corpus_searches_everything_even_with_a_cited_reference():
    store = _build_store()
    embedder = FakeEmbedder({"pergunta sobre A": [0.99, 0.01, 0.0]})
    retriever = Retriever(embedder, store, reranker=None, top_k=5, rerank_top_k=5)

    chunk = AnswerChunk(id="c1", answer_id="a1", position=0, text="pergunta sobre A", cited_reference_ids=["refA"])
    curated = retriever.retrieve_whole_corpus(chunk)

    assert curated.skip_reason is None
    assert any(p.reference_id == "refB" for p in curated.passages)


def test_assembled_context_preserves_provenance_per_passage():
    store = _build_store()
    embedder = FakeEmbedder({"pergunta sobre A": [0.99, 0.01, 0.0]})
    retriever = Retriever(embedder, store, reranker=None, top_k=5, rerank_top_k=5)

    chunk = AnswerChunk(id="c1", answer_id="a1", position=0, text="pergunta sobre A", cited_reference_ids=["refA"])
    curated = retriever.retrieve(chunk)

    assert "refA" in curated.assembled_context
    assert "conteudo de A1" in curated.assembled_context or "conteudo de A2" in curated.assembled_context


class _StubReranker:
    """Mantém a ordem/score dos candidatos (identidade), só corta em top_k."""

    def rerank(self, query, candidates, top_k):
        return [(c, s) for c, s in candidates][:top_k]


def _doc_store() -> FaissVectorStore:
    store = FaissVectorStore(dimension=2)
    chunks = [
        _chunk(0, "refA", "intro de A"),
        _chunk(1, "refA", "o fato citado esta aqui"),
        _chunk(2, "refA", "continuacao do fato"),
        _chunk(3, "refA", "conclusao de A"),
        _chunk(4, "refB", "conteudo de B"),
    ]
    embeddings = [
        [0.2, 0.9],
        [1.0, 0.0],   # hit
        [0.1, 0.9],
        [0.0, 1.0],
        [0.9, 0.1],
    ]
    store.add(chunks, embeddings)
    return store


def test_retrieve_expanded_includes_neighbours_in_document_order():
    store = _doc_store()
    embedder = FakeEmbedder({"pergunta": [1.0, 0.0]})
    retriever = Retriever(embedder, store, reranker=_StubReranker(), top_k=10, rerank_top_k=1)

    chunk = AnswerChunk(id="c1", answer_id="a1", position=0, text="pergunta", cited_reference_ids=["refA"])
    curated = retriever.retrieve_expanded(chunk, neighbor_window=1, rerank_top_k=1)

    ids = [p.reference_chunk_id for p in curated.passages]
    # hit = refA-1; janela ±1 -> refA-0, refA-1, refA-2, em ordem de documento
    assert ids == ["refA-0", "refA-1", "refA-2"]
    assert all(p.reference_id == "refA" for p in curated.passages)
    assert "o fato citado esta aqui" in curated.assembled_context


def test_retrieve_expanded_skips_when_no_citation():
    store = _doc_store()
    embedder = FakeEmbedder({"pergunta": [1.0, 0.0]})
    retriever = Retriever(embedder, store, reranker=None, top_k=10, rerank_top_k=5)
    chunk = AnswerChunk(id="c1", answer_id="a1", position=0, text="pergunta", cited_reference_ids=[])

    assert retriever.retrieve_expanded(chunk).skip_reason is not None
