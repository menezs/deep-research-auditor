from __future__ import annotations

from ..logging_config import get_logger
from ..models import AnswerChunk, CuratedDocument, RetrievedPassage
from .embeddings import Embedder
from .reranker import Reranker
from .vector_store import VectorStore

logger = get_logger(__name__)


class Retriever:
    """Recupera, para cada AnswerChunk, apenas os trechos das referencias
    que ele de fato cita — corrige o bug central do syntex, em que a
    busca semantica ignorava `original_references` e buscava no indice
    inteiro, podendo devolver trechos de uma referencia diferente da
    citada pelo AnswerChunk.

    Quando a(s) referencia(s) citada(s) nao tem chunks indexados (porque
    a Reference esta DEAD/INACCESSIBLE, ou porque a extracao nao
    resolveu o marcador), a busca degrada para o corpus inteiro e isso e
    sinalizado explicitamente via `CuratedDocument.retrieval_degraded`,
    em vez de acontecer silenciosamente como acontece hoje."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        reranker: Reranker | None = None,
        top_k: int = 50,
        rerank_top_k: int = 20,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = reranker
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k

    def retrieve(self, chunk: AnswerChunk) -> CuratedDocument:
        query_embedding = self.embedder.encode([chunk.text])[0]

        allowed_ids = None
        degraded = False
        if chunk.cited_reference_ids:
            allowed_ids = self.vector_store.embedding_ids_for_references(chunk.cited_reference_ids)
            if not allowed_ids:
                logger.warning(
                    "Nenhum chunk indexado para as referencias citadas por %s (%s); "
                    "degradando para busca no corpus inteiro",
                    chunk.id,
                    chunk.cited_reference_ids,
                )
                allowed_ids = None
                degraded = True
        else:
            degraded = True

        results = self.vector_store.search(query_embedding, self.top_k, allowed_ids=allowed_ids)

        if self.reranker is not None:
            results = self.reranker.rerank(chunk.text, results, self.rerank_top_k)
        else:
            results = results[: self.rerank_top_k]

        passages = [
            RetrievedPassage(reference_chunk_id=rc.id, reference_id=rc.reference_id, score=score, text=rc.text)
            for rc, score in results
        ]
        return CuratedDocument(
            answer_chunk_id=chunk.id,
            passages=passages,
            assembled_context=_assemble_context(passages),
            retrieval_degraded=degraded,
        )


def _assemble_context(passages: list[RetrievedPassage]) -> str:
    """Monta o contexto textual entregue ao juiz LLM preservando
    proveniencia por trecho (score + referencia) — o `json_to_markdown.py`
    do syntex descartava score/referencia ao concatenar os trechos."""
    if not passages:
        return ""
    blocks = [f"[referencia={p.reference_id} score={p.score:.3f}]\n{p.text}" for p in passages]
    return "\n\n---\n\n".join(blocks)
