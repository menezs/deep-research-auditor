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

    Por padrao (`full_corpus_mode=False`), a busca e escopada as
    referencias citadas pelo chunk. Quando o chunk nao cita nenhuma
    referencia, ou quando a(s) referencia(s) citada(s) nao tem chunks
    indexados (porque a Reference esta DEAD/INACCESSIBLE, ou porque a
    extracao nao resolveu o marcador), o chunk nao pode ser auditado com
    integridade nesse modo — em vez de degradar silenciosamente para o
    corpus inteiro, `retrieve` sinaliza isso via `CuratedDocument.skip_reason`
    para que o chamador pule o julgamento desse chunk.

    Com `full_corpus_mode=True`, a citacao e ignorada e a busca sempre usa
    o corpus inteiro — reflete o fato de que um Deep Research tipicamente
    usa todo o conhecimento encontrado, nao so o que citou explicitamente."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        reranker: Reranker | None = None,
        top_k: int = 50,
        rerank_top_k: int = 20,
        full_corpus_mode: bool = False,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = reranker
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.full_corpus_mode = full_corpus_mode

    def retrieve(self, chunk: AnswerChunk) -> CuratedDocument:
        query_embedding = self.embedder.encode([chunk.text])[0]

        allowed_ids = None
        skip_reason = None
        if not self.full_corpus_mode:
            if not chunk.cited_reference_ids:
                skip_reason = (
                    "Chunk nao cita nenhuma referencia; modo de recuperacao atual e escopado por citacao."
                )
            else:
                allowed_ids = self.vector_store.embedding_ids_for_references(chunk.cited_reference_ids)
                if not allowed_ids:
                    skip_reason = (
                        f"Referencia(s) citada(s) {chunk.cited_reference_ids} nao possui(em) conteudo "
                        "indexado (nao baixada(s)/inacessivel(is)) e o modo de recuperacao atual e "
                        "escopado por citacao."
                    )

        if skip_reason is not None:
            logger.warning("Chunk %s nao sera auditado: %s", chunk.id, skip_reason)
            return CuratedDocument(answer_chunk_id=chunk.id, assembled_context="", skip_reason=skip_reason)

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
        )


def _assemble_context(passages: list[RetrievedPassage]) -> str:
    """Monta o contexto textual entregue ao juiz LLM preservando
    proveniencia por trecho (score + referencia) — o `json_to_markdown.py`
    do syntex descartava score/referencia ao concatenar os trechos."""
    if not passages:
        return ""
    blocks = [f"[referencia={p.reference_id} score={p.score:.3f}]\n{p.text}" for p in passages]
    return "\n\n---\n\n".join(blocks)
