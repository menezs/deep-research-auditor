from __future__ import annotations

from ..logging_config import get_logger
from ..models import AnswerChunk, CuratedDocument, ReferenceChunk, RetrievedPassage
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

    `retrieve` e SEMPRE escopado as referencias citadas pelo chunk. Quando
    o chunk nao cita nenhuma referencia, ou quando a(s) referencia(s)
    citada(s) nao tem chunks indexados (porque a Reference esta
    DEAD/INACCESSIBLE, ou porque a extracao nao resolveu o marcador), o
    chunk nao pode ser auditado com integridade — em vez de degradar
    silenciosamente para o corpus inteiro, `retrieve` sinaliza isso via
    `CuratedDocument.skip_reason` para que o chamador pule o julgamento
    desse chunk.

    A busca no corpus inteiro (ignorando a citacao) existe apenas em
    `retrieve_whole_corpus`, usada exclusivamente pela Etapa C da
    verificacao (`judging/verification/cross_reference.py`) para checar se
    um claim marcado UNSUPPORTED pela fonte citada e sustentado/contradito
    por outra referencia baixada — nunca para reclassificar para
    SUPPORTED."""

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
        if not chunk.cited_reference_ids:
            return self._skipped(chunk, "Chunk nao cita nenhuma referencia.")
        allowed_ids = self.vector_store.embedding_ids_for_references(chunk.cited_reference_ids)
        if not allowed_ids:
            return self._skipped(
                chunk,
                f"Referencia(s) citada(s) {chunk.cited_reference_ids} nao possui(em) conteudo indexado "
                "(nao baixada(s)/inacessivel(is)).",
            )
        return self._search_and_assemble(chunk, allowed_ids)

    def retrieve_whole_corpus(self, chunk: AnswerChunk) -> CuratedDocument:
        """Recupera trechos de TODO o corpus indexado, ignorando quais
        referencias o chunk cita. Uso restrito a Etapa C da verificacao."""
        return self._search_and_assemble(chunk, allowed_ids=None)

    def _search_and_assemble(self, chunk: AnswerChunk, allowed_ids: set[int] | None) -> CuratedDocument:
        query_embedding = self.embedder.encode([chunk.text])[0]
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

    def _skipped(self, chunk: AnswerChunk, reason: str) -> CuratedDocument:
        logger.warning("Chunk %s nao sera auditado: %s", chunk.id, reason)
        return CuratedDocument(answer_chunk_id=chunk.id, assembled_context="", skip_reason=reason)


    def retrieve_expanded(
        self, chunk: AnswerChunk, *, neighbor_window: int = 1, rerank_top_k: int | None = None
    ) -> CuratedDocument:
        """Como `retrieve` (escopado pelas referencias citadas), mas:
        (1) reordena com um `rerank_top_k` maior e (2) expande cada trecho
        recuperado com os
        `neighbor_window` trechos vizinhos da mesma referencia, remontando
        o contexto em ORDEM DE DOCUMENTO.

        Etapa A da verificacao de um veredito UNSUPPORTED: recupera o fato
        que caiu na fronteira de um chunk de 512 tokens ou logo fora da
        janela de rerank."""
        if not chunk.cited_reference_ids:
            return CuratedDocument(
                answer_chunk_id=chunk.id, assembled_context="", skip_reason="Chunk nao cita nenhuma referencia."
            )
        allowed_ids = self.vector_store.embedding_ids_for_references(chunk.cited_reference_ids)
        if not allowed_ids:
            return CuratedDocument(
                answer_chunk_id=chunk.id,
                assembled_context="",
                skip_reason="Referencia(s) citada(s) sem conteudo indexado.",
            )

        query_embedding = self.embedder.encode([chunk.text])[0]
        results = self.vector_store.search(query_embedding, max(self.top_k, len(allowed_ids)), allowed_ids=allowed_ids)
        limit = rerank_top_k or self.rerank_top_k
        if self.reranker is not None:
            results = self.reranker.rerank(chunk.text, results, limit)
        else:
            results = results[:limit]

        score_by_id = {rc.id: score for rc, score in results}
        expanded = self._expand_with_neighbors([rc for rc, _ in results], neighbor_window)
        passages = [
            RetrievedPassage(
                reference_chunk_id=rc.id,
                reference_id=rc.reference_id,
                score=score_by_id.get(rc.id, 0.0),
                text=rc.text,
            )
            for rc in expanded
        ]
        return CuratedDocument(
            answer_chunk_id=chunk.id,
            passages=passages,
            assembled_context=_assemble_context_in_document_order(passages),
        )

    def _expand_with_neighbors(self, hits: list[ReferenceChunk], window: int) -> list[ReferenceChunk]:
        ref_order = list(dict.fromkeys(h.reference_id for h in hits))
        by_ref = {ref_id: self.vector_store.reference_chunks(ref_id) for ref_id in ref_order}
        wanted_ids: set[str] = set()
        for hit in hits:
            siblings = by_ref[hit.reference_id]
            pos = next((i for i, c in enumerate(siblings) if c.id == hit.id), None)
            if pos is None:
                wanted_ids.add(hit.id)
                continue
            for i in range(max(0, pos - window), min(len(siblings), pos + window + 1)):
                wanted_ids.add(siblings[i].id)
        return [c for ref_id in ref_order for c in by_ref[ref_id] if c.id in wanted_ids]


def _assemble_context(passages: list[RetrievedPassage]) -> str:
    """Monta o contexto textual entregue ao juiz LLM preservando
    proveniencia por trecho (score + referencia) — o `json_to_markdown.py`
    do syntex descartava score/referencia ao concatenar os trechos."""
    if not passages:
        return ""
    blocks = [f"[referencia={p.reference_id} score={p.score:.3f}]\n{p.text}" for p in passages]
    return "\n\n---\n\n".join(blocks)


def _assemble_context_in_document_order(passages: list[RetrievedPassage]) -> str:
    """Variante usada pela expansao de contexto (Etapa A): os trechos ja
    vem em ordem de documento (nao por score), entao o cabecalho marca so
    a proveniencia, sem sugerir um ranking."""
    if not passages:
        return ""
    blocks = [f"[referencia={p.reference_id} trecho={p.reference_chunk_id}]\n{p.text}" for p in passages]
    return "\n\n---\n\n".join(blocks)
