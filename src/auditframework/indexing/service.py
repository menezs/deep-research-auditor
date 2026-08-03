from __future__ import annotations

from typing import Callable

from tqdm import tqdm

from ..logging_config import STAGE_COLORS, get_logger
from ..models import Document, ReferenceChunk
from .chunkers import DocumentChunker
from .embeddings import Embedder
from .vector_store import FaissVectorStore

logger = get_logger(__name__)


def build_index(
    documents: list[Document],
    read_markdown: Callable[[str], str],
    embedder: Embedder,
    chunker: DocumentChunker | None = None,
) -> FaissVectorStore:
    """Divide cada Document em ReferenceChunks, gera embeddings em lote e
    monta o FaissVectorStore. `read_markdown(reference_id) -> str` evita
    acoplar este servico a um layout de disco especifico (ex: ao
    `ReferenceRegistry` da Fase 1)."""
    chunker = chunker or DocumentChunker()
    store = FaissVectorStore(embedder.dimension)

    all_chunks: list[ReferenceChunk] = []
    next_id = 0
    for document in tqdm(documents, desc="Extraindo chunks de documentos", unit="doc", colour=STAGE_COLORS["indexing"]):
        markdown = read_markdown(document.reference_id)
        chunks = chunker.chunk(reference_id=document.reference_id, markdown=markdown, start_id=next_id)
        next_id += len(chunks)
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.warning("Nenhum chunk gerado a partir de %d documentos", len(documents))
        return store

    logger.info("Gerando embeddings para %d chunks de %d documentos", len(all_chunks), len(documents))
    embeddings = embedder.encode([chunk.text for chunk in all_chunks])
    store.add(all_chunks, embeddings)
    return store
