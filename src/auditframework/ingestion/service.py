from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from tqdm import tqdm

from ..common.errors import DeadReferenceError, ExtractionError, InaccessibleReferenceError
from ..logging_config import STAGE_COLORS, get_logger
from ..models import Document, Reference, ReferenceStatus
from .converters import convert_to_markdown
from .fetcher import FetchResult, HttpFetcher
from .registry import ReferenceRegistry

logger = get_logger(__name__)


class Fetcher(Protocol):
    def fetch(self, url: str) -> FetchResult: ...
    def fetch_via_playwright(self, url: str) -> FetchResult: ...


def ingest_references(
    references: list[Reference],
    registry: ReferenceRegistry,
    *,
    max_workers: int = 4,
    fetcher: Fetcher | None = None,
) -> tuple[list[Reference], list[Document]]:
    """Baixa e converte cada Reference para Markdown, atualizando seu
    status (DOWNLOADED/DEAD/INACCESSIBLE/ERROR).

    E idempotente: referencias cujo documento ja existe no registry sao
    puladas sem nova requisicao HTTP — substitui o rastreamento fragil de
    "ja baixados" por nome de arquivo usado hoje no CorpusForge.

    O `documents` retornado inclui TODOS os documentos atualmente
    disponiveis para as referencias informadas — os baixados nesta
    chamada e os que ja existiam no registry de uma execucao anterior —
    para que uma retomada (`audit resume`) possa reindexar sem precisar
    rebaixar nada."""
    fetcher = fetcher or HttpFetcher()
    updated: dict[str, Reference] = {ref.id: ref for ref in references}
    documents: list[Document] = []

    already_downloaded = [ref for ref in references if registry.has_document(ref.id)]
    pending = [ref for ref in references if not registry.has_document(ref.id)]
    logger.info(
        "Ingestao: %d referencias, %d ja baixadas, %d pendentes",
        len(references),
        len(already_downloaded),
        len(pending),
    )

    for ref in already_downloaded:
        # o status de `ref` reflete o que o chamador passou (ex: PENDING,
        # se veio direto da extracao) — aqui sabemos com certeza que o
        # documento existe, entao o status precisa refletir isso, ou a
        # proxima `save_references` sobrescreveria um DOWNLOADED anterior
        # com um status desatualizado.
        updated[ref.id] = ref.model_copy(update={"status": ReferenceStatus.DOWNLOADED})
    documents.extend(registry.load_document(ref.id) for ref in already_downloaded)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ref = {executor.submit(_ingest_one, ref, fetcher): ref for ref in pending}
        for future in tqdm(
            as_completed(future_to_ref),
            total=len(future_to_ref),
            desc="Baixando referências",
            unit="ref",
            colour=STAGE_COLORS["ingestion"],
        ):
            new_ref, document, markdown = future.result()
            updated[new_ref.id] = new_ref
            if document is not None and markdown is not None:
                registry.save_document(document, markdown)
                documents.append(document)

    final_references = list(updated.values())
    registry.save_references(final_references)
    return final_references, documents


def _ingest_one(
    reference: Reference, fetcher: Fetcher
) -> tuple[Reference, Document | None, str | None]:
    try:
        result, markdown = _fetch_and_convert(reference, fetcher)
    except DeadReferenceError as exc:
        logger.warning("Referencia morta: %s", reference.raw_url)
        return _with_status(reference, ReferenceStatus.DEAD, str(exc)), None, None
    except InaccessibleReferenceError as exc:
        logger.warning("Referencia inacessivel: %s", reference.raw_url)
        return _with_status(reference, ReferenceStatus.INACCESSIBLE, str(exc)), None, None
    except ExtractionError as exc:
        logger.warning("Falha ao converter %s: %s", reference.raw_url, exc)
        return _with_status(reference, ReferenceStatus.ERROR, str(exc)), None, None
    except Exception as exc:  # defensivo: fronteira de rede/parsing de terceiros
        logger.exception("Erro inesperado ao ingerir %s", reference.raw_url)
        return _with_status(reference, ReferenceStatus.ERROR, str(exc)), None, None

    updated_ref = reference.model_copy(
        update={
            "status": ReferenceStatus.DOWNLOADED,
            "http_status": result.http_status,
            "fetched_at": datetime.now(timezone.utc),
            "error_message": None,
        }
    )
    document = Document(
        reference_id=reference.id,
        markdown_path=Path("documents") / f"{reference.id}.md",
        content_hash=hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
        fetch_method=result.fetch_method,
        word_count=len(markdown.split()),
    )
    return updated_ref, document, markdown


def _fetch_and_convert(reference: Reference, fetcher: Fetcher) -> tuple[FetchResult, str]:
    result = fetcher.fetch(reference.raw_url)
    try:
        markdown = convert_to_markdown(result.content, result.content_type, reference.raw_url)
    except ExtractionError:
        if result.fetch_method == "playwright":
            raise
        result = fetcher.fetch_via_playwright(reference.raw_url)
        markdown = convert_to_markdown(result.content, result.content_type, reference.raw_url)
    return result, markdown


def _with_status(reference: Reference, status: ReferenceStatus, error_message: str) -> Reference:
    return reference.model_copy(
        update={
            "status": status,
            "fetched_at": datetime.now(timezone.utc),
            "error_message": error_message,
        }
    )
