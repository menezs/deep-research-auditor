from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from tqdm import tqdm

from ..common.errors import DeadReferenceError, ExtractionError, InaccessibleReferenceError
from ..logging_config import STAGE_COLORS, get_logger
from ..models import Document, Reference, ReferenceStatus
from .converters import convert_to_markdown, is_pdf_content
from .fetcher import FetchResult, HttpFetcher
from .registry import ReferenceRegistry

logger = get_logger(__name__)

# Teto de tempo por referencia — rede de seguranca contra qualquer hang
# (DNS travado, playwright preso, host inesperado). O caso comum ja e
# resolvido pelos timeouts curtos do HttpFetcher; isto e o backstop.
_REF_BUDGET_SECONDS = 90.0


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
    fetcher_was_provided = fetcher is not None
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
            list(future_to_ref),
            total=len(future_to_ref),
            desc="Baixando referências",
            unit="ref",
            colour=STAGE_COLORS["ingestion"],
        ):
            ref = future_to_ref[future]
            try:
                new_ref, document, markdown = future.result(timeout=_REF_BUDGET_SECONDS)
            except FuturesTimeoutError:
                future.cancel()
                logger.warning(
                    "Referencia %s nao concluiu o download em %.0fs — marcada inacessivel",
                    ref.raw_url,
                    _REF_BUDGET_SECONDS,
                )
                new_ref, document, markdown = (
                    _with_status(
                        ref, ReferenceStatus.INACCESSIBLE, f"download excedeu {_REF_BUDGET_SECONDS:.0f}s"
                    ),
                    None,
                    None,
                )
            updated[new_ref.id] = new_ref
            if document is not None and markdown is not None:
                registry.save_document(document, markdown)
                documents.append(document)

    # Segunda passada, serial, so para as referencias que ficaram
    # INACCESSIBLE por um motivo possivelmente transitorio (rate-limit,
    # 5xx, reset de conexao). Um "read timeout" e assinatura de anti-bot
    # que nao responde ao nosso cliente — insistir e desperdicio. DEAD
    # (404) e ERROR tambem nao sao retentados.
    retriable = [
        r for r in updated.values() if r.status == ReferenceStatus.INACCESSIBLE and _worth_retrying(r)
    ]
    if retriable:
        retry_fetcher = (
            fetcher if fetcher_was_provided else HttpFetcher(timeout=(10, 40), max_retries=1, backoff=2.0)
        )
        logger.info(
            "Retentando %d referencia(s) inacessivel(is) (falha possivelmente transitoria)", len(retriable)
        )
        for ref in tqdm(
            retriable,
            desc="Retentando inacessíveis",
            unit="ref",
            colour=STAGE_COLORS["ingestion"],
        ):
            new_ref, document, markdown = _ingest_one_bounded(ref, retry_fetcher, _REF_BUDGET_SECONDS)
            updated[new_ref.id] = new_ref
            if document is not None and markdown is not None:
                registry.save_document(document, markdown)
                documents.append(document)
            if new_ref.status == ReferenceStatus.DOWNLOADED:
                logger.info("Referencia %s recuperada na segunda tentativa", ref.raw_url)

    final_references = list(updated.values())
    registry.save_references(final_references)
    return final_references, documents


def _worth_retrying(reference: Reference) -> bool:
    """A 2a passada so faz sentido para falhas potencialmente
    transitorias. Um `read timeout` (o servidor aceita a conexao e nunca
    responde) e assinatura de anti-bot por fingerprint — nao adianta
    insistir com o mesmo cliente."""
    msg = (reference.error_message or "").lower()
    return "read timeout" not in msg and "excedeu" not in msg


def _ingest_one_bounded(
    reference: Reference, fetcher: Fetcher, budget_seconds: float
) -> tuple[Reference, Document | None, str | None]:
    """`_ingest_one` com teto de tempo (a 2a passada e serial e um host
    problematico ainda poderia travar todo o lote)."""
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_ingest_one, reference, fetcher)
    try:
        return future.result(timeout=budget_seconds)
    except FuturesTimeoutError:
        future.cancel()
        logger.warning(
            "Referencia %s excedeu %.0fs na segunda tentativa — mantida inacessivel",
            reference.raw_url,
            budget_seconds,
        )
        return (
            _with_status(
                reference, ReferenceStatus.INACCESSIBLE, f"2a tentativa excedeu {budget_seconds:.0f}s"
            ),
            None,
            None,
        )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


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
        if result.fetch_method == "playwright" or is_pdf_content(result.content_type, reference.raw_url):
            # Playwright nao extrai texto de um PDF (abre o visualizador
            # nativo do Chromium) — nao adianta reter essa mesma falha
            # por ate 60s.
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
