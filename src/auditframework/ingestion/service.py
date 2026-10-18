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
from .barrier import Barrier, detect_barrier
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
    status (DOWNLOADED/DEAD/INACCESSIBLE/ERROR/BLOCKED). BLOCKED = o
    servidor respondeu, mas o conteudo e uma barreira de acesso
    (`ingestion.barrier`); o texto fica em `<id>.blocked.md`, fora da analise.

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
    # Barreira ja detectada numa execucao anterior: nao baixa de novo (o
    # `audit resume` refaria o mesmo download para cair na mesma tela).
    already_blocked = [
        ref for ref in references if not registry.has_document(ref.id) and registry.has_blocked_document(ref.id)
    ]
    pending = [
        ref for ref in references if not registry.has_document(ref.id) and not registry.has_blocked_document(ref.id)
    ]
    logger.info(
        "Ingestao: %d referencias, %d ja baixadas, %d ja bloqueadas, %d pendentes",
        len(references),
        len(already_downloaded),
        len(already_blocked),
        len(pending),
    )
    for ref in already_blocked:
        updated[ref.id] = _restore_blocked(ref, registry)

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
            _persist(registry, new_ref, document, markdown, documents)

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
            _persist(registry, new_ref, document, markdown, documents)
            if new_ref.status == ReferenceStatus.DOWNLOADED:
                logger.info("Referencia %s recuperada na segunda tentativa", ref.raw_url)

    final_references = list(updated.values())
    registry.save_references(final_references)
    return final_references, documents


def reclassify_documents(registry: ReferenceRegistry, *, dry_run: bool = False) -> list[Reference]:
    """Aplica a deteccao de barreiras aos documentos ja baixados de uma
    execucao (anteriores a esta checagem), sem baixar nada de novo.

    Devolve as referencias cujo conteudo e barreira, ja com status
    `BLOCKED`. Sem `dry_run`, cada documento vai para quarentena
    (`<id>.blocked.md`) e `references.json` e regravado."""
    references = registry.load_references(include_artifacts=True)
    blocked: list[Reference] = []
    updated: list[Reference] = []
    for ref in references:
        barrier = None
        if registry.has_document(ref.id):
            barrier = detect_barrier(registry.document_path(ref.id).read_text(encoding="utf-8"))
        if barrier is None:
            updated.append(ref)
            continue
        new_ref = _blocked(ref, barrier, ref.http_status).model_copy(update={"fetched_at": ref.fetched_at})
        blocked.append(new_ref)
        updated.append(new_ref)
    if blocked and not dry_run:
        for ref in blocked:
            registry.quarantine_document(ref.id)
        registry.save_references(updated)
    return blocked


def _persist(
    registry: ReferenceRegistry,
    reference: Reference,
    document: Document | None,
    markdown: str | None,
    documents: list[Document],
) -> None:
    """Grava o resultado de uma ingestao: documento valido em `<id>.md`,
    barreira de acesso em `<id>.blocked.md` (fora da analise, mas conferivel)."""
    if document is not None and markdown is not None:
        registry.save_document(document, markdown)
        documents.append(document)
    elif reference.status == ReferenceStatus.BLOCKED and markdown is not None:
        registry.save_blocked_document(reference.id, markdown)


def _blocked(reference: Reference, barrier: Barrier, http_status: int | None) -> Reference:
    return reference.model_copy(
        update={
            "status": ReferenceStatus.BLOCKED,
            "barrier_kind": barrier.kind,
            "http_status": http_status,
            "fetched_at": datetime.now(timezone.utc),
            "error_message": f"barreira de acesso ({barrier.kind}): {barrier.evidence}",
        }
    )


def _restore_blocked(reference: Reference, registry: ReferenceRegistry) -> Reference:
    """Status de uma referencia cuja barreira ja esta gravada. Quem chama
    pode passar a referencia recem-extraida (PENDING), entao o tipo da
    barreira e redetectado do arquivo quando nao vier preenchido."""
    if reference.status == ReferenceStatus.BLOCKED and reference.barrier_kind:
        return reference
    text = registry.blocked_document_path(reference.id).read_text(encoding="utf-8")
    barrier = detect_barrier(text) or Barrier("no_content", " ".join(text.split())[:120])
    return _blocked(reference, barrier, reference.http_status)


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
        barrier = detect_barrier(markdown)
        if barrier is not None:
            result, markdown, barrier = _retry_barrier_with_playwright(reference, fetcher, result, markdown, barrier)
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

    if barrier is not None:
        logger.warning("Barreira de acesso (%s) em %s", barrier.kind, reference.raw_url)
        return _blocked(reference, barrier, result.http_status), None, markdown

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


def _retry_barrier_with_playwright(
    reference: Reference, fetcher: Fetcher, result: FetchResult, markdown: str, barrier: Barrier
) -> tuple[FetchResult, str, Barrier | None]:
    """Uma barreira de JavaScript/anti-bot as vezes cai num navegador de
    verdade: tenta uma vez via Playwright antes de bloquear. PDF e conteudo
    ja vindo do Playwright nao tem o que ganhar com isso."""
    if result.fetch_method == "playwright" or is_pdf_content(result.content_type, reference.raw_url):
        return result, markdown, barrier
    try:
        retry = fetcher.fetch_via_playwright(reference.raw_url)
        retry_markdown = convert_to_markdown(retry.content, retry.content_type, reference.raw_url)
    except (InaccessibleReferenceError, ExtractionError):
        return result, markdown, barrier
    return retry, retry_markdown, detect_barrier(retry_markdown)


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
