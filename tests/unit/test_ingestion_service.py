from pathlib import Path

from auditframework.common.errors import DeadReferenceError, InaccessibleReferenceError
from auditframework.ingestion.fetcher import FetchResult
from auditframework.ingestion.registry import ReferenceRegistry
from auditframework.ingestion.service import ingest_references
from auditframework.models import Reference, ReferenceStatus


class FakeFetcher:
    """Test double: simula HttpFetcher sem tocar a rede, e conta quantas
    vezes cada URL foi buscada (para verificar idempotencia)."""

    def __init__(self, behavior: dict[str, object]):
        self.behavior = behavior
        self.calls: dict[str, int] = {}

    def fetch(self, url: str) -> FetchResult:
        self.calls[url] = self.calls.get(url, 0) + 1
        outcome = self.behavior[url]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def fetch_via_playwright(self, url: str) -> FetchResult:
        return self.fetch(url)


def _reference(ref_id: str, url: str) -> Reference:
    return Reference(
        id=ref_id,
        citation_markers=["[1]"],
        raw_url=url,
        normalized_url=url,
        source_answer_id="answer-1",
        tool_name="ChatGPT",
    )


_ARTICLE_HTML = """
<html>
<head><title>Artigo de teste</title></head>
<body>
<article>
<h1>Marco Civil da Internet</h1>
<p>O Marco Civil da Internet estabelece principios, garantias, direitos e
deveres para o uso da internet no Brasil, incluindo neutralidade de rede
e protecao de dados pessoais dos usuarios.</p>
</article>
</body>
</html>
"""


def _html_result(body: str = _ARTICLE_HTML) -> FetchResult:
    return FetchResult(content=body.encode("utf-8"), content_type="text/html", fetch_method="requests", http_status=200)


def test_successful_reference_becomes_downloaded_document(tmp_path: Path):
    ref = _reference("ok1", "https://example.com/ok")
    registry = ReferenceRegistry(tmp_path)
    fetcher = FakeFetcher({"https://example.com/ok": _html_result()})

    updated, documents = ingest_references([ref], registry, fetcher=fetcher)

    assert updated[0].status == ReferenceStatus.DOWNLOADED
    assert len(documents) == 1
    assert registry.has_document("ok1")


def test_dead_reference_is_marked_dead_without_document(tmp_path: Path):
    ref = _reference("dead1", "https://example.com/404")
    registry = ReferenceRegistry(tmp_path)
    fetcher = FakeFetcher({"https://example.com/404": DeadReferenceError("404")})

    updated, documents = ingest_references([ref], registry, fetcher=fetcher)

    assert updated[0].status == ReferenceStatus.DEAD
    assert updated[0].error_message == "404"
    assert documents == []
    assert not registry.has_document("dead1")


def test_inaccessible_reference_is_marked_inaccessible(tmp_path: Path):
    ref = _reference("inacc1", "https://example.com/403")
    registry = ReferenceRegistry(tmp_path)
    fetcher = FakeFetcher({"https://example.com/403": InaccessibleReferenceError("403 persistente")})

    updated, _ = ingest_references([ref], registry, fetcher=fetcher)

    assert updated[0].status == ReferenceStatus.INACCESSIBLE


def test_already_downloaded_reference_is_not_refetched(tmp_path: Path):
    ref = _reference("ok1", "https://example.com/ok")
    registry = ReferenceRegistry(tmp_path)
    fetcher = FakeFetcher({"https://example.com/ok": _html_result()})

    ingest_references([ref], registry, fetcher=fetcher)
    ingest_references([ref], registry, fetcher=fetcher)

    assert fetcher.calls["https://example.com/ok"] == 1


def test_second_call_still_returns_the_already_downloaded_document(tmp_path: Path):
    """Regressao: a primeira versao so devolvia documentos baixados
    *nessa* chamada, entao uma segunda chamada (ex: ao retomar uma
    execucao) devolvia uma lista vazia mesmo com o documento ja
    disponivel em disco — o que quebraria a indexacao apos um resume."""
    ref = _reference("ok1", "https://example.com/ok")
    registry = ReferenceRegistry(tmp_path)
    fetcher = FakeFetcher({"https://example.com/ok": _html_result()})

    ingest_references([ref], registry, fetcher=fetcher)
    updated, documents = ingest_references([ref], registry, fetcher=fetcher)

    assert len(documents) == 1
    assert documents[0].reference_id == "ok1"
    assert updated[0].status == ReferenceStatus.DOWNLOADED


def test_second_call_with_fresh_pending_reference_object_keeps_downloaded_status(tmp_path: Path):
    """Regressao: se o chamador passar um novo objeto Reference (status
    PENDING por padrao, como acontece apos reextrair citacoes), uma
    referencia ja baixada nao pode retroceder para PENDING no
    references.json persistido."""
    ref = _reference("ok1", "https://example.com/ok")
    registry = ReferenceRegistry(tmp_path)
    fetcher = FakeFetcher({"https://example.com/ok": _html_result()})
    ingest_references([ref], registry, fetcher=fetcher)

    fresh_ref = _reference("ok1", "https://example.com/ok")
    assert fresh_ref.status == ReferenceStatus.PENDING

    updated, _ = ingest_references([fresh_ref], registry, fetcher=fetcher)

    assert updated[0].status == ReferenceStatus.DOWNLOADED
    assert registry.load_references()[0].status == ReferenceStatus.DOWNLOADED


def test_mixed_batch_updates_each_reference_independently(tmp_path: Path):
    ok_ref = _reference("ok1", "https://example.com/ok")
    dead_ref = _reference("dead1", "https://example.com/404")
    registry = ReferenceRegistry(tmp_path)
    fetcher = FakeFetcher(
        {
            "https://example.com/ok": _html_result(),
            "https://example.com/404": DeadReferenceError("nao encontrado"),
        }
    )

    updated, documents = ingest_references([ok_ref, dead_ref], registry, fetcher=fetcher)

    statuses = {ref.id: ref.status for ref in updated}
    assert statuses["ok1"] == ReferenceStatus.DOWNLOADED
    assert statuses["dead1"] == ReferenceStatus.DEAD
    assert len(documents) == 1
