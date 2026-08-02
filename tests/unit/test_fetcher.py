from unittest.mock import MagicMock, patch

import pytest
import requests

from auditframework.common.errors import DeadReferenceError, InaccessibleReferenceError
from auditframework.ingestion.fetcher import HttpFetcher


def _response(status_code: int, content: bytes = b"<html></html>", headers: dict | None = None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content
    resp.headers = headers or {"Content-Type": "text/html"}
    return resp


@patch("auditframework.ingestion.fetcher.requests.get")
def test_successful_fetch_returns_result(mock_get):
    mock_get.return_value = _response(200, b"<html>ok</html>")
    fetcher = HttpFetcher()

    result = fetcher.fetch("https://example.com")

    assert result.fetch_method == "requests"
    assert result.http_status == 200
    assert result.content == b"<html>ok</html>"


@patch("auditframework.ingestion.fetcher.requests.get")
def test_404_raises_dead_reference_error(mock_get):
    mock_get.return_value = _response(404)
    fetcher = HttpFetcher()

    with pytest.raises(DeadReferenceError):
        fetcher.fetch("https://example.com/nao-existe")


@patch("time.sleep", return_value=None)
@patch("auditframework.ingestion.fetcher.requests.get")
def test_persistent_connection_error_raises_inaccessible(mock_get, _mock_sleep):
    mock_get.side_effect = requests.exceptions.ConnectionError("boom")
    fetcher = HttpFetcher(max_retries=2)

    with pytest.raises(InaccessibleReferenceError):
        fetcher.fetch("https://example.com")

    assert mock_get.call_count == 3  # tentativa inicial + 2 retries


@patch("time.sleep", return_value=None)
@patch("auditframework.ingestion.fetcher.requests.get")
def test_transient_error_then_success_recovers(mock_get, _mock_sleep):
    mock_get.side_effect = [
        requests.exceptions.Timeout("slow"),
        _response(200, b"ok"),
    ]
    fetcher = HttpFetcher(max_retries=2)

    result = fetcher.fetch("https://example.com")

    assert result.content == b"ok"
    assert mock_get.call_count == 2


@patch("auditframework.ingestion.fetcher.requests.get")
def test_url_with_percent20_falls_back_to_hyphen_variant_on_404(mock_get):
    """Regressao: URLs reconstruidas pela extracao com `%20` no lugar de
    um ponto de quebra de linha ambiguo (poderia ser `-` ou nenhum
    separador) devem tentar essas variantes antes de desistir — mesma
    tecnica do CorpusForge (`_try_url_variations`)."""

    def side_effect(url, **kwargs):
        if url == "https://example.com/doenca%20de-alzheimer":
            return _response(404)
        if url == "https://example.com/doenca-de-alzheimer":
            return _response(200, b"ok")
        raise AssertionError(f"URL inesperada: {url}")

    mock_get.side_effect = side_effect
    fetcher = HttpFetcher()

    result = fetcher.fetch("https://example.com/doenca%20de-alzheimer")

    assert result.content == b"ok"
    assert mock_get.call_count == 2


@patch("auditframework.ingestion.fetcher.requests.get")
def test_url_with_percent20_falls_back_to_no_separator_variant(mock_get):
    """A variante `-` tambem pode falhar (ex: a URL real nao tinha
    separador nenhum no ponto de quebra) — nesse caso tenta a variante
    sem separador antes de desistir."""

    def side_effect(url, **kwargs):
        if url == "https://example.com/do%20enca":
            return _response(404)
        if url == "https://example.com/do-enca":
            return _response(404)
        if url == "https://example.com/doenca":
            return _response(200, b"ok")
        raise AssertionError(f"URL inesperada: {url}")

    mock_get.side_effect = side_effect
    fetcher = HttpFetcher()

    result = fetcher.fetch("https://example.com/do%20enca")

    assert result.content == b"ok"
    assert mock_get.call_count == 3


@patch("auditframework.ingestion.fetcher.requests.get")
def test_url_with_percent20_raises_original_error_when_no_variant_works(mock_get):
    mock_get.return_value = _response(404)
    fetcher = HttpFetcher()

    with pytest.raises(DeadReferenceError, match="doenca%20de-alzheimer"):
        fetcher.fetch("https://example.com/doenca%20de-alzheimer")

    assert mock_get.call_count == 3  # original + 2 variantes


@patch("auditframework.ingestion.fetcher.requests.get")
def test_url_without_percent20_does_not_try_variants_on_404(mock_get):
    mock_get.return_value = _response(404)
    fetcher = HttpFetcher()

    with pytest.raises(DeadReferenceError):
        fetcher.fetch("https://example.com/pagina-normal")

    assert mock_get.call_count == 1


@patch("auditframework.ingestion.fetcher.HttpFetcher.fetch_via_playwright")
@patch("auditframework.ingestion.fetcher.requests.get")
def test_403_falls_back_to_cloudscraper_then_playwright(mock_get, mock_playwright):
    mock_get.return_value = _response(403)

    fake_cloudscraper = MagicMock()
    fake_scraper = MagicMock()
    fake_scraper.get.return_value = _response(403)
    fake_cloudscraper.create_scraper.return_value = fake_scraper

    from auditframework.ingestion.fetcher import FetchResult

    mock_playwright.return_value = FetchResult(
        content=b"<html>via playwright</html>",
        content_type="text/html",
        fetch_method="playwright",
        http_status=200,
    )

    with patch.dict("sys.modules", {"cloudscraper": fake_cloudscraper}):
        fetcher = HttpFetcher()
        result = fetcher.fetch("https://protegido.example.com")

    assert result.fetch_method == "playwright"
    mock_playwright.assert_called_once_with("https://protegido.example.com")
