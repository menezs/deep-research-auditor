from unittest.mock import MagicMock, patch

import pytest
import requests

from auditframework.common.errors import DeadReferenceError, InaccessibleReferenceError
from auditframework.ingestion.fetcher import HttpFetcher, _retry_after_seconds


def _response(status_code: int, content: bytes = b"<html></html>", headers: dict | None = None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content
    resp.headers = headers or {"Content-Type": "text/html"}
    return resp


def _hit(status_code: int, content: bytes = b"<html></html>", headers: dict | None = None, method: str = "curl_cffi"):
    """Retorno de `_http_get`: (response, metodo)."""
    return _response(status_code, content, headers), method


@patch("auditframework.ingestion.fetcher._http_get")
def test_successful_fetch_returns_result(mock_get):
    mock_get.return_value = _hit(200, b"<html>ok</html>")
    fetcher = HttpFetcher()

    result = fetcher.fetch("https://example.com")

    assert result.fetch_method == "curl_cffi"
    assert result.http_status == 200
    assert result.content == b"<html>ok</html>"


@patch("auditframework.ingestion.fetcher._http_get")
def test_404_raises_dead_reference_error(mock_get):
    mock_get.return_value = _hit(404)
    fetcher = HttpFetcher()

    with pytest.raises(DeadReferenceError):
        fetcher.fetch("https://example.com/nao-existe")


@patch("time.sleep", return_value=None)
@patch("auditframework.ingestion.fetcher._http_get")
def test_persistent_connection_error_raises_inaccessible(mock_get, _mock_sleep):
    mock_get.side_effect = requests.exceptions.ConnectionError("boom")
    fetcher = HttpFetcher(max_retries=2)

    with pytest.raises(InaccessibleReferenceError):
        fetcher.fetch("https://example.com")

    assert mock_get.call_count == 3  # tentativa inicial + 2 retries


@patch("time.sleep", return_value=None)
@patch("auditframework.ingestion.fetcher._http_get")
def test_transient_error_then_success_recovers(mock_get, _mock_sleep):
    mock_get.side_effect = [
        requests.exceptions.Timeout("slow"),
        _hit(200, b"ok"),
    ]
    fetcher = HttpFetcher(max_retries=2)

    result = fetcher.fetch("https://example.com")

    assert result.content == b"ok"
    assert mock_get.call_count == 2


@patch("time.sleep", return_value=None)
@patch("auditframework.ingestion.fetcher._http_get")
def test_read_timeout_is_retried_at_most_once(mock_get, _mock_sleep):
    """Um read timeout e assinatura de anti-bot que nao responde ao nosso
    cliente — no maximo 1 nova tentativa, ignorando `max_retries` alto."""
    mock_get.side_effect = requests.exceptions.ReadTimeout("tarpit")
    fetcher = HttpFetcher(max_retries=5)

    with pytest.raises(InaccessibleReferenceError, match="read timeout"):
        fetcher.fetch("https://tarpit.example.com")

    assert mock_get.call_count == 2  # inicial + 1 unica retentativa


@patch("auditframework.ingestion.fetcher._http_get")
def test_url_with_percent20_falls_back_to_hyphen_variant_on_404(mock_get):
    """Regressao: URLs reconstruidas pela extracao com `%20` no lugar de
    um ponto de quebra de linha ambiguo (poderia ser `-` ou nenhum
    separador) devem tentar essas variantes antes de desistir — mesma
    tecnica do CorpusForge (`_try_url_variations`)."""

    def side_effect(url, **kwargs):
        if url == "https://example.com/doenca%20de-alzheimer":
            return _hit(404)
        if url == "https://example.com/doenca-de-alzheimer":
            return _hit(200, b"ok")
        raise AssertionError(f"URL inesperada: {url}")

    mock_get.side_effect = side_effect
    fetcher = HttpFetcher()

    result = fetcher.fetch("https://example.com/doenca%20de-alzheimer")

    assert result.content == b"ok"
    assert mock_get.call_count == 2


@patch("auditframework.ingestion.fetcher._http_get")
def test_url_with_percent20_falls_back_to_no_separator_variant(mock_get):
    """A variante `-` tambem pode falhar (ex: a URL real nao tinha
    separador nenhum no ponto de quebra) — nesse caso tenta a variante
    sem separador antes de desistir."""

    def side_effect(url, **kwargs):
        if url == "https://example.com/do%20enca":
            return _hit(404)
        if url == "https://example.com/do-enca":
            return _hit(404)
        if url == "https://example.com/doenca":
            return _hit(200, b"ok")
        raise AssertionError(f"URL inesperada: {url}")

    mock_get.side_effect = side_effect
    fetcher = HttpFetcher()

    result = fetcher.fetch("https://example.com/do%20enca")

    assert result.content == b"ok"
    assert mock_get.call_count == 3


@patch("auditframework.ingestion.fetcher._http_get")
def test_url_with_percent20_raises_original_error_when_no_variant_works(mock_get):
    mock_get.return_value = _hit(404)
    fetcher = HttpFetcher()

    with pytest.raises(DeadReferenceError, match="doenca%20de-alzheimer"):
        fetcher.fetch("https://example.com/doenca%20de-alzheimer")

    assert mock_get.call_count == 3  # original + 2 variantes


@patch("auditframework.ingestion.fetcher._http_get")
def test_url_without_percent20_does_not_try_variants_on_404(mock_get):
    mock_get.return_value = _hit(404)
    fetcher = HttpFetcher()

    with pytest.raises(DeadReferenceError):
        fetcher.fetch("https://example.com/pagina-normal")

    assert mock_get.call_count == 1


@patch("auditframework.ingestion.fetcher.HttpFetcher.fetch_via_playwright")
@patch("auditframework.ingestion.fetcher._http_get")
def test_403_falls_back_to_cloudscraper_then_playwright(mock_get, mock_playwright):
    mock_get.return_value = _hit(403)

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


@pytest.mark.parametrize(
    "header,expected",
    [(None, 30.0), ("5", 5.0), ("120", 30.0), ("3600", 30.0), ("Mon, 01 Jan 2035 00:00:00 GMT", 30.0)],
)
def test_retry_after_is_capped(header, expected):
    assert _retry_after_seconds(header) == expected


class TestRedditJson:
    def test_thread_url_maps_to_json_endpoint(self):
        from auditframework.ingestion.fetcher import _reddit_thread_json_url

        assert _reddit_thread_json_url(
            "https://www.reddit.com/r/x/comments/abc123/some_title/"
        ) == "https://www.reddit.com/r/x/comments/abc123/some_title.json"
        assert _reddit_thread_json_url(
            "https://old.reddit.com/r/x/comments/abc123/t?foo=1"
        ) == "https://old.reddit.com/r/x/comments/abc123/t.json?foo=1"

    def test_non_thread_reddit_and_other_hosts_return_none(self):
        from auditframework.ingestion.fetcher import _reddit_thread_json_url

        assert _reddit_thread_json_url("https://www.reddit.com/r/x/") is None
        assert _reddit_thread_json_url("https://www.reddit.com/user/foo") is None
        assert _reddit_thread_json_url("https://en.wikipedia.org/wiki/X") is None

    @patch("auditframework.ingestion.fetcher.requests.get")
    def test_fetch_uses_json_endpoint_for_reddit_threads(self, mock_get):
        mock_get.return_value = _response(200, b'[{}]', {"Content-Type": "application/json"})
        result = HttpFetcher().fetch("https://www.reddit.com/r/x/comments/abc123/t/")

        assert result.fetch_method == "reddit_json"
        assert result.content_type == "application/json"
        assert mock_get.call_args[0][0] == "https://www.reddit.com/r/x/comments/abc123/t.json"

    @patch("auditframework.ingestion.fetcher._http_get")
    @patch("auditframework.ingestion.fetcher.requests.get")
    def test_reddit_json_failure_falls_back_to_html(self, mock_get, mock_http_get):
        mock_get.return_value = _response(403, b"blocked", {"Content-Type": "application/json"})
        mock_http_get.return_value = _hit(200, b"<html>ok</html>")

        result = HttpFetcher().fetch("https://www.reddit.com/r/x/comments/abc123/t/")

        assert result.fetch_method == "curl_cffi"
        assert result.content == b"<html>ok</html>"
