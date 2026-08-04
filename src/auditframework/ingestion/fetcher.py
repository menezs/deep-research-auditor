from __future__ import annotations

import time
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

import requests

from ..common.errors import DeadReferenceError, InaccessibleReferenceError

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,pt-BR;q=0.8,pt;q=0.7",
}

_REDDIT_HOSTS = {"reddit.com", "www.reddit.com"}


def _rewrite_known_hosts(url: str) -> str:
    """Reddit exige um desafio JS de verificacao em `www.reddit.com` que o
    Playwright headless nunca resolve (fica em polling ate estourar o
    timeout de `networkidle`). `old.reddit.com` serve o mesmo conteudo sem
    esse desafio."""
    parts = urlsplit(url)
    if parts.netloc in _REDDIT_HOSTS:
        parts = parts._replace(netloc="old.reddit.com")
        return urlunsplit(parts)
    return url


def _is_pdf_url(url: str) -> bool:
    return urlsplit(url).path.lower().endswith(".pdf")


class _Http403Error(Exception):
    """Sinal de controle interno: 403 recebido, tentar proximo fallback."""


@dataclass
class FetchResult:
    content: bytes
    content_type: str
    fetch_method: str
    http_status: int


class HttpFetcher:
    """Baixa o conteudo de uma URL com fallback em cascata: requests ->
    cloudscraper (anti-bot) -> playwright (paginas JS-renderizadas ou
    protegidas por JS-challenge que o cloudscraper nao resolve).

    Porta a logica de fallback do CorpusForge (`FileConverter._fetch_html`),
    mas substitui os dicts de erro livre por excecoes tipadas
    (`DeadReferenceError`/`InaccessibleReferenceError`)."""

    def __init__(
        self,
        timeout: tuple[float, float] = (10, 60),
        max_retries: int = 2,
        backoff: float = 2.0,
    ) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff

    def fetch(self, url: str) -> FetchResult:
        url = _rewrite_known_hosts(url)
        try:
            return self._fetch_once(url)
        except DeadReferenceError:
            if "%20" not in url:
                raise
            for variant in self._url_variations(url):
                try:
                    return self._fetch_once(variant)
                except (DeadReferenceError, InaccessibleReferenceError):
                    continue
            raise

    def _fetch_once(self, url: str) -> FetchResult:
        try:
            return self._get(url, verify=True)
        except requests.exceptions.SSLError:
            try:
                return self._get(url, verify=False)
            except _Http403Error:
                return self._fetch_with_cloudscraper(url)
        except _Http403Error:
            return self._fetch_with_cloudscraper(url)

    @staticmethod
    def _url_variations(url: str) -> list[str]:
        """Duas variantes de reparo para uma URL reconstruida com `%20` no
        lugar de um ponto de quebra de linha ambiguo do PDF original (a
        extracao de referencias sempre normaliza espaco literal para
        `%20`, nunca deixa espaco bruto na URL armazenada) — o `%20`
        poderia representar um separador `-`, ou nenhum separador nenhum.
        Mesma tecnica usada pelo CorpusForge
        (`FileConverter._try_url_variations`), aqui reaproveitando toda a
        cadeia de fallback existente (`requests -> cloudscraper ->
        playwright`) para cada variante, em vez de uma sondagem HEAD
        separada."""
        return [url.replace("%20", "-"), url.replace("%20", "")]

    def fetch_via_playwright(self, url: str) -> FetchResult:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise InaccessibleReferenceError(
                f"Falha ao acessar {url} e 'playwright' nao instalado "
                "(instale o extra [ingestion] e rode "
                "`python -m playwright install chromium`)"
            ) from exc

        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch()
                page = browser.new_page(extra_http_headers=_HEADERS)
                page.goto(url, wait_until="domcontentloaded", timeout=45_000)
                page.wait_for_timeout(2_000)
                html = page.content()
                browser.close()
        except Exception as exc:  # biblioteca externa: superficie de erro ampla
            raise InaccessibleReferenceError(
                f"Falha ao renderizar {url} via playwright: {exc}"
            ) from exc

        return FetchResult(
            content=html.encode("utf-8"),
            content_type="text/html",
            fetch_method="playwright",
            http_status=200,
        )

    def _get(self, url: str, *, verify: bool, attempt: int = 0) -> FetchResult:
        try:
            response = requests.get(
                url, headers=_HEADERS, timeout=self.timeout, verify=verify, allow_redirects=True
            )
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            if attempt >= self.max_retries:
                raise InaccessibleReferenceError(f"Falha ao conectar em {url}: {exc}") from exc
            time.sleep(self.backoff * (2**attempt))
            return self._get(url, verify=verify, attempt=attempt + 1)

        if response.status_code == 404:
            raise DeadReferenceError(f"Referencia nao encontrada (404): {url}")

        if response.status_code == 403:
            raise _Http403Error(url)

        if response.status_code == 429:
            if attempt >= self.max_retries:
                raise InaccessibleReferenceError(f"Rate limit persistente em {url}")
            retry_after = float(response.headers.get("Retry-After", 60))
            time.sleep(retry_after)
            return self._get(url, verify=verify, attempt=attempt + 1)

        if response.status_code >= 400:
            if attempt >= self.max_retries:
                raise InaccessibleReferenceError(f"HTTP {response.status_code} em {url}")
            time.sleep(self.backoff * (2**attempt))
            return self._get(url, verify=verify, attempt=attempt + 1)

        return FetchResult(
            content=response.content,
            content_type=response.headers.get("Content-Type", ""),
            fetch_method="requests",
            http_status=response.status_code,
        )

    def _fetch_with_cloudscraper(self, url: str) -> FetchResult:
        try:
            import cloudscraper
        except ImportError as exc:
            raise InaccessibleReferenceError(
                f"HTTP 403 em {url} e 'cloudscraper' nao instalado (extra [ingestion])"
            ) from exc

        scraper = cloudscraper.create_scraper()
        response = scraper.get(url, headers=_HEADERS, timeout=self.timeout)

        if response.status_code == 403:
            if _is_pdf_url(url):
                # Playwright nao extrai texto de um PDF (abre o visualizador
                # nativo do Chromium) — tentar so adiaria essa mesma falha
                # por ate 60s.
                raise InaccessibleReferenceError(f"HTTP 403 em {url} (PDF, sem fallback via playwright)")
            return self.fetch_via_playwright(url)
        if response.status_code == 404:
            raise DeadReferenceError(f"Referencia nao encontrada (404): {url}")
        if response.status_code >= 400:
            raise InaccessibleReferenceError(
                f"HTTP {response.status_code} em {url} (cloudscraper)"
            )

        return FetchResult(
            content=response.content,
            content_type=response.headers.get("Content-Type", ""),
            fetch_method="cloudscraper",
            http_status=response.status_code,
        )
