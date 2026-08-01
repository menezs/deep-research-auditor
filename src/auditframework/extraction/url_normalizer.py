from __future__ import annotations

from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit

_TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "gclid",
    "fbclid",
    "mc_cid",
    "mc_eid",
    "igshid",
    "ref",
    "ref_src",
    "spm",
    "yclid",
    "msclkid",
}


def normalize_url(raw_url: str) -> str:
    """Normaliza uma URL para fins de deduplicacao entre execucoes e entre
    respostas de ferramentas de Deep Research diferentes que citem a
    mesma pagina.

    Corrige a normalizacao fraca do CorpusForge (`_normalize_ref_key`),
    que so fazia lowercase + strip de "/" final + canonicalizacao de DOI,
    e nao removia parametros de tracking, nem unificava esquema/`www.`,
    nem decodificava percent-encoding."""
    url = raw_url.strip()

    doi = _extract_doi(url)
    if doi:
        return f"https://doi.org/{doi.lower()}"

    parts = urlsplit(unquote(url))

    scheme = "https" if parts.scheme in ("http", "https", "") else parts.scheme

    netloc = parts.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    if scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[:-3]
    elif scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[:-4]

    path = parts.path.rstrip("/")

    query_pairs = sorted(
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() not in _TRACKING_PARAMS
    )
    query = urlencode(query_pairs)

    return urlunsplit((scheme, netloc, path, query, ""))


def _extract_doi(url: str) -> str | None:
    lowered = url.lower()
    if "doi.org/" in lowered:
        return lowered.split("doi.org/", 1)[-1].strip("/")
    if lowered.startswith("10.") and "/" in lowered:
        return lowered.strip("/")
    return None
