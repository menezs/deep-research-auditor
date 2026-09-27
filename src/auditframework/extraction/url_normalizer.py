from __future__ import annotations

import re
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


_PMID_RE = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", re.IGNORECASE)
_PMC_RE = re.compile(
    r"(?:ncbi\.nlm\.nih\.gov/pmc/articles|pmc\.ncbi\.nlm\.nih\.gov/articles)/(PMC\d+)",
    re.IGNORECASE,
)
_DOI_IN_PATH_RE = re.compile(r"(10\.\d{4,9}/[^\s?#]+)", re.IGNORECASE)
_NATURE_RE = re.compile(r"nature\.com/articles/([a-z0-9.-]+)", re.IGNORECASE)
_ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", re.IGNORECASE)


def work_key(url: str) -> str:
    """Identificador da OBRA (o documento), independente do endereco usado
    para acessa-la: PMID > PMC > DOI > Nature > arXiv > URL normalizada.

    `normalize_url` responde "e a mesma pagina?"; isto responde "e o mesmo
    documento?" — perguntas diferentes, ambas necessarias. O mesmo artigo
    citado por PubMed, por DOI e pelo site do periodico gera tres URLs
    normalizadas distintas e uma unica `work_key`. Medir SELECAO de fontes
    (quem citou o que) pela URL subestima a intersecao; medir verificacao de
    TRECHO precisa da URL, porque enderecos diferentes da mesma obra expoem
    quantidades diferentes de texto (o PubMed mostra so o resumo).

    Limite conhecido: casos que exigem crosswalk externo (PubMed citado sem
    o DOI aparecer na URL) nao se resolvem por regex — um PMID e um DOI da
    mesma obra continuam sendo duas chaves aqui."""
    match = _PMID_RE.search(url)
    if match:
        return f"PMID:{match.group(1)}"

    match = _PMC_RE.search(url)
    if match:
        return f"PMC:{match.group(1).upper()}"

    match = _DOI_IN_PATH_RE.search(url)
    if match:
        return f"DOI:{match.group(1).rstrip('/').lower()}"

    match = _NATURE_RE.search(url)
    if match:
        # Todo periodico da familia Nature usa o prefixo de registrante
        # 10.1038 — o path e o sufixo do DOI sem o prefixo.
        return f"DOI:10.1038/{match.group(1).lower()}"

    match = _ARXIV_RE.search(url)
    if match:
        return f"ARXIV:{match.group(1)}"

    return normalize_url(url)
