from __future__ import annotations

from pathlib import Path


def extract_pdf_hyperlink_urls(path: Path) -> set[str]:
    """Le os URIs das anotacoes de hyperlink embutidas num PDF — a URL
    real armazenada no link, independente de como o texto visivel foi
    renderizado/quebrado em linha. Mais confiavel que reconstruir uma URL
    a partir do texto convertido pelo `pymupdf4llm` (`extraction/loaders.py`),
    que pode descartar hifens em pontos de quebra sem deixar nenhum sinal
    textual disso.

    Import tardio de `fitz` (pymupdf, dependencia transitiva de
    `pymupdf4llm`) para nao exigir o extra `[ingestion]` fora do caso de
    uso. Nunca levanta excecao: se `fitz` nao estiver instalado ou o PDF
    falhar ao abrir, retorna um set vazio — a extracao de referencias deve
    continuar normalmente sem esse reparo."""
    try:
        import fitz
    except ImportError:
        return set()

    try:
        doc = fitz.open(path)
    except Exception:
        return set()

    urls: set[str] = set()
    with doc:
        for page in doc:
            for link in page.get_links():
                uri = link.get("uri")
                if uri and uri.startswith(("http://", "https://")):
                    urls.add(uri.split("#", 1)[0])
    return urls
