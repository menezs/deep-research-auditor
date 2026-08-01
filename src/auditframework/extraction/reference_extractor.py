from __future__ import annotations

import re
from typing import Protocol

from pydantic import BaseModel, Field

from ..common.errors import LLMParseError
from ..common.llm_client import LLMClient
from ..models import Reference
from .url_normalizer import normalize_url

_MARKER_RUN = re.compile(r"(?:\[\s*(\d+)\s*\]\s*)+")
_URL_RE = re.compile(r'https?://[^\s<>\[\]()"]+')


def _extract_url(text: str) -> tuple[str, int] | None:
    """Encontra a primeira URL em `text` e reconecta um token de
    continuacao imediatamente seguinte, se houver.

    Conversores PDF->Markdown (ex: pymupdf4llm) as vezes quebram uma URL
    longa demais para uma linha do PDF inserindo um espaco no ponto de
    quebra em vez de manter a URL contigua (ex: "...transparente-p
    ara-boa-governanca-de-ia"). Como isso so pode acontecer logo apos a
    URL nessa mesma linha, e um token de continuacao legitimo nao
    contem espacos, reconectar apenas quando exatamente um token segue
    a URL (sem espaco depois dele) e seguro: nao afeta URLs seguidas de
    prosa (que teria mais de uma palavra)."""
    match = _URL_RE.search(text)
    if match is None:
        return None
    url = match.group(0)
    trailing = text[match.end() :].strip()
    if trailing and " " not in trailing:
        url += trailing
    return url, match.start()


class ReferenceExtractionStrategy(Protocol):
    def extract(self, text: str, *, source_answer_id: str, tool_name: str) -> list[Reference]: ...


class RegexReferenceExtractor:
    """Extrai referencias da lista de fontes que ChatGPT/Gemini/Perplexity
    tipicamente produzem ao final da resposta, no formato:

        [1] Titulo do documento
        https://exemplo.com/artigo

    ou com varios marcadores apontando para a mesma URL:

        [1] [3] [7] Titulo do documento
        https://exemplo.com/artigo

    Estrategia padrao, sem dependencia de LLM. Para respostas cuja lista
    de fontes nao segue esse formato, use `LLMReferenceExtractor` (Fase 3),
    que reaproveita o `LLMClient` compartilhado com o estagio de
    julgamento — evitando duplicar, como acontecia entre CorpusForge e
    audit_with_llm, duas implementacoes de cliente LLM incompativeis
    entre si."""

    def extract(self, text: str, *, source_answer_id: str, tool_name: str) -> list[Reference]:
        by_id: dict[str, Reference] = {}
        for markers, title, raw_url in _find_reference_entries(text):
            normalized = normalize_url(raw_url)
            ref_id = Reference.id_for_url(normalized)
            existing = by_id.get(ref_id)
            if existing is not None:
                merged_markers = list(dict.fromkeys(existing.citation_markers + markers))
                by_id[ref_id] = existing.model_copy(update={"citation_markers": merged_markers})
                continue
            by_id[ref_id] = Reference(
                id=ref_id,
                citation_markers=markers,
                raw_url=raw_url,
                normalized_url=normalized,
                title=title or None,
                source_answer_id=source_answer_id,
                tool_name=tool_name,
            )
        return list(by_id.values())


def _find_reference_entries(text: str) -> list[tuple[list[str], str, str]]:
    entries: list[tuple[list[str], str, str]] = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        marker_match = _MARKER_RUN.match(stripped)
        if not marker_match:
            continue
        markers = [f"[{n}]" for n in re.findall(r"\d+", marker_match.group(0))]
        rest = stripped[marker_match.end():].strip()

        found = _extract_url(rest)
        if found is not None:
            url, url_start = found
            title = rest[:url_start].strip()
            entries.append((markers, title, url))
            continue

        # URL nao esta na mesma linha dos marcadores: procura nas linhas
        # seguintes, pulando linhas em branco (conversores PDF->Markdown
        # como pymupdf4llm costumam inserir uma linha em branco entre o
        # titulo e a URL) e acumulando linhas nao-vazias sem URL como
        # continuacao do titulo (titulos longos podem quebrar em mais de
        # uma linha). Para na proxima entrada de referencia para nao
        # "vazar" a URL de um item seguinte para este.
        title_parts = [rest] if rest else []
        found_url: str | None = None
        for next_line in lines[i + 1 :]:
            candidate = next_line.strip()
            if not candidate:
                continue
            if _MARKER_RUN.match(candidate):
                break
            candidate_found = _extract_url(candidate)
            if candidate_found is not None:
                found_url = candidate_found[0]
                break
            title_parts.append(candidate)

        if found_url:
            entries.append((markers, " ".join(title_parts).strip(), found_url))

    return entries


def extract_references(
    text: str,
    *,
    source_answer_id: str,
    tool_name: str,
    strategy: ReferenceExtractionStrategy | None = None,
) -> list[Reference]:
    strategy = strategy or RegexReferenceExtractor()
    return strategy.extract(text, source_answer_id=source_answer_id, tool_name=tool_name)


class _ExtractedReference(BaseModel):
    citation_markers: list[str] = Field(default_factory=list)
    url: str
    title: str | None = None


class _ExtractedReferenceList(BaseModel):
    references: list[_ExtractedReference] = Field(default_factory=list)


_LLM_EXTRACTION_SYSTEM_MESSAGE = (
    "Voce extrai referencias bibliograficas de respostas de ferramentas de "
    "Deep Research com precisao. Nao invente URLs; extraia apenas o que "
    "estiver explicitamente presente no texto."
)


class LLMReferenceExtractor:
    """Estrategia alternativa ao `RegexReferenceExtractor`, para respostas
    cuja lista de fontes nao segue o formato "[N] Titulo\\nURL". Reaproveita
    o `LLMClient` compartilhado com o estagio de julgamento (`judging/`)."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def extract(self, text: str, *, source_answer_id: str, tool_name: str) -> list[Reference]:
        prompt = (
            "Extraia todas as referencias citadas na lista de fontes desta "
            "resposta de uma ferramenta de Deep Research. Para cada "
            'referencia, retorne os marcadores de citacao no formato "[N]" '
            "que apontam para ela (pode haver mais de um), a URL completa "
            "e, se houver, o titulo.\n\n"
            f"TEXTO:\n{text}"
        )
        try:
            output, _usage = self.llm_client.complete_json(
                system_message=_LLM_EXTRACTION_SYSTEM_MESSAGE,
                user_prompt=prompt,
                schema=_ExtractedReferenceList,
            )
        except LLMParseError:
            return []

        by_id: dict[str, Reference] = {}
        for item in output.references:
            normalized = normalize_url(item.url)
            ref_id = Reference.id_for_url(normalized)
            existing = by_id.get(ref_id)
            if existing is not None:
                merged_markers = list(dict.fromkeys(existing.citation_markers + item.citation_markers))
                by_id[ref_id] = existing.model_copy(update={"citation_markers": merged_markers})
                continue
            by_id[ref_id] = Reference(
                id=ref_id,
                citation_markers=item.citation_markers,
                raw_url=item.url,
                normalized_url=normalized,
                title=item.title,
                source_answer_id=source_answer_id,
                tool_name=tool_name,
            )
        return list(by_id.values())
