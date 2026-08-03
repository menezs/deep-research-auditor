from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol
from urllib.parse import unquote

from pydantic import BaseModel, Field

from ..common.errors import LLMParseError
from ..common.llm_client import LLMClient
from ..models import Reference
from .pdf_links import extract_pdf_hyperlink_urls
from .url_normalizer import normalize_url

_MARKER_RUN = re.compile(r"(?:\[\s*(\d+)\s*\]\s*)+")
_URL_RE = re.compile(r'https?://[^\s<>\[\]()"]+')

_REFERENCE_SECTION_HEADING = re.compile(
    r"^#{1,6}\s*\**\s*(refer[eê]ncias|references|fontes|sources|bibliografia)\b",
    re.IGNORECASE | re.MULTILINE,
)
"""Cabecalho de secao de lista de fontes. Tolera negrito markdown ao redor
da palavra-chave e texto extra depois dela (ex: "### **Referencias
citadas**" do Gemini em PDF) — nao exige mais que a linha seja *so* a
palavra-chave, so que comece com ela."""

_ASTERISM = "⁂"
_ASTERISM_TOKEN_RE = re.compile(
    r"(?:\A|(?<=\s))(?:[#*\-][ \t]*)*(?P<marker>\d+)\.[ \t]*"
    r"|<u>(?P<url_part>.*?)</u>",
    re.IGNORECASE | re.DOTALL,
)


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
    prosa (que teria mais de uma palavra). Um trailing comecando com `<`
    nunca e reconectado — `_URL_RE` ja exclui `<`/`>` do match, entao um
    trailing assim so pode ser uma tag HTML colada (ex: `</u>` do formato
    de lista do Perplexity), nunca uma continuacao legitima de URL."""
    match = _URL_RE.search(text)
    if match is None:
        return None
    url = match.group(0)
    trailing = text[match.end() :].strip()
    if trailing and " " not in trailing and not trailing.startswith("<"):
        url += trailing
    return url, match.start()


class ReferenceExtractionStrategy(Protocol):
    def extract(self, text: str, *, source_answer_id: str, tool_name: str) -> list[Reference]: ...


class RegexReferenceExtractor:
    """Extrai referencias da lista de fontes que ChatGPT/Gemini/Perplexity
    tipicamente produzem ao final da resposta. Suporta 4 formatos, cada um
    tratado por uma passada independente. Os dois primeiros (colchetes e
    `⁂`) sao unidos incondicionalmente — um documento real ja observado
    mistura ambos (bracket no corpo + `⁂` numa lista a parte) — e o
    formato de "Titulo, URL" sem marcacao nenhuma (`_find_heading_titled_pairs`)
    so entra como ultimo recurso, quando os dois primeiros nao acharem
    nada, para nao competir/duplicar entradas ja resolvidas:

    ChatGPT/Gemini (PDF/MD) — marcador `[N]` entre colchetes
    (`_find_reference_entries`):

        [1] Titulo do documento
        https://exemplo.com/artigo

    ou com varios marcadores apontando para a mesma URL:

        [1] [3] [7] Titulo do documento
        https://exemplo.com/artigo

    Perplexity (PDF) — lista numerada sem colchetes, apos um separador `⁂`
    (`_find_asterism_list_entries`):

        1. <u>https://exemplo.com/artigo</u>

    Gemini (PDF) — mesma gramatica acima, mas ancorada pelo cabecalho da
    secao de fontes em vez de `⁂` (o Gemini nao usa esse separador):

        ### **Referências citadas**
        1. Titulo do documento, <u>https://exemplo.com/artigo</u>

    Perplexity (DOCX)/Gemini (DOCX) — sem numeracao nem `<u>`, uma entrada
    por linha apos `⁂` (`_find_asterism_bare_url_entries`, so URL) ou apos
    o cabecalho da secao de fontes (`_find_heading_titled_pairs`, com
    titulo — usado quando python-docx ja devolve o paragrafo intacto,
    sem marcacao nenhuma ao redor da URL):

        Titulo do documento, https://exemplo.com/artigo

    Estrategia padrao, sem dependencia de LLM. Para respostas cuja lista
    de fontes nao segue nenhum desses formatos, use `LLMReferenceExtractor`
    (Fase 3), que reaproveita o `LLMClient` compartilhado com o estagio de
    julgamento — evitando duplicar, como acontecia entre CorpusForge e
    audit_with_llm, duas implementacoes de cliente LLM incompativeis
    entre si."""

    def extract(self, text: str, *, source_answer_id: str, tool_name: str) -> list[Reference]:
        by_id: dict[str, Reference] = {}
        entries = _find_reference_entries(text) + _find_asterism_entries(text)
        if not entries:
            entries = _find_heading_titled_pairs(text)
        for markers, title, raw_url in entries:
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


def _reference_section_tail(text: str) -> str | None:
    """Retorna o texto a partir de onde a lista de fontes comeca, usando
    qualquer uma das duas ancoras ja observadas: o separador `⁂`
    (Perplexity) ou o cabecalho da secao de referencias
    (`_REFERENCE_SECTION_HEADING`, ex: ChatGPT/Gemini) — a que vier POR
    ULTIMO no texto, se houver mais de uma. Usar a ultima (nao a
    primeira) importa quando um cabecalho chamado "Referências" existe
    cedo no documento mas NAO e o inicio real da lista (ex: um resumo com
    lista numerada propria antes do `⁂` de verdade) — nesse caso a ancora
    mais especifica e proxima da lista de fato deve vencer. `None` se
    nenhuma ancora existir (documento em formato desconhecido)."""
    starts: list[int] = []
    asterism_pos = text.find(_ASTERISM)
    if asterism_pos != -1:
        starts.append(asterism_pos + 1)
    heading_match = _REFERENCE_SECTION_HEADING.search(text)
    if heading_match is not None:
        starts.append(heading_match.end())
    if not starts:
        return None
    return text[max(starts) :]


def _find_asterism_list_entries(text: str) -> list[tuple[list[str], str, str]]:
    """Extrai uma lista de fontes numerada (sem colchetes), apos um
    separador "asterismo" (⁂, Perplexity) ou um cabecalho de secao de
    referencias (ChatGPT/Gemini, ver `_reference_section_tail`) — formato
    estruturalmente diferente do `[N] Titulo\\nURL` tratado por
    `_find_reference_entries`:

        1. <u>https://exemplo.com/artigo</u>
        2. <u>https://exemplo.com/a</u> 3. <u>https://exemplo.com/b</u>

    Retorna `[]` se nenhuma ancora existir — nao afeta documentos em
    outros formatos. So opera no texto APOS a ancora (nunca no resto do
    documento), para nao confundir uma lista numerada comum do corpo da
    resposta com uma entrada de referencia.

    Cada `N.` (tolerando ruido de markdown antes, ex: `### ` ou `- `)
    inicia uma entrada nova; cada trecho `<u>...</u>` encontrado ANTES do
    proximo `N.` e concatenado (sem separador) a URL da entrada atual —
    resolve tanto varias entradas na mesma linha fisica quanto uma URL
    quebrada em varias linhas pelo conversor de PDF (com ou sem linha em
    branco/ruido de markdown entre os pedacos, incluindo um titulo entre o
    marcador e a URL — ex: "1. Titulo, <u>URL</u>" do Gemini — que
    simplesmente nao casa com nenhuma das duas alternativas do token e por
    isso e ignorado), ja que uma continuacao nunca tem seu proprio numero
    na frente. Espacos literais DENTRO de um unico trecho `<u>` (nunca
    artefato de quebra de linha, que so ocorre ENTRE trechos) sao
    preservados e viram `%20` na URL final."""
    tail = _reference_section_tail(text)
    if tail is None:
        return []

    entries: list[tuple[list[str], str, str]] = []
    current_marker: str | None = None
    current_parts: list[str] = []

    def flush() -> None:
        if current_marker is None or not current_parts:
            return
        joined = "".join(part.replace("\n", "").replace("\r", "").replace("\t", " ") for part in current_parts)
        url = joined.strip().replace(" ", "%20")
        if url:
            entries.append(([f"[{current_marker}]"], "", url))

    for match in _ASTERISM_TOKEN_RE.finditer(tail):
        marker = match.group("marker")
        if marker is not None:
            flush()
            current_marker = marker
            current_parts = []
        else:
            current_parts.append(match.group("url_part"))
    flush()

    return entries


def _find_asterism_bare_url_entries(text: str) -> list[tuple[list[str], str, str]]:
    """Fallback para quando a lista pos-⁂ nao tem nenhuma numeracao (nem
    `N.`, nem `<u>` — formato observado em respostas .docx do Perplexity,
    convertidas via `python-docx`, que nao produzem marcacao nenhuma ao
    redor das URLs): apenas uma URL por linha, em texto puro. Nesse
    formato o marcador `[N]` e a propria ORDEM de ocorrencia (1a URL da
    lista = `[1]`, 2a = `[2]`, ...) — confirmado cruzando a lista com as
    citacoes `[N]` que aparecem no resumo em prosa antes do `⁂`.

    Cada linha nao-vazia contem SO a URL e nada mais (sem titulo, sem
    prosa) — diferente de `_extract_url` (usado onde uma URL pode vir
    seguida de prosa de verdade e por isso so reconecta um unico token
    sem espaco), aqui e seguro pegar a linha inteira a partir do inicio
    da URL, convertendo qualquer espaco literal remanescente (ex: nome de
    arquivo com espaco, tipo "Relatorio Final.pdf") em `%20`."""
    asterism_pos = text.find(_ASTERISM)
    if asterism_pos == -1:
        return []
    tail = text[asterism_pos + 1 :]

    entries: list[tuple[list[str], str, str]] = []
    position = 0
    for line in tail.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _URL_RE.search(stripped)
        if match is None:
            continue
        url = stripped[match.start() :].strip().replace(" ", "%20")
        position += 1
        entries.append(([f"[{position}]"], "", url))

    return entries


def _find_asterism_entries(text: str) -> list[tuple[list[str], str, str]]:
    """Escolhe entre os dois formatos de lista pos-⁂ ja observados: a
    numerada (`_find_asterism_list_entries`, ex: PDF do Perplexity) tem
    prioridade; se ela nao achar nada mas o `⁂` existir, cai para o
    formato de URLs nuas sem numeracao (`_find_asterism_bare_url_entries`,
    ex: DOCX do Perplexity)."""
    numbered = _find_asterism_list_entries(text)
    if numbered:
        return numbered
    return _find_asterism_bare_url_entries(text)


def _find_heading_titled_pairs(text: str) -> list[tuple[list[str], str, str]]:
    """Ultimo fallback: lista de fontes sem NENHUMA marcacao — nem
    colchetes, nem numeracao, nem `<u>`, nem `⁂` — apenas "Titulo, URL"
    por linha, ancorada so pelo cabecalho da secao de referencias
    (`_REFERENCE_SECTION_HEADING`). Observado no Gemini em .docx: o
    `DocxAnswerLoader` ja devolve cada paragrafo intacto (URL como texto
    puro, ja que o hyperlink do Word e resolvido antes deste ponto), sem
    nenhum sinal estrutural alem do proprio cabecalho.

    Nunca ancorado por `⁂` (isso e papel de `_find_asterism_bare_url_entries`)
    — as duas funcoes cobrem a mesma ideia (URL nua, marcador `[N]`
    inferido pela ordem de ocorrencia) para cada uma das duas ancoras
    possiveis, sem se sobrepor, porque so uma delas roda por vez (cadeia
    de prioridade em `RegexReferenceExtractor.extract`).

    Diferente da bare-URL do Perplexity, aqui cada linha TEM um titulo
    antes da URL, separado por virgula/espaco — o texto antes do inicio
    da URL vira `title`."""
    heading_match = _REFERENCE_SECTION_HEADING.search(text)
    if heading_match is None:
        return []
    tail = text[heading_match.end() :]

    entries: list[tuple[list[str], str, str]] = []
    position = 0
    for line in tail.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _URL_RE.search(stripped)
        if match is None:
            continue
        url = stripped[match.start() :].strip()
        title = stripped[: match.start()].rstrip(" ,").strip()
        position += 1
        entries.append(([f"[{position}]"], title, url))

    return entries


def _fuzzy_url_key(url: str) -> str:
    """Chave de comparacao "ignorando" separadores ambiguos (hifen,
    espaco/`%20`) que a conversao PDF->texto pode inserir, remover ou
    trocar num ponto de quebra de linha: percent-decodifica, remove
    hifens/espacos e normaliza para minusculas. Duas URLs que so diferem
    em ONDE/SE ha um separador nesses pontos colapsam para a mesma
    chave; URLs genuinamente diferentes (ex: um link truncado por uma
    tabela markdown corrompida) nao colapsam, porque o conteudo real
    diverge, nao so a pontuacao."""
    return re.sub(r"[-\s]", "", unquote(url)).lower()


_MAX_DROPPED_CHARS = 5


def _is_subsequence(shorter: str, longer: str) -> bool:
    remaining = iter(longer)
    return all(ch in remaining for ch in shorter)


def _best_matching_known_url(url: str, known_urls: set[str]) -> str | None:
    """Melhor candidato entre `known_urls` para reparar `url`, tolerando
    nao so separadores ambiguos (`_fuzzy_url_key`, casamento exato) mas
    tambem LETRAS FALTANDO — corrupcao observada em PDFs cuja fonte
    incorporada tem glifos de ligadura (`fi`, `ffi`, `tt`, ...) sem
    mapeamento ToUnicode completo (ex: "https" -> "htps", "office" ->
    "ofce"). Como o texto perdido nao deixa nenhum sinal de ONDE faltou
    uma letra, uma chave exata nao resolve isso.

    Testei similaridade de string generica (`difflib.SequenceMatcher`)
    primeiro e descartei: uma URL com um caractere EXTRA no final (ex:
    "...pagina" vs "...paginaX", que precisam continuar sendo tratadas
    como paginas diferentes) da uma similaridade tao alta (0.984) quanto
    os casos reais de ligadura (0.978-0.995) — os dois cenarios se
    sobrepoem, um limiar generico nao os separa.

    Em vez disso, modela a corrupcao real com precisao: ligadura so
    REMOVE caracteres, nunca adiciona/substitui. Um candidato so e aceito
    se `url` (chave difusa) for uma SUBSEQUENCIA de `known` (chave
    difusa) — preserva a ordem, so tolera caracteres faltando — e a
    diferenca de tamanho for pequena (`_MAX_DROPPED_CHARS`, com folga
    sobre o pior caso real observado: 4 caracteres perdidos numa unica
    URL longa com varias ligaduras). Isso rejeita corretamente o caso
    "caractere extra" (nunca e subsequencia de uma string MAIS CURTA) sem
    precisar de um limiar ajustado a dedo."""
    key = _fuzzy_url_key(url)
    best_url: str | None = None
    best_drop = _MAX_DROPPED_CHARS + 1
    for known in known_urls:
        known_key = _fuzzy_url_key(known)
        if key == known_key:
            return known
        drop = len(known_key) - len(key)
        if 0 < drop < best_drop and drop <= _MAX_DROPPED_CHARS and _is_subsequence(key, known_key):
            best_url, best_drop = known, drop
    return best_url


def _repair_using_pdf_links(references: list[Reference], known_urls: set[str]) -> list[Reference]:
    """Corrige `raw_url` usando os hyperlinks embutidos no PDF original
    (`extract_pdf_hyperlink_urls`) quando a URL extraida do texto
    convertido e suficientemente parecida (`_best_matching_known_url`) com
    uma URL real do PDF que e diferente dela — tipicamente um hifen
    descartado pelo `pymupdf4llm` num ponto de quebra de linha (ex: "de
    marco" -> "demarco") ou uma letra perdida por ligadura tipografica sem
    mapeamento completo (ex: "https" -> "htps"), nenhum dos dois com sinal
    textual de que algo faltou. `known_urls` vazio (PDF sem hyperlinks,
    `fitz` indisponivel, etc.) faz desta funcao um no-op."""
    if not known_urls:
        return references

    by_id: dict[str, Reference] = {}
    for reference in references:
        fixed_url = _best_matching_known_url(reference.raw_url, known_urls) or reference.raw_url
        if fixed_url == reference.raw_url:
            repaired = reference
        else:
            normalized = normalize_url(fixed_url)
            repaired = reference.model_copy(
                update={
                    "raw_url": fixed_url,
                    "normalized_url": normalized,
                    "id": Reference.id_for_url(normalized),
                }
            )

        existing = by_id.get(repaired.id)
        if existing is not None:
            merged_markers = list(dict.fromkeys(existing.citation_markers + repaired.citation_markers))
            by_id[repaired.id] = existing.model_copy(update={"citation_markers": merged_markers})
        else:
            by_id[repaired.id] = repaired

    return list(by_id.values())


def extract_references(
    text: str,
    *,
    source_answer_id: str,
    tool_name: str,
    strategy: ReferenceExtractionStrategy | None = None,
    answer_path: Path | None = None,
) -> list[Reference]:
    strategy = strategy or RegexReferenceExtractor()
    references = strategy.extract(text, source_answer_id=source_answer_id, tool_name=tool_name)
    if answer_path is not None and answer_path.suffix.lower() == ".pdf":
        references = _repair_using_pdf_links(references, extract_pdf_hyperlink_urls(answer_path))
    return references


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
