from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple, Protocol
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, Field

from ..common.errors import LLMParseError
from ..common.llm_client import LLMClient
from ..logging_config import get_logger
from ..models import Reference
from .pdf_links import extract_pdf_hyperlink_urls
from .url_normalizer import normalize_url, work_key

logger = get_logger(__name__)

_MARKER_RUN = re.compile(r"(?:\[\s*(\d+)\s*\]\s*)+")

# Parenteses NAO sao excluidos daqui: URLs reais os contem com frequencia
# (PII do Lancet, "PIIS0140-6736(21)02178-4"; artigo da Wikipedia com
# desambiguacao, "Foo_(disambiguation)"; PDF com "(1)" no nome). Um `)` que
# na verdade fecha markdown/prosa em volta da URL e removido depois por
# `_balance_parens`, contando abre/fecha — nunca por exclusao no regex, que
# cortaria a URL bem antes do `)` que pertence a ela. Cada URL truncada
# assim virava uma "obra" falsamente exclusiva na analise de intersecao.
_URL_RE = re.compile(r'https?://[^\s<>\[\]"]+')

# Pontuacao de prosa que pode encostar no fim de uma URL. Sem `)` — esse e
# decidido por `_balance_parens`, nao por remocao cega.
_URL_TRAILING_JUNK = ".,;:”\"'）]>»"


def _balance_parens(url: str) -> str:
    """Remove `)` finais que fecham parenteses de FORA da URL (prosa ou
    markdown em volta dela, ex: "(ver https://x.com/a)"), preservando os
    que pertencem a URL de verdade — esses ficam balanceados e a contagem
    para de remover."""
    while url.endswith(")") and url.count("(") < url.count(")"):
        url = url[:-1]
    return url


def _clean_url(url: str) -> str:
    """Tira pontuacao de prosa colada ao fim da URL, alternando com o
    balanceamento de parenteses ate estabilizar (ex: `...artigo).` precisa
    de duas passadas)."""
    previous = None
    while url != previous:
        previous = url
        url = url.rstrip(_URL_TRAILING_JUNK)
        url = _balance_parens(url)
    return url


_REFERENCE_SECTION_HEADING = re.compile(
    r"^(?:#{1,6}[ \t]*\**[ \t]*|\*\*[ \t]*)(refer[eê]ncias|references|fontes|sources|bibliografia)\b",
    re.IGNORECASE | re.MULTILINE,
)
"""Cabecalho de secao de lista de fontes. Tolera negrito markdown ao redor
da palavra-chave e texto extra depois dela (ex: "### **Referencias
citadas**" do Gemini em PDF) — nao exige que a linha seja *so* a
palavra-chave, so que comece com ela.

Aceita tambem o cabecalho marcado APENAS por negrito, sem `#` (`**References**`,
do Grok; `**Fontes**`, do ChatGPT). Sem isso a lista de fontes nao era
reconhecida e ficava dentro do corpo, com cada entrada disputando espaco com
as afirmacoes. Exigir `#` OU `**` no inicio da linha e o que separa um
cabecalho de uma frase que por acaso comece com "References"."""

_BARE_REFERENCE_HEADING = re.compile(
    r"^[ \t]*(refer[eê]ncias|references|fontes|sources|bibliografia)\b[^.!?\n]{0,40}$",
    re.IGNORECASE | re.MULTILINE,
)
"""Cabecalho de lista de fontes que chegou SEM marcacao nenhuma — nem `#`,
nem negrito. Acontece na conversao de `.docx`, que perde o estilo do
paragrafo e entrega `Referências Completas` como linha comum.

Sozinho este padrao e perigoso: ele casa tambem um titulo de secao do CORPO
da resposta ("Fontes oficiais", "Referências completas" no eco do prompt), e
cortar ali descartaria a resposta inteira — 239 afirmacoes num arquivo real.
Por isso todo candidato daqui passa por `_dominated_by_urls`; os com `#` ou
`**` nao precisam, porque a marcacao ja e a evidencia."""

# Uma lista de fontes e quase toda URL; prosa quase nunca tem. Os limites
# separam com folga os casos reais medidos no corpus: 0.08 nos dois titulos
# de secao do corpo, 0.65 e 0.81 nas duas listas de fontes.
#
# Nao unificar isso aplicando a densidade tambem ao cabecalho marcado: ela
# REPROVA tres listas de fontes reais, porque densidade mede formato, nao
# natureza — 0.32 num `.md` cujas entradas ocupam varias linhas, e 2 URLs
# (abaixo do minimo) numa lista legitimamente pequena.
_MIN_LIST_URLS = 3
_MIN_URL_LINE_RATIO = 0.4


def _dominated_by_urls(tail: str) -> bool:
    """`True` se `tail` tem cara de lista de fontes: varias URLs, e elas
    dominam as linhas nao vazias."""
    lines = [stripped for stripped in (line.strip() for line in tail.splitlines()) if stripped]
    urls = sum(1 for line in lines if "://" in line)
    return urls >= _MIN_LIST_URLS and urls >= len(lines) * _MIN_URL_LINE_RATIO


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
    return _clean_url(url), match.start()


# Prioridade da lista de onde a entrada veio, quando o documento tem MAIS DE
# UMA lista de fontes. O Perplexity emite duas: uma ancorada pelo cabecalho
# (`## References`), cuja numeracao e a que o corpo do texto cita, e outra
# depois do separador `⁂`, que enumera todas as fontes consultadas de 1 a N
# de novo. A mesma URL aparece nas duas sob numeros diferentes; sem
# prioridade, a referencia acumulava os dois numeros e o relatorio mostrava
# "cita [1] [26]" para um trecho que cita so `[1]`.
_PRIORITY_PRIMARY = 0
_PRIORITY_SECONDARY = 1


class _Entry(NamedTuple):
    """Uma entrada da lista de fontes, antes de virar `Reference`."""

    markers: list[str]
    title: str
    url: str
    priority: int = _PRIORITY_PRIMARY


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
        priority_by_id: dict[str, int] = {}
        entries = _find_reference_entries(text) + _find_asterism_entries(text)
        if not entries:
            entries = _find_heading_titled_pairs(text)
        for entry in entries:
            normalized = normalize_url(entry.url)
            ref_id = Reference.id_for_url(normalized)
            existing = by_id.get(ref_id)
            if existing is None:
                by_id[ref_id] = Reference(
                    id=ref_id,
                    citation_markers=entry.markers,
                    raw_url=entry.url,
                    normalized_url=normalized,
                    work_key=work_key(entry.url),
                    from_secondary_list=entry.priority == _PRIORITY_SECONDARY,
                    title=entry.title or None,
                    source_answer_id=source_answer_id,
                    tool_name=tool_name,
                )
                priority_by_id[ref_id] = entry.priority
                continue

            # A mesma URL de novo. Vinda de uma lista de prioridade MENOR (a
            # do `⁂`), o numero dela e descartado: quem cita no corpo e a
            # numeracao da lista principal. Da MESMA lista, os numeros se
            # somam — e o caso legitimo de "[1] [3] [7] Titulo" apontando
            # para uma URL so, e o corpo pode citar qualquer um deles.
            best = priority_by_id[ref_id]
            if entry.priority > best:
                logger.info(
                    "numero(s) %s da lista secundaria de fontes descartado(s): %s ja esta listada como %s",
                    ", ".join(entry.markers) or "(sem numero)",
                    entry.url,
                    ", ".join(existing.citation_markers) or "(sem numero)",
                )
                continue
            if entry.priority < best:
                update = {"citation_markers": entry.markers, "from_secondary_list": False}
                priority_by_id[ref_id] = entry.priority
            else:
                update = {"citation_markers": list(dict.fromkeys(existing.citation_markers + entry.markers))}
            by_id[ref_id] = existing.model_copy(update=update)
        return list(by_id.values())


def _find_reference_entries(text: str) -> list[_Entry]:
    entries: list[_Entry] = []
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
            entries.append(_Entry(markers, title, url))
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
            entries.append(_Entry(markers, " ".join(title_parts).strip(), found_url))

    return entries


def _only_reference_lines(chunk: str) -> bool:
    """`True` se `chunk` contem apenas linhas que carregam URL, cabecalhos e
    o separador `⁂` — nenhuma linha de prosa comum.

    Exige URL na linha, nao so um prefixo de entrada (`1.` / `[1]`): uma
    lista numerada de resumo no corpo da resposta ("1. **Fonte A** - resumo
    qualquer") tem exatamente a mesma forma de uma entrada de referencia
    sem URL, e confundir as duas arrastaria o resumo para dentro da lista de
    fontes. Entrada real cuja URL esta na linha SEGUINTE tambem bloqueia o
    recuo — conservador de proposito: na duvida, mantem o corte na ancora
    mais proxima da lista."""
    for line in chunk.splitlines():
        stripped = line.strip()
        if not stripped or _ASTERISM in stripped or stripped.startswith("#"):
            continue
        if "://" in stripped or not stripped.strip("|").strip():  # URL, ou pipe-arte vazia
            continue
        return False
    return True


def find_reference_section(text: str) -> tuple[int, int] | None:
    """`(inicio da ancora, inicio do conteudo depois dela)` para onde a
    lista de fontes comeca, ou `None` se nenhuma ancora existir (documento
    em formato desconhecido).

    Ancoras: o separador `⁂` (Perplexity) e o cabecalho da secao de
    referencias (`_REFERENCE_SECTION_HEADING`, ex: ChatGPT/Gemini). So
    conta ancora que tenha URL depois dela — testado por `://`, nao por
    `http`, porque uma URL corrompida por ligadura tipografica do PDF chega
    aqui como `htps://` e ainda assim e a lista de fontes.

    Parte da ULTIMA ancora — importa quando um cabecalho "Referências"
    existe cedo no documento mas nao e o inicio real da lista — e RECUA
    para uma ancora anterior enquanto tudo entre as duas for lista de
    fontes, para que a leitura da lista inclua a primeira delas quando as
    duas sao contiguas (visto no Perplexity: uma lista em prosa seguida de
    outra, numerada, depois do `⁂`).

    NAO serve para decidir onde o corpo auditavel termina — para isso existe
    `body_section_start`, que corta mais cedo. Esta ancora e a que os
    parsers de lista usam, e precisa ser conservadora: recuar aqui sem que
    as duas listas sejam contiguas faz o parser do `⁂` varrer entradas de
    outro formato e fabricar referencias lixo (medido: 9 URLs `https:///[24][4]`
    num PDF cuja lista principal traz marcador, nao URL, dentro do `<u>`)."""
    anchors: list[tuple[int, int]] = [
        (match.start(), match.end()) for match in _REFERENCE_SECTION_HEADING.finditer(text)
    ]
    anchors += [(match.start(), match.end()) for match in re.finditer(_ASTERISM, text)]
    anchors = sorted(anchor for anchor in anchors if "://" in text[anchor[0] :])
    if not anchors:
        return None

    chosen = anchors[-1]
    for earlier in reversed(anchors[:-1]):
        if not _only_reference_lines(text[earlier[1] : chosen[0]]):
            break
        chosen = earlier
    return chosen


def body_section_start(text: str) -> int | None:
    """Onde o corpo auditavel termina, ou `None` se o documento nao tem
    lista de fontes reconhecivel.

    Deliberadamente MAIS CEDO que `find_reference_section`: nada depois de
    um cabecalho de referencias e uma afirmacao a auditar, ainda que a lista
    que o segue nao tenha o formato que os parsers reconhecem. O corte da
    ancora de leitura falhava exatamente ai — ele exige que TODA linha entre
    um cabecalho e a lista final carregue URL, e o conversor de PDF empurra
    a URL de uma entrada para a linha seguinte. O resultado eram entradas da
    lista virando afirmacoes julgadas: 59 num corpus de 27 arquivos, entre
    elas fragmentos de data de bibliografia ("Mar 2026.") indo ao juiz LLM.

    Os dois cortes ficam separados de proposito; a invariante e
    `body_section_start <= find_reference_section`, nao igualdade."""
    candidates = [
        match.start()
        for match in _REFERENCE_SECTION_HEADING.finditer(text)
        if "://" in text[match.start() :]
    ]
    candidates += [
        match.start()
        for match in _BARE_REFERENCE_HEADING.finditer(text)
        if _dominated_by_urls(text[match.start() :])
    ]
    anchor = find_reference_section(text)
    if anchor is not None:
        candidates.append(anchor[0])
    return min(candidates) if candidates else None


def _reference_section_tail(text: str) -> str | None:
    anchor = find_reference_section(text)
    return None if anchor is None else text[anchor[1] :]


def _find_asterism_list_entries(text: str) -> list[_Entry]:
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

    # Quando o trecho comeca no cabecalho e o `⁂` aparece mais adiante, ele
    # separa duas listas: a de cima e a numeracao que o corpo cita
    # (prioridade maior). `-1` (o proprio `⁂` era a ancora, ou nao existe)
    # significa lista unica — tudo com a mesma prioridade, e o merge volta a
    # somar os numeros como antes.
    asterism_pos = tail.find(_ASTERISM)

    entries: list[_Entry] = []
    current_marker: str | None = None
    current_parts: list[str] = []
    current_priority = _PRIORITY_PRIMARY

    def flush() -> None:
        if current_marker is None or not current_parts:
            return
        joined = "".join(part.replace("\n", "").replace("\r", "").replace("\t", " ") for part in current_parts)
        url = joined.strip().replace(" ", "%20")
        if url:
            entries.append(_Entry([f"[{current_marker}]"], "", url, current_priority))

    for match in _ASTERISM_TOKEN_RE.finditer(tail):
        marker = match.group("marker")
        if marker is not None:
            flush()
            current_marker = marker
            current_parts = []
            current_priority = (
                _PRIORITY_SECONDARY if 0 <= asterism_pos <= match.start() else _PRIORITY_PRIMARY
            )
        else:
            current_parts.append(match.group("url_part"))
    flush()

    return entries


def _find_asterism_bare_url_entries(text: str) -> list[_Entry]:
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

    entries: list[_Entry] = []
    position = 0
    for line in tail.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _URL_RE.search(stripped)
        if match is None:
            continue
        url = _clean_url(stripped[match.start() :].strip().replace(" ", "%20"))
        position += 1
        entries.append(_Entry([f"[{position}]"], "", url))

    return entries


def _find_asterism_entries(text: str) -> list[_Entry]:
    """Escolhe entre os dois formatos de lista pos-⁂ ja observados: a
    numerada (`_find_asterism_list_entries`, ex: PDF do Perplexity) tem
    prioridade; se ela nao achar nada mas o `⁂` existir, cai para o
    formato de URLs nuas sem numeracao (`_find_asterism_bare_url_entries`,
    ex: DOCX do Perplexity)."""
    numbered = _find_asterism_list_entries(text)
    if numbered:
        return numbered
    return _find_asterism_bare_url_entries(text)


def _find_heading_titled_pairs(text: str) -> list[_Entry]:
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

    entries: list[_Entry] = []
    position = 0
    for line in tail.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _URL_RE.search(stripped)
        if match is None:
            continue
        url = _clean_url(stripped[match.start() :].strip())
        title = stripped[: match.start()].rstrip(" ,").strip()
        position += 1
        entries.append(_Entry([f"[{position}]"], title, url))

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
    # `known_urls` e um set: ordena para que, havendo mais de um candidato
    # aceitavel, o reparo seja deterministico entre execucoes.
    candidates = sorted(known_urls)

    for known in candidates:
        if key == _fuzzy_url_key(known):
            return known

    # Prefixo: a URL extraida e um prefixo de uma URL embutida e perdeu o
    # final numa quebra de linha — o ULTIMO segmento inteiro
    # (".../businesses-and-occupations/" sem "samsung-electronics-co-ltd") ou
    # o fim de um segmento no MEIO dele (".../making-the-case-for-a-four" sem
    # "-day-working-week/"). O segundo caso nao deixa sinal nenhum de
    # corrupcao e perde caracteres demais para o orcamento de ligadura.
    #
    # Aceita continuacao de no maximo UM segmento: varios segmentos a mais
    # sao outra pagina, mais profunda, nao a mesma truncada. Exige unicidade,
    # porque "/artigo" e prefixo legitimo de "/artigo-2" — havendo dois
    # candidatos nada e reparado. Exige path alem da raiz, para nunca
    # transformar um host nu numa pagina qualquer dele.
    if urlsplit(url).path.strip("/"):
        prefixed = {
            base
            for base in (known.split("#", 1)[0] for known in candidates)
            if base != url and base.startswith(url) and "/" not in base[len(url) :].strip("/")
        }
        if len(prefixed) == 1:
            return prefixed.pop()

    best_url: str | None = None
    best_drop = _MAX_DROPPED_CHARS + 1
    for known in candidates:
        # Compara sem o fragmento: um reparo nunca deve ACRESCENTAR um
        # `#:~:text=` (destaque que o navegador poe na URL copiada) a uma
        # URL que o documento cita sem ele.
        base = known.split("#", 1)[0]
        drop = len(_fuzzy_url_key(base)) - len(key)
        if 0 < drop < best_drop and _is_subsequence(key, _fuzzy_url_key(base)):
            best_url, best_drop = base, drop
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
                    "work_key": work_key(fixed_url),
                    "id": Reference.id_for_url(normalized),
                }
            )

        existing = by_id.get(repaired.id)
        if existing is None:
            by_id[repaired.id] = repaired
            continue

        # Duas entradas que so viraram a MESMA URL depois do reparo. Acontece
        # porque o conversor corrompeu cada ocorrencia de um jeito diferente
        # ("firmsin" numa, "bigg est" na outra), entao elas passaram pela
        # deduplicacao de `extract` como URLs distintas. Vale a mesma
        # prioridade daquela etapa, agora lida de `from_secondary_list`:
        # entre listas diferentes, fica o numero da principal (a numeracao
        # que o corpo cita); da mesma lista, os numeros se somam, porque o
        # corpo pode citar qualquer um deles.
        if existing.from_secondary_list == repaired.from_secondary_list:
            merged = list(dict.fromkeys(existing.citation_markers + repaired.citation_markers))
            by_id[repaired.id] = existing.model_copy(update={"citation_markers": merged})
            continue
        principal, secondary = (
            (existing, repaired) if repaired.from_secondary_list else (repaired, existing)
        )
        logger.info(
            "numero(s) %s da lista secundaria descartado(s): apos o reparo pelos hyperlinks do PDF, "
            "%s e a mesma URL listada como %s",
            ", ".join(secondary.citation_markers) or "(sem numero)",
            repaired.raw_url,
            ", ".join(principal.citation_markers) or "(sem numero)",
        )
        by_id[repaired.id] = principal.model_copy(update={"from_secondary_list": False})

    return list(by_id.values())


# Referencia injetada pelo exportador de PDF: guia de como formatar citacao
# academica, documentacao da propria ferramenta, forum sobre como pedir
# citacoes a uma IA — nao e fonte do conteudo do relatorio. Observado no PDF
# do Perplexity: uma lista real curta seguida de dezenas de entradas
# numeradas, em formato identico as reais, mas sobre COMO citar. Nao tem
# posicao fixa em relacao ao `⁂`, entao o filtro e por conteudo da URL, nao
# por posicao no bloco. Lista aberta — caso novo encontrado no corpus entra
# aqui, com o arquivo/ferramenta onde apareceu.
_CONTAMINATION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"scribbr\.com",
        r"guides\.lib\.",
        r"/writing-guide/",
        r"numbered_citation_style",
        r"rag-citations",
        r"citing-references-in-text",
        r"in-text-citation",
        r"cse8-sequence-in-text",
        r"how-can-i-get-ai-to-give-factual-answers",  # Workday/Perplexity run 2
        r"docs\.perplexity\.ai",
        # Perplexity/four-day-week run 1: asme.org "elements-of-a-paper".
        r"/author-guidelines/",
        r"elements-of-a-paper",
    )
]


def _drop_colliding_markers(references: list[Reference]) -> list[Reference]:
    """Garante que cada marcador `[N]` resolva para UMA referencia.

    Uma resposta do Perplexity ja observada traz duas listas de fontes — uma
    sob o cabecalho e outra depois do `⁂` — ambas numeradas a partir de 1.
    As duas sao lidas (a de cima e a que o corpo cita; a de baixo traz
    fontes que so aparecem nela), mas as numeracoes colidem: `[3]` existia
    em duas referencias, e `_build_marker_index` resolvia pela ultima,
    dependendo da ordem do dict. O relatorio ficava autocontraditorio —
    "cita [3], corroborado por [3]".

    Mantem a primeira reivindicacao em ordem de documento e tira o marcador
    das seguintes. A referencia em si nao e descartada: ela continua
    listada, e sem marcador aparece como "listada mas nunca citada", que e
    exatamente o que se pode afirmar sobre ela.

    Artefato do exportador resolve depois de todo mundo: ele nao tem posicao
    fixa no documento e pode aparecer ANTES de uma fonte real que use o
    mesmo numero — sem isso, um guia de formatacao de citacao roubaria o
    numero de uma fonte de verdade so por vir primeiro."""
    claimed: dict[str, str] = {}
    kept_by_id: dict[str, list[str]] = {}
    # `sorted` e estavel: fontes reais na ordem do documento, artefatos depois.
    for reference in sorted(references, key=lambda r: r.exporter_artifact):
        kept = [m for m in reference.citation_markers if claimed.setdefault(m, reference.id) == reference.id]
        kept_by_id[reference.id] = kept
        dropped = [m for m in reference.citation_markers if m not in kept]
        if dropped:
            logger.warning(
                "marcador(es) %s de %s ja pertencem a outra referencia da lista (numeracao repetida "
                "em duas listas de fontes) — mantida a primeira ocorrencia",
                ", ".join(dropped),
                reference.raw_url,
            )
    # devolve na ordem original: quem consome `references.json` espera a
    # ordem do documento, nao a da resolucao de conflito
    return [
        reference if kept_by_id[reference.id] == reference.citation_markers
        else reference.model_copy(update={"citation_markers": kept_by_id[reference.id]})
        for reference in references
    ]


def _warn_duplicate_works(references: list[Reference]) -> None:
    """Avisa quando a MESMA obra foi listada sob marcadores diferentes (ex:
    o mesmo artigo pelo PubMed e pelo DOI). Nao deduplica: os dois
    enderecos continuam sendo verificados a parte, porque cada um expoe uma
    quantidade diferente do texto. O aviso importa porque duas entradas da
    mesma obra citadas lado a lado (`[3] [4]`) produzem aparencia de
    corroboracao independente a partir de uma fonte so."""
    by_work: dict[str, list[Reference]] = defaultdict(list)
    for reference in references:
        if reference.work_key:
            by_work[reference.work_key].append(reference)
    for key, group in by_work.items():
        if len(group) > 1:
            logger.warning(
                "obra %s listada %d vezes sob marcadores diferentes (%s) — corroboracao "
                "aparente pode vir de uma fonte so",
                key,
                len(group),
                ", ".join(m for r in group for m in r.citation_markers) or "sem marcador",
            )


def _mark_exporter_contamination(references: list[Reference]) -> list[Reference]:
    """Marca (nao remove) as entradas injetadas pelo exportador.

    Apagar tornava a exclusao inauditavel: nao havia como contar quantas
    foram nem verificar que o filtro nao comeu uma fonte legitima — e os
    padroes sao genericos o bastante para isso importar (um relatorio CUJO
    TEMA seja citacao teria guias de citacao como fonte de verdade).
    Marcadas, ficam no `references.json`, fora da ingestao e das metricas, e
    viram um numero que o relatorio reporta."""
    out: list[Reference] = []
    for reference in references:
        if any(p.search(reference.raw_url) for p in _CONTAMINATION_PATTERNS):
            logger.warning(
                "referencia %s marcada como artefato do exportador (guia de citacao/documentacao, "
                "nao fonte do conteudo): %s — nao sera baixada nem auditada",
                reference.citation_markers or reference.id,
                reference.raw_url,
            )
            reference = reference.model_copy(update={"exporter_artifact": True})
        out.append(reference)
    return out


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
    references = _mark_exporter_contamination(references)
    references = _drop_colliding_markers(references)
    _warn_duplicate_works([r for r in references if not r.exporter_artifact])
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
                work_key=work_key(item.url),
                title=item.title,
                source_answer_id=source_answer_id,
                tool_name=tool_name,
            )
        return list(by_id.values())
