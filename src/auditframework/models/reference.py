from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class ReferenceStatus(str, Enum):
    PENDING = "pending"
    DOWNLOADED = "downloaded"
    DEAD = "dead"  # 404
    INACCESSIBLE = "inaccessible"  # 403/timeout/SSL apos esgotar retries
    ERROR = "error"


class Reference(BaseModel):
    """Uma referencia citada por uma resposta de Deep Research.

    `id` e derivado deterministicamente da URL normalizada (hash), nao de
    uma posicao sequencial de extracao — isso corrige o bug do CorpusForge
    em que a mesma URL podia receber ids diferentes entre execucoes por
    depender da ordem, nao-deterministica, de extracao via LLM."""

    id: str
    citation_markers: list[str] = Field(default_factory=list)
    raw_url: str
    normalized_url: str
    work_key: str = ""
    """Identificador da OBRA (o documento), independente do endereco —
    `extraction.url_normalizer.work_key`. Duas referencias com URLs
    diferentes e a mesma `work_key` (PubMed e DOI do mesmo artigo) sao a
    mesma obra citada duas vezes. Fica AO LADO de `normalized_url`, nunca
    no lugar dela: a verificacao de trecho depende do endereco acessado,
    porque cada um expoe uma quantidade diferente do texto."""
    title: str | None = None
    from_secondary_list: bool = False
    """A entrada veio da SEGUNDA lista de fontes do documento — a que o
    Perplexity emite depois do separador `⁂`, reenumerando de 1 a N todas as
    fontes consultadas. A numeracao que o corpo do texto cita e a da lista
    principal (sob o cabecalho `References`), entao, quando a mesma URL
    aparece nas duas, o numero desta e descartado em favor do de la."""
    exporter_artifact: bool = False
    """Entrada injetada pelo exportador de PDF (guia de formatacao de
    citacao, documentacao da propria ferramenta) em vez de fonte do
    conteudo. Fica gravada em `references.json` para que a exclusao seja
    auditavel e contavel, mas nao e baixada nem entra em nenhuma metrica —
    `ReferenceRegistry.load_references()` a omite por padrao."""
    status: ReferenceStatus = ReferenceStatus.PENDING
    http_status: int | None = None
    fetched_at: datetime | None = None
    error_message: str | None = None
    source_answer_id: str
    tool_name: str

    @staticmethod
    def id_for_url(normalized_url: str) -> str:
        return hashlib.sha256(normalized_url.encode("utf-8")).hexdigest()[:16]
