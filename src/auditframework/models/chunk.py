from __future__ import annotations

from pydantic import BaseModel, Field


class AnswerChunk(BaseModel):
    """Uma afirmacao da resposta de Deep Research com a citacao que a
    ancora — a sentenca em que um marcador `[N]` aparece.

    Todo chunk carrega pelo menos um marcador: texto sem citacao nao gera
    chunk, porque a auditoria verifica a relacao entre uma afirmacao e a
    fonte que ela cita, e sem citacao nao existe essa relacao.
    `cited_reference_ids` ainda pode sair vazio quando o marcador nao tem
    entrada na lista de referencias — o trecho existe, a fonte nao."""

    id: str
    answer_id: str
    position: int
    text: str
    cited_markers: list[str] = Field(default_factory=list)
    """Os marcadores como aparecem NESTE trecho (`["[8]"]`), na ordem. Uma
    referencia pode estar listada sob varios numeros (`[8]` e `[10]` para a
    mesma URL), entao `cited_reference_ids` nao permite reconstituir qual
    numero o trecho usou — e o texto do chunk termina onde o `[` comeca, de
    proposito. Sem isto o relatorio exibia o primeiro numero da referencia,
    nao o citado. Vazio em runs gravadas antes deste campo."""
    cited_reference_ids: list[str] = Field(default_factory=list)


class ReferenceChunk(BaseModel):
    """Um trecho semanticamente coerente de um Document indexado."""

    id: str
    reference_id: str
    section: str | None = None
    text: str
    token_count: int
    embedding_id: int
