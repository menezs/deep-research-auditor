from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from tqdm import tqdm

from .common.errors import ConfigurationError, LLMParseError
from .config import Settings
from .extraction.loaders import load_answer
from .extraction.reference_extractor import ReferenceExtractionStrategy, extract_references
from .common.llm_client import LLMClient, create_llm_client
from .indexing.chunkers import AnswerChunker, DocumentChunker
from .indexing.embeddings import BGEEmbedder, Embedder
from .indexing.reranker import Reranker
from .indexing.retriever import Retriever
from .indexing.service import build_index
from .indexing.vector_store import FaissVectorStore
from .ingestion.service import Fetcher, ingest_references
from .ingestion.registry import ReferenceRegistry
from .judging.judge import Verifier
from .logging_config import get_logger
from .models import AnswerChunk, AuditResult, CuratedDocument, SkippedChunk
from .reporting.aggregator import aggregate_report
from .reporting.render import render_json, render_markdown

logger = get_logger(__name__)

_REFERENCE_SECTION_HEADING = re.compile(
    r"^#{1,6}\s*(refer[eê]ncias|references|fontes|sources|bibliografia)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _strip_reference_section(text: str) -> str:
    """Remove a secao final de lista de fontes antes do chunking da
    resposta — sem isso, `AnswerChunker` trataria cada entrada da lista
    de referencias (que tambem comeca com um marcador `[N]`) como se
    fosse uma alegacao a ser julgada, desperdicando chamadas de LLM e
    poluindo o relatorio com "chunks" que sao apenas citacoes."""
    match = _REFERENCE_SECTION_HEADING.search(text)
    if match is None:
        return text
    return text[: match.start()].rstrip()


class PipelineStage(Protocol):
    """Contrato comum a todo estagio do pipeline.

    Cada estagio concreto recebe suas dependencias via construtor
    (Dependency Injection) e opera exclusivamente sobre os artefatos
    persistidos em `ctx.run_dir` — nunca em memoria entre estagios —
    para que `audit resume` possa pular estagios ja concluidos sem
    manter nenhum processo vivo entre execucoes."""

    name: str

    def run(self, ctx: "RunContext") -> None: ...


@dataclass
class RunContext:
    """Estado de uma execucao: `run_id` + Settings + metadados de entrada.

    Substitui a pratica atual de espalhar paths/config por variaveis
    globais e argumentos de CLI duplicados em cada um dos tres
    repositorios."""

    run_id: str
    settings: Settings
    answer_path: Path
    tool_name: str
    full_corpus_mode: bool = False
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    stages_completed: list[str] = field(default_factory=list)

    @property
    def run_dir(self) -> Path:
        return self.settings.run_dir(self.run_id)


def _meta_path(run_dir: Path) -> Path:
    return run_dir / "run_meta.json"


def save_run_meta(ctx: RunContext) -> None:
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "answer_path": str(ctx.answer_path),
        "tool_name": ctx.tool_name,
        "full_corpus_mode": ctx.full_corpus_mode,
        "started_at": ctx.started_at,
    }
    _meta_path(ctx.run_dir).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_run_context(run_id: str, settings: Settings) -> RunContext:
    """Reconstroi o `RunContext` de uma execucao ja iniciada, a partir do
    `run_meta.json` gravado por `audit run` — usado por `audit resume` e
    `audit report`, que so recebem o `run_id` na linha de comando."""
    meta_path = _meta_path(settings.run_dir(run_id))
    if not meta_path.exists():
        raise ConfigurationError(f"Run {run_id!r} nao encontrada (nenhum run_meta.json em {meta_path.parent})")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return RunContext(
        run_id=run_id,
        settings=settings,
        answer_path=Path(meta["answer_path"]),
        tool_name=meta["tool_name"],
        full_corpus_mode=meta.get("full_corpus_mode", False),
        started_at=meta["started_at"],
    )


def _stage_state_path(run_dir: Path) -> Path:
    return run_dir / "state.json"


def _load_stage_state(run_dir: Path) -> list[str]:
    path = _stage_state_path(run_dir)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["stages_completed"]


def _save_stage_state(run_dir: Path, stages_completed: list[str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _stage_state_path(run_dir).write_text(
        json.dumps({"stages_completed": stages_completed}, indent=2), encoding="utf-8"
    )


def _save_answer_chunks(run_dir: Path, chunks: list[AnswerChunk]) -> None:
    payload = [json.loads(chunk.model_dump_json()) for chunk in chunks]
    (run_dir / "answer_chunks.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_answer_chunks(run_dir: Path) -> list[AnswerChunk]:
    raw = json.loads((run_dir / "answer_chunks.json").read_text(encoding="utf-8"))
    return [AnswerChunk.model_validate(item) for item in raw]


def load_audit_results(path: Path) -> list[AuditResult]:
    if not path.exists():
        return []
    return [AuditResult.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_skipped_chunks(path: Path) -> list[SkippedChunk]:
    if not path.exists():
        return []
    return [SkippedChunk.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class ExtractionStage:
    """Resposta bruta (md/pdf/docx) -> `list[Reference]` + `list[AnswerChunk]`.

    Ex-CorpusForge (extracao) + a parte de chunking da resposta que
    depende dos ids de referencia ja resolvidos. Cacheia o texto extraido
    e uma copia do arquivo original em `run_dir/input/`, para que um
    `audit resume` nao dependa do arquivo de entrada continuar disponivel
    no mesmo path caso a extracao ja tenha sido concluida."""

    name = "extraction"

    def __init__(self, extractor: ReferenceExtractionStrategy | None = None):
        self.extractor = extractor

    def run(self, ctx: RunContext) -> None:
        text = load_answer(ctx.answer_path)

        input_dir = ctx.run_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / "answer.md").write_text(text, encoding="utf-8")
        shutil.copy2(ctx.answer_path, input_dir / f"original{ctx.answer_path.suffix}")

        references = extract_references(
            text,
            source_answer_id=ctx.run_id,
            tool_name=ctx.tool_name,
            strategy=self.extractor,
            answer_path=ctx.answer_path,
        )
        ReferenceRegistry(ctx.run_dir).save_references(references)
        logger.info("Extraidas %d referencias de %s", len(references), ctx.answer_path)

        body = _strip_reference_section(text)
        chunks = AnswerChunker().chunk(body, answer_id=ctx.run_id, references=references)
        _save_answer_chunks(ctx.run_dir, chunks)
        logger.info("Resposta dividida em %d chunks", len(chunks))


class IngestionStage:
    """`list[Reference]` -> `list[Document]` baixado e convertido para
    Markdown, atualizando o status de cada referencia. Ex-CorpusForge
    (download)."""

    name = "ingestion"

    def __init__(self, fetcher: Fetcher | None = None, max_workers: int = 4):
        self.fetcher = fetcher
        self.max_workers = max_workers

    def run(self, ctx: RunContext) -> None:
        registry = ReferenceRegistry(ctx.run_dir)
        references = registry.load_references()
        _, documents = ingest_references(references, registry, max_workers=self.max_workers, fetcher=self.fetcher)
        logger.info("Ingestao concluida: %d documentos disponiveis", len(documents))


class IndexingStage:
    """`list[Document]` -> indice FAISS de `ReferenceChunk`s. Ex-syntex
    (chunking de documentos + embeddings + indexacao)."""

    name = "indexing"

    def __init__(self, embedder: Embedder, chunker: DocumentChunker | None = None):
        self.embedder = embedder
        self.chunker = chunker

    def run(self, ctx: RunContext) -> None:
        registry = ReferenceRegistry(ctx.run_dir)
        references = registry.load_references()
        documents = [registry.load_document(ref.id) for ref in references if registry.has_document(ref.id)]

        def read_markdown(reference_id: str) -> str:
            return registry.document_path(reference_id).read_text(encoding="utf-8")

        store = build_index(documents, read_markdown, self.embedder, self.chunker)
        index_dir = ctx.run_dir / "index"
        store.save(index_dir / "faiss.index", index_dir / "chunks.json")
        logger.info("Indice construido a partir de %d documentos", len(documents))


class JudgingStage:
    """Para cada `AnswerChunk`: recupera o contexto curado (escopado pela
    referencia citada) e submete ao juiz LLM. Ex-audit_with_llm.

    Duas fases sequenciais, deliberadamente separadas: (1) recupera e
    persiste em disco o `CuratedDocument` de TODOS os chunks pendentes
    (busca vetorial + rerank, sem chamada de LLM) e so depois (2) julga
    cada chunk em serie, um por vez — nunca em paralelo — usando o
    contexto ja persistido na fase 1. Isso garante que os `curated_*.json`
    de todos os chunks existam antes da primeira chamada ao LLM juiz, e
    que uma interrupcao durante o julgamento (fase 2, lenta, uma chamada
    de rede por chunk) nunca perca o trabalho de retrieval (fase 1,
    rapido, local) ja feito.

    Resumivel por chunk: `audit_results.jsonl` e `skipped_chunks.jsonl` sao
    append-only, entao um `audit resume` so processa os chunks que ainda
    nao tem resultado nem registro de skip persistido, mesmo que o
    processo tenha sido interrompido no meio; a fase 1 tambem pula chunks
    cujo `curated_*.json` ja existe em disco. Um chunk cuja saida do LLM
    juiz nao pode ser interpretada (JSON invalido/fora do schema) e pulado
    com um aviso, sem derrubar o resto do run — fica pendente e e
    retentado numa proxima `audit resume`. Ja um chunk sem evidencia citada
    disponivel (`CuratedDocument.skip_reason`) nunca chega a chamar o LLM —
    e registrado em `skipped_chunks.jsonl` com a justificativa e nao e
    reprocessado."""

    name = "judging"

    def __init__(
        self,
        llm_client: LLMClient,
        embedder: Embedder,
        reranker: Reranker | None = None,
        top_k: int = 50,
        rerank_top_k: int = 20,
        full_corpus_mode: bool = False,
    ):
        self.llm_client = llm_client
        self.embedder = embedder
        self.reranker = reranker
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.full_corpus_mode = full_corpus_mode

    def run(self, ctx: RunContext) -> None:
        chunks = _load_answer_chunks(ctx.run_dir)
        index_dir = ctx.run_dir / "index"
        store = FaissVectorStore.load(self.embedder.dimension, index_dir / "faiss.index", index_dir / "chunks.json")
        retriever = Retriever(
            self.embedder, store, self.reranker, self.top_k, self.rerank_top_k, full_corpus_mode=self.full_corpus_mode
        )
        verifier = Verifier(self.llm_client)

        results_path = ctx.run_dir / "audit_results.jsonl"
        skipped_path = ctx.run_dir / "skipped_chunks.jsonl"
        curated_dir = ctx.run_dir / "curated"
        curated_dir.mkdir(parents=True, exist_ok=True)
        already_judged = {r.answer_chunk_id for r in load_audit_results(results_path)}
        already_skipped = {s.answer_chunk_id for s in load_skipped_chunks(skipped_path)}

        pending = [c for c in chunks if c.id not in already_judged and c.id not in already_skipped]
        logger.info(
            "Julgamento: %d chunks, %d ja julgados, %d ja pulados, %d pendentes",
            len(chunks),
            len(already_judged),
            len(already_skipped),
            len(pending),
        )

        curated_by_chunk: dict[str, CuratedDocument] = {}
        for chunk in tqdm(pending, desc="Recuperando contexto", unit="chunk"):
            curated_path = curated_dir / f"curated_{chunk.id}.json"
            if curated_path.exists():
                curated_by_chunk[chunk.id] = CuratedDocument.model_validate_json(
                    curated_path.read_text(encoding="utf-8")
                )
                continue
            curated = retriever.retrieve(chunk)
            curated_path.write_text(curated.model_dump_json(indent=2), encoding="utf-8")
            curated_by_chunk[chunk.id] = curated

        with results_path.open("a", encoding="utf-8") as fh, skipped_path.open("a", encoding="utf-8") as skipped_fh:
            for chunk in tqdm(pending, desc="Julgando chunks", unit="chunk"):
                curated = curated_by_chunk[chunk.id]
                if curated.skip_reason is not None:
                    logger.warning("Chunk %s nao auditado: %s", chunk.id, curated.skip_reason)
                    skipped_fh.write(
                        SkippedChunk(answer_chunk_id=chunk.id, reason=curated.skip_reason).model_dump_json() + "\n"
                    )
                    skipped_fh.flush()
                    continue
                try:
                    result = verifier.verify(chunk, curated)
                except LLMParseError as exc:
                    logger.warning(
                        "Saida do juiz LLM nao pode ser interpretada para chunk %s — "
                        "pulando, retente com `audit resume`: %s",
                        chunk.id,
                        exc,
                    )
                    continue
                fh.write(result.model_dump_json() + "\n")
                fh.flush()


class ReportingStage:
    """Agrega `AuditResult` + `Reference` + `AnswerChunk` num `Report`
    final e renderiza `report.md`/`report.json`. Este estagio nao existia
    em codigo algum nos tres repositorios originais (Fase 4)."""

    name = "reporting"

    def run(self, ctx: RunContext) -> None:
        registry = ReferenceRegistry(ctx.run_dir)
        references = registry.load_references()
        chunks = _load_answer_chunks(ctx.run_dir)
        results = load_audit_results(ctx.run_dir / "audit_results.jsonl")
        skipped = load_skipped_chunks(ctx.run_dir / "skipped_chunks.jsonl")

        started_at = datetime.fromisoformat(ctx.started_at)
        processing_time = (datetime.now(timezone.utc) - started_at).total_seconds()

        report = aggregate_report(
            run_id=ctx.run_id,
            answer_id=ctx.run_id,
            tool_name=ctx.tool_name,
            chunks=chunks,
            references=references,
            results=results,
            skipped=skipped,
            processing_time_seconds=processing_time,
        )
        (ctx.run_dir / "report.md").write_text(
            render_markdown(report, chunks=chunks, references=references, results=results), encoding="utf-8"
        )
        (ctx.run_dir / "report.json").write_text(render_json(report), encoding="utf-8")
        logger.info("Relatorio gerado em %s", ctx.run_dir / "report.md")


class Pipeline:
    """Orquestrador fim-a-fim (Pipeline pattern), com checkpoint em disco
    (`state.json`) entre estagios — permite que `audit resume <run_id>`
    pule estagios ja concluidos sem reprocessar tudo, o que hoje nao
    existe de forma consistente entre CorpusForge (`--retry-errors`) e
    audit_with_llm (`--retry`)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._stages: list[PipelineStage] = []

    def add_stage(self, stage: PipelineStage) -> None:
        self._stages.append(stage)

    def run(self, ctx: RunContext) -> RunContext:
        ctx.stages_completed = _load_stage_state(ctx.run_dir)
        for stage in self._stages:
            if stage.name in ctx.stages_completed:
                logger.info("Pulando estagio ja concluido: %s", stage.name)
                continue
            logger.info("Executando estagio: %s", stage.name)
            stage.run(ctx)
            ctx.stages_completed.append(stage.name)
            _save_stage_state(ctx.run_dir, ctx.stages_completed)
        return ctx


def build_pipeline(settings: Settings, *, full_corpus_mode: bool = False) -> Pipeline:
    """Monta o pipeline real, resolvendo cada dependencia (Embedder,
    Reranker, LLMClient) a partir do `Settings` (Factory pattern) — os
    adapters concretos (`BGEEmbedder`, `Reranker`, `OpenAICompatibleClient`/
    `AnthropicClient`) so importam suas dependencias pesadas
    (`sentence-transformers`, SDKs de LLM) dentro do proprio construtor,
    entao montar o `Pipeline` continua barato ate aqui ser chamado."""
    embedder = BGEEmbedder(settings.embedding_model, settings.model_cache_dir)
    reranker = Reranker(settings.reranker_model, settings.model_cache_dir)
    llm_client = create_llm_client(settings)

    pipeline = Pipeline(settings)
    pipeline.add_stage(ExtractionStage())
    pipeline.add_stage(IngestionStage())
    pipeline.add_stage(IndexingStage(embedder))
    pipeline.add_stage(
        JudgingStage(
            llm_client,
            embedder,
            reranker,
            settings.retrieval_top_k,
            settings.rerank_top_k,
            full_corpus_mode=full_corpus_mode,
        )
    )
    pipeline.add_stage(ReportingStage())
    return pipeline
