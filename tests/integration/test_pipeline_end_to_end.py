"""Teste de integracao fim-a-fim do pipeline (extraction -> ingestion ->
indexing -> judging -> reporting), inteiramente offline: substitui os
adapters que tocam rede/modelos reais (Fetcher, Embedder, LLMClient) por
fakes deterministicos, mas usa a implementacao REAL de todo o resto
(extracao por regex, conversao HTML->Markdown via trafilatura, FAISS,
Retriever, Verifier, aggregator/render). O objetivo e provar que os
contratos entre estagios (Fase 0-4) realmente se encaixam quando
orquestrados pelo `Pipeline` (Fase 5), algo que nenhum teste unitario
isolado consegue garantir."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("faiss")
pytest.importorskip("trafilatura")

from auditframework.common.errors import DeadReferenceError, LLMParseError
from auditframework.common.llm_client import LLMUsage
from auditframework.config import Settings
from auditframework.ingestion.fetcher import FetchResult
from auditframework.judging.prompts import JudgeOutput
from auditframework.pipeline import (
    ExtractionStage,
    IndexingStage,
    IngestionStage,
    JudgingStage,
    Pipeline,
    ReportingStage,
    RunContext,
    save_run_meta,
)

_LGPD_URL = "https://example.com/lgpd"
_MARCO_CIVIL_URL = "https://example.com/marco-civil"

_ANSWER_MD = f"""# Resposta

A LGPD estabelece bases legais para o tratamento de dados pessoais. [1]

O Marco Civil da Internet garante neutralidade de rede. [2]

## Referências

[1] Lei Geral de Protecao de Dados
{_LGPD_URL}

[2] Marco Civil da Internet
{_MARCO_CIVIL_URL}
"""

_LGPD_HTML = """
<html><head><title>LGPD</title></head><body><article>
<h1>Lei Geral de Protecao de Dados</h1>
<p>A LGPD estabelece bases legais para o tratamento de dados pessoais no
Brasil, incluindo consentimento e legitimo interesse, e cria a ANPD como
autoridade fiscalizadora.</p>
</article></body></html>
"""

_MARCO_CIVIL_HTML = """
<html><head><title>Marco Civil</title></head><body><article>
<h1>Marco Civil da Internet</h1>
<p>O Marco Civil da Internet garante neutralidade de rede, assegurando
tratamento isonomico a qualquer pacote de dados, sem distincao por
conteudo, origem ou destino.</p>
</article></body></html>
"""


class FakeFetcher:
    def __init__(self, behavior: dict[str, FetchResult]):
        self.behavior = behavior
        self.calls: dict[str, int] = {}

    def fetch(self, url: str) -> FetchResult:
        self.calls[url] = self.calls.get(url, 0) + 1
        result = self.behavior[url]
        if isinstance(result, Exception):
            raise result
        return result

    def fetch_via_playwright(self, url: str) -> FetchResult:
        return self.fetch(url)


class KeywordEmbedder:
    """Embedder fake: mapeia qualquer texto (chunk de resposta ou de
    documento) para um vetor one-hot de acordo com a primeira
    palavra-chave que aparecer nele. Evita depender do texto exato
    produzido pelo `MarkdownSplitter` real (que pode variar espacos/
    quebras de linha) — só importa que a palavra-chave sobreviva."""

    dimension = 2

    def __init__(self, keyword_vectors: dict[str, list[float]]):
        self._keyword_vectors = keyword_vectors

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._vector_for(text) for text in texts]

    def _vector_for(self, text: str) -> list[float]:
        for keyword, vector in self._keyword_vectors.items():
            if keyword in text:
                return vector
        raise AssertionError(f"nenhuma palavra-chave reconhecida no texto: {text!r}")


class FakeLLMClient:
    """Escolhe a resposta a devolver com base numa palavra-chave presente
    no prompt (que embute o `chunk.text` verbatim) — permite simular
    juizes com veredito diferente por chunk sem acoplar ao mock a um
    numero fixo de chamadas."""

    model = "fake-judge"

    def __init__(self, responses_by_keyword: dict[str, JudgeOutput]):
        self._responses = responses_by_keyword
        self.calls: list[str] = []

    def complete_json(self, *, system_message: str, user_prompt: str, schema):
        self.calls.append(user_prompt)
        for keyword, output in self._responses.items():
            if keyword in user_prompt:
                return output, LLMUsage(prompt_tokens=42, completion_tokens=8, latency_ms=5, cost_usd=0.0001)
        raise AssertionError(f"nenhum fake configurado para o prompt: {user_prompt!r}")


@pytest.fixture
def embedder() -> KeywordEmbedder:
    return KeywordEmbedder({"LGPD": [1.0, 0.0], "neutralidade": [0.0, 1.0]})


@pytest.fixture
def fetcher() -> FakeFetcher:
    return FakeFetcher(
        {
            _LGPD_URL: FetchResult(content=_LGPD_HTML.encode("utf-8"), content_type="text/html", fetch_method="requests", http_status=200),
            _MARCO_CIVIL_URL: FetchResult(
                content=_MARCO_CIVIL_HTML.encode("utf-8"), content_type="text/html", fetch_method="requests", http_status=200
            ),
        }
    )


@pytest.fixture
def llm_client() -> FakeLLMClient:
    return FakeLLMClient(
        {
            "LGPD estabelece bases legais": JudgeOutput(
                verdict="supported", justification="O contexto confirma a alegacao.", cited_excerpts=["A LGPD estabelece bases legais"]
            ),
            "Marco Civil da Internet garante neutralidade": JudgeOutput(
                verdict="contradicted", justification="O contexto contradiz a alegacao.", cited_excerpts=[]
            ),
        }
    )


def _build_pipeline(settings: Settings, fetcher: FakeFetcher, embedder: KeywordEmbedder, llm_client: FakeLLMClient) -> Pipeline:
    pipeline = Pipeline(settings)
    pipeline.add_stage(ExtractionStage())
    pipeline.add_stage(IngestionStage(fetcher=fetcher))
    pipeline.add_stage(IndexingStage(embedder))
    pipeline.add_stage(JudgingStage(llm_client, embedder, reranker=None, top_k=5, rerank_top_k=5))
    pipeline.add_stage(ReportingStage())
    return pipeline


def _make_ctx(tmp_path, answer_path) -> tuple[Settings, RunContext]:
    settings = Settings(data_dir=tmp_path / "data", model_cache_dir=tmp_path / "model_cache")
    ctx = RunContext(run_id="run-test", settings=settings, answer_path=answer_path, tool_name="ChatGPT")
    return settings, ctx


def test_pipeline_runs_all_stages_and_produces_a_coherent_report(tmp_path, fetcher, embedder, llm_client):
    answer_path = tmp_path / "answer.md"
    answer_path.write_text(_ANSWER_MD, encoding="utf-8")
    settings, ctx = _make_ctx(tmp_path, answer_path)
    save_run_meta(ctx)

    pipeline = _build_pipeline(settings, fetcher, embedder, llm_client)
    pipeline.run(ctx)

    run_dir = ctx.run_dir
    assert ctx.stages_completed == ["extraction", "ingestion", "indexing", "judging", "reporting"]

    references = json.loads((run_dir / "references.json").read_text(encoding="utf-8"))
    assert len(references) == 2
    assert all(r["status"] == "downloaded" for r in references)

    chunks = json.loads((run_dir / "answer_chunks.json").read_text(encoding="utf-8"))
    assert len(chunks) == 2
    # a secao "## Referencias" nao deve virar chunks (ver _strip_reference_section)
    assert all("Lei Geral de Protecao de Dados" not in c["text"] for c in chunks)

    results = (run_dir / "audit_results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(results) == 2
    verdicts = {json.loads(line)["verdict"] for line in results}
    assert verdicts == {"supported", "contradicted"}

    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["pct_supported"] == 50.0
    assert report["pct_contradicted"] == 50.0
    assert len(report["reference_stats"]) == 2

    report_md = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "[[1]]" in report_md or "[1]" in report_md
    assert "SUPPORTED" in report_md and "CONTRADICTED" in report_md


def test_resume_only_judges_chunks_without_a_persisted_result(tmp_path, fetcher, embedder, llm_client):
    answer_path = tmp_path / "answer.md"
    answer_path.write_text(_ANSWER_MD, encoding="utf-8")
    settings, ctx = _make_ctx(tmp_path, answer_path)
    save_run_meta(ctx)

    # roda so ate o fim da indexacao "manualmente" (sem o estagio de judging)
    partial_pipeline = Pipeline(settings)
    partial_pipeline.add_stage(ExtractionStage())
    partial_pipeline.add_stage(IngestionStage(fetcher=fetcher))
    partial_pipeline.add_stage(IndexingStage(embedder))
    partial_pipeline.run(ctx)

    # simula que uma execucao anterior ja julgou o chunk 0 antes de cair
    chunks = json.loads((ctx.run_dir / "answer_chunks.json").read_text(encoding="utf-8"))
    already_judged_id = chunks[0]["id"]
    from auditframework.models import AuditResult, AuditVerdict

    preexisting = AuditResult(
        answer_chunk_id=already_judged_id, verdict=AuditVerdict.SUPPORTED, justification="ja julgado antes", judge_model="fake-judge"
    )
    (ctx.run_dir / "audit_results.jsonl").write_text(preexisting.model_dump_json() + "\n", encoding="utf-8")

    # forca o estado a "esquecer" apenas o estagio de judging/reporting, como um resume real faria
    ctx.stages_completed = ["extraction", "ingestion", "indexing"]
    from auditframework.pipeline import _save_stage_state

    _save_stage_state(ctx.run_dir, ctx.stages_completed)

    full_pipeline = _build_pipeline(settings, fetcher, embedder, llm_client)
    full_pipeline.run(ctx)

    results = (ctx.run_dir / "audit_results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(results) == 2
    # o juiz fake so deve ter sido chamado para o chunk pendente, nao para o ja persistido
    assert len(llm_client.calls) == 1


class OrderCheckingLLMClient(FakeLLMClient):
    """Registra, na primeira chamada ao juiz, quantos `curated_*.json` ja
    existem em disco — usado para provar que a fase de retrieval (todos
    os chunks) roda por inteiro antes da primeira chamada de julgamento."""

    def __init__(self, responses_by_keyword: dict[str, JudgeOutput], curated_dir):
        super().__init__(responses_by_keyword)
        self._curated_dir = curated_dir
        self.curated_count_at_first_call: int | None = None

    def complete_json(self, *, system_message: str, user_prompt: str, schema):
        if self.curated_count_at_first_call is None:
            self.curated_count_at_first_call = len(list(self._curated_dir.glob("curated_*.json")))
        return super().complete_json(system_message=system_message, user_prompt=user_prompt, schema=schema)


def test_all_curated_documents_are_built_before_the_first_judge_call(tmp_path, fetcher, embedder):
    """Regressao: os `curated_*.json` de TODOS os chunks pendentes devem
    existir em disco antes de qualquer chamada ao LLM juiz — retrieval
    (fase 1, local/rapido) e julgamento (fase 2, uma chamada de rede por
    chunk) sao fases sequenciais e distintas de `JudgingStage`, nao
    interpoladas chunk a chunk."""
    answer_path = tmp_path / "answer.md"
    answer_path.write_text(_ANSWER_MD, encoding="utf-8")
    settings, ctx = _make_ctx(tmp_path, answer_path)
    save_run_meta(ctx)

    llm_client = OrderCheckingLLMClient(
        {
            "LGPD estabelece bases legais": JudgeOutput(
                verdict="supported", justification="ok", cited_excerpts=[]
            ),
            "Marco Civil da Internet garante neutralidade": JudgeOutput(
                verdict="contradicted", justification="ok", cited_excerpts=[]
            ),
        },
        curated_dir=ctx.run_dir / "curated",
    )

    pipeline = _build_pipeline(settings, fetcher, embedder, llm_client)
    pipeline.run(ctx)

    assert llm_client.curated_count_at_first_call == 2


class RaisingOnceLLMClient(FakeLLMClient):
    """Levanta `LLMParseError` na primeira vez que um prompt contendo
    `fail_keyword` e recebido — simula a saida do LLM juiz vindo
    truncada/nao-JSON, sem depender de um provider real."""

    def __init__(self, responses_by_keyword: dict[str, JudgeOutput], fail_keyword: str):
        super().__init__(responses_by_keyword)
        self._fail_keyword = fail_keyword
        self._already_failed = False

    def complete_json(self, *, system_message: str, user_prompt: str, schema):
        if not self._already_failed and self._fail_keyword in user_prompt:
            self._already_failed = True
            raise LLMParseError("resposta do LLM nao e JSON valido: '...trecho truncado'")
        return super().complete_json(system_message=system_message, user_prompt=user_prompt, schema=schema)


def test_chunk_with_unparseable_judge_output_is_skipped_without_crashing_the_run(tmp_path, fetcher, embedder):
    """Regressao: uma saida do juiz LLM que nao pode ser interpretada (ex:
    JSON truncado por limite de tokens) nao deve derrubar o run inteiro —
    o chunk fica pendente (sem resultado persistido) e o restante do
    julgamento continua normalmente."""
    answer_path = tmp_path / "answer.md"
    answer_path.write_text(_ANSWER_MD, encoding="utf-8")
    settings, ctx = _make_ctx(tmp_path, answer_path)
    save_run_meta(ctx)

    llm_client = RaisingOnceLLMClient(
        {
            "LGPD estabelece bases legais": JudgeOutput(
                verdict="supported", justification="ok", cited_excerpts=[]
            ),
            "Marco Civil da Internet garante neutralidade": JudgeOutput(
                verdict="contradicted", justification="ok", cited_excerpts=[]
            ),
        },
        fail_keyword="Marco Civil da Internet garante neutralidade",
    )

    pipeline = _build_pipeline(settings, fetcher, embedder, llm_client)
    pipeline.run(ctx)  # nao deve levantar excecao

    assert ctx.stages_completed == ["extraction", "ingestion", "indexing", "judging", "reporting"]
    results = (ctx.run_dir / "audit_results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(results) == 1
    verdicts = {json.loads(line)["verdict"] for line in results}
    assert verdicts == {"supported"}


def test_chunk_citing_a_dead_reference_is_skipped_not_judged(tmp_path, embedder, llm_client):
    """Regressao: no modo padrao (escopado por citacao), um chunk cuja
    unica referencia citada nao pode ser baixada nao deve ser julgado
    contra evidencia de outra fonte que ele nunca citou — deve ser
    registrado em `skipped_chunks.jsonl` com justificativa, sem chamar o
    LLM, e contabilizado no report."""
    answer_path = tmp_path / "answer.md"
    answer_path.write_text(_ANSWER_MD, encoding="utf-8")
    settings, ctx = _make_ctx(tmp_path, answer_path)
    save_run_meta(ctx)

    fetcher = FakeFetcher(
        {
            _LGPD_URL: FetchResult(content=_LGPD_HTML.encode("utf-8"), content_type="text/html", fetch_method="requests", http_status=200),
            _MARCO_CIVIL_URL: DeadReferenceError("404"),
        }
    )

    pipeline = _build_pipeline(settings, fetcher, embedder, llm_client)
    pipeline.run(ctx)

    assert ctx.stages_completed == ["extraction", "ingestion", "indexing", "judging", "reporting"]

    results = (ctx.run_dir / "audit_results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(results) == 1
    assert json.loads(results[0])["verdict"] == "supported"
    # o juiz fake nao deve ter sido chamado para o chunk cuja referencia morreu
    assert len(llm_client.calls) == 1

    skipped = (ctx.run_dir / "skipped_chunks.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(skipped) == 1
    skipped_record = json.loads(skipped[0])
    assert skipped_record["reason"]  # justificativa nao-vazia, persistida para o report

    report = json.loads((ctx.run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["count_skipped"] == 1
    assert report["count_supported"] == 1

    report_md = (ctx.run_dir / "report.md").read_text(encoding="utf-8")
    assert "SKIPPED" in report_md
    assert "Chunks Não Auditados" in report_md


def test_resume_does_not_reprocess_an_already_skipped_chunk(tmp_path, embedder, llm_client):
    """Regressao: um chunk ja registrado em `skipped_chunks.jsonl` numa
    execucao anterior nao deve ser reavaliado (nem gerar um novo registro
    duplicado, nem tentar chamar o LLM) num `audit resume`."""
    answer_path = tmp_path / "answer.md"
    answer_path.write_text(_ANSWER_MD, encoding="utf-8")
    settings, ctx = _make_ctx(tmp_path, answer_path)
    save_run_meta(ctx)

    fetcher = FakeFetcher(
        {
            _LGPD_URL: FetchResult(content=_LGPD_HTML.encode("utf-8"), content_type="text/html", fetch_method="requests", http_status=200),
            _MARCO_CIVIL_URL: DeadReferenceError("404"),
        }
    )

    pipeline = _build_pipeline(settings, fetcher, embedder, llm_client)
    pipeline.run(ctx)
    assert len(llm_client.calls) == 1
    skipped_before = (ctx.run_dir / "skipped_chunks.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(skipped_before) == 1

    # forca um resume: "esquece" apenas judging/reporting, como um resume real faria
    ctx.stages_completed = ["extraction", "ingestion", "indexing"]
    from auditframework.pipeline import _save_stage_state

    _save_stage_state(ctx.run_dir, ctx.stages_completed)

    pipeline.run(ctx)

    skipped_after = (ctx.run_dir / "skipped_chunks.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(skipped_after) == 1  # nao duplicou o registro
    assert len(llm_client.calls) == 1  # o juiz nao foi chamado de novo


def test_full_corpus_mode_never_skips_even_with_a_dead_cited_reference(tmp_path, embedder, llm_client):
    """No modo `full_corpus_mode`, a citacao e ignorada — mesmo um chunk
    cuja referencia citada esta morta deve ser julgado, usando o corpus
    inteiro (que ainda tem a outra referencia baixada) como evidencia."""
    answer_path = tmp_path / "answer.md"
    answer_path.write_text(_ANSWER_MD, encoding="utf-8")
    settings, ctx = _make_ctx(tmp_path, answer_path)
    ctx.full_corpus_mode = True
    save_run_meta(ctx)

    fetcher = FakeFetcher(
        {
            _LGPD_URL: FetchResult(content=_LGPD_HTML.encode("utf-8"), content_type="text/html", fetch_method="requests", http_status=200),
            _MARCO_CIVIL_URL: DeadReferenceError("404"),
        }
    )

    pipeline = Pipeline(settings)
    pipeline.add_stage(ExtractionStage())
    pipeline.add_stage(IngestionStage(fetcher=fetcher))
    pipeline.add_stage(IndexingStage(embedder))
    pipeline.add_stage(JudgingStage(llm_client, embedder, reranker=None, top_k=5, rerank_top_k=5, full_corpus_mode=True))
    pipeline.add_stage(ReportingStage())
    pipeline.run(ctx)

    skipped_path = ctx.run_dir / "skipped_chunks.jsonl"
    assert not skipped_path.exists() or skipped_path.read_text(encoding="utf-8").strip() == ""
    results = (ctx.run_dir / "audit_results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(results) == 2

    report = json.loads((ctx.run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["count_skipped"] == 0
