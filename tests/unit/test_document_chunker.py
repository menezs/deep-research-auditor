import pytest

from auditframework.indexing.chunkers import DocumentChunker

pytest.importorskip("semantic_text_splitter")
pytest.importorskip("tiktoken")

_MARKDOWN = """# Marco Civil da Internet

O Marco Civil da Internet estabelece principios, garantias, direitos e
deveres para o uso da internet no Brasil.

## Neutralidade de rede

A neutralidade de rede impede que provedores discriminem trafego por
origem, destino, conteudo, servico ou aplicacao, salvo excecoes tecnicas
e de emergencia previstas em lei.

## Protecao de dados

Antes da LGPD, o Marco Civil ja continha disposicoes sobre a guarda e
protecao de registros de conexao e de acesso a aplicacoes de internet.
"""


def test_chunks_are_produced_with_positive_token_counts():
    chunks = DocumentChunker(max_tokens=40, overlap=0).chunk(reference_id="ref1", markdown=_MARKDOWN)

    assert len(chunks) > 1
    assert all(c.token_count > 0 for c in chunks)


def test_embedding_ids_are_sequential_starting_from_start_id():
    chunks = DocumentChunker(max_tokens=40, overlap=0).chunk(reference_id="ref1", markdown=_MARKDOWN, start_id=100)

    ids = [c.embedding_id for c in chunks]
    assert ids == list(range(100, 100 + len(chunks)))


def test_section_is_inferred_from_preceding_header():
    chunks = DocumentChunker(max_tokens=40, overlap=0).chunk(reference_id="ref1", markdown=_MARKDOWN)

    sections = {c.section for c in chunks}
    assert "Neutralidade de rede" in sections or "Marco Civil da Internet" in sections


def test_all_chunks_reference_the_same_document():
    chunks = DocumentChunker(max_tokens=40, overlap=0).chunk(reference_id="ref1", markdown=_MARKDOWN)
    assert all(c.reference_id == "ref1" for c in chunks)
