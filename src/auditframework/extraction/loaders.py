from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..common.errors import ExtractionError


class AnswerLoader(Protocol):
    def load(self, path: Path) -> str: ...


class MarkdownAnswerLoader:
    def load(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")


class PdfAnswerLoader:
    def load(self, path: Path) -> str:
        import pymupdf4llm

        return pymupdf4llm.to_markdown(str(path))


class DocxAnswerLoader:
    def load(self, path: Path) -> str:
        import docx

        document = docx.Document(str(path))
        paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)


_LOADERS: dict[str, AnswerLoader] = {
    ".md": MarkdownAnswerLoader(),
    ".markdown": MarkdownAnswerLoader(),
    ".pdf": PdfAnswerLoader(),
    ".docx": DocxAnswerLoader(),
}


def load_answer(path: Path) -> str:
    """Le um arquivo de resposta de Deep Research e retorna seu texto
    (Strategy pattern por formato — md/pdf/docx). Os loaders de pdf/docx
    importam suas dependencias sob demanda para que o pacote base nao
    exija o extra `[ingestion]` apenas para lidar com respostas .md."""
    loader = _LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ExtractionError(f"Formato de resposta nao suportado: {path.suffix!r}")
    return loader.load(path)
