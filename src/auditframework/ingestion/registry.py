from __future__ import annotations

import json
from pathlib import Path

from ..models import Document, Reference


class ReferenceRegistry:
    """Persistencia idempotente de Reference/Document para uma execucao
    (Repository pattern).

    Substitui a leitura/escrita de JSON solta espalhada pelo `main.py` do
    CorpusForge por uma unica interface, hoje apoiada em arquivos, mas
    trocavel por um backend em banco de dados sem alterar quem a chama."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.documents_dir = run_dir / "documents"
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.references_path = run_dir / "references.json"

    def load_references(self) -> list[Reference]:
        if not self.references_path.exists():
            return []
        raw = json.loads(self.references_path.read_text(encoding="utf-8"))
        return [Reference.model_validate(item) for item in raw]

    def save_references(self, references: list[Reference]) -> None:
        payload = [json.loads(ref.model_dump_json()) for ref in references]
        self._atomic_write(self.references_path, json.dumps(payload, ensure_ascii=False, indent=2))

    def document_path(self, reference_id: str) -> Path:
        return self.documents_dir / f"{reference_id}.md"

    def metadata_path(self, reference_id: str) -> Path:
        return self.documents_dir / f"{reference_id}.meta.json"

    def has_document(self, reference_id: str) -> bool:
        return self.document_path(reference_id).exists()

    def load_document(self, reference_id: str) -> Document:
        raw = json.loads(self.metadata_path(reference_id).read_text(encoding="utf-8"))
        return Document.model_validate(raw)

    def save_document(self, document: Document, markdown_content: str) -> None:
        self._atomic_write(self.document_path(document.reference_id), markdown_content)
        self._atomic_write(self.metadata_path(document.reference_id), document.model_dump_json(indent=2))

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(path)
