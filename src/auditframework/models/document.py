from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class Document(BaseModel):
    """Conteudo baixado e convertido para markdown de uma Reference."""

    reference_id: str
    markdown_path: Path
    content_hash: str
    fetch_method: Literal["requests", "cloudscraper", "playwright", "local_file"]
    word_count: int
