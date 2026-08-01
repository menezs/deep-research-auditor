from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path


def make_run_id(input_path: Path) -> str:
    """`run_id = hash(arquivo_de_entrada) + timestamp`.

    Substitui a convencao manual de nomes de pasta usada hoje
    (`gemini-direito`, `gpt-direito`, ...) por um identificador
    deterministico quanto ao conteudo de entrada, permitindo cache e
    idempotencia entre execucoes do mesmo arquivo."""
    content_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()[:12]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{input_path.stem}_{content_hash}_{timestamp}"
