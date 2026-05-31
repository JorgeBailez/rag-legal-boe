"""Genera chunks recuperables (parent-child) de una norma BOE ya parseada.

Uso:
    uv run python scripts/chunk_boe_document.py BOE-A-2015-10565

No llama a internet: lee `data/processed/documents/<norm_id>.json` y escribe
`data/processed/chunks/<norm_id>.json`. Devuelve exit code != 0 si falla.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Permite `import src...` al ejecutar el script directamente.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.chunker import (  # noqa: E402
    create_chunks,
    load_processed_document,
    save_chunks,
)

DOCUMENTS_DIR = Path("data/processed/documents")
OUTPUT_DIR = Path("data/processed/chunks")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(
            "Uso: python scripts/chunk_boe_document.py <BOE-ID>\n"
            "Ejemplo: python scripts/chunk_boe_document.py BOE-A-2015-10565",
            file=sys.stderr,
        )
        return 2

    norm_id = argv[0]
    document_path = DOCUMENTS_DIR / f"{norm_id}.json"

    try:
        document = load_processed_document(document_path)
        chunks_document = create_chunks(document)
        out_path = save_chunks(chunks_document, OUTPUT_DIR)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    checks = chunks_document["quality_checks"]
    print(f"Chunks generados: {out_path}")
    print(
        f"  bloques fuente: {checks['source_blocks_count']} | "
        f"indexables: {checks['indexable_blocks_count']} | "
        f"chunks: {checks['chunks_count']} | "
        f"oversized: {len(checks['oversized_chunks'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
