"""Reprocesa localmente el corpus MVP (parser + chunker) desde el raw ya descargado.

Uso:
    uv run python scripts/process_mvp_corpus.py

No llama a internet ni re-descarga: por cada norma del catálogo canónico
(`data/corpus/seed_corpus.json`) regenera `data/processed/documents/<id>.json` y
`data/processed/chunks/<id>.json`. Devuelve exit code != 0 si falla alguna.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.boe.corpus import load_seed_corpus  # noqa: E402
from src.boe.parser import parse_boe_document, save_processed_document  # noqa: E402
from src.core.exceptions import ParsingError  # noqa: E402
from src.preprocessing.chunker import create_chunks, save_chunks  # noqa: E402

RAW_DIR = Path("data/raw/boe")
MANIFEST_DIR = Path("data/manifests")
DOCUMENTS_DIR = Path("data/processed/documents")
CHUNKS_DIR = Path("data/processed/chunks")


def main() -> int:
    norms = load_seed_corpus()
    failures = 0
    for norm in norms:
        norm_id = norm["norm_id"]
        try:
            document = parse_boe_document(norm_id, RAW_DIR, MANIFEST_DIR / f"{norm_id}.json")
            save_processed_document(document, DOCUMENTS_DIR)
            chunks_document = create_chunks(document)
            save_chunks(chunks_document, CHUNKS_DIR)
        except (ParsingError, FileNotFoundError) as exc:
            failures += 1
            print(f"  ⚠ {norm_id}: {exc}", file=sys.stderr)
            continue
        print(
            f"  {norm_id}: bloques={len(document['blocks'])} "
            f"chunks={chunks_document['quality_checks']['chunks_count']}"
        )

    print(f"\nNormas reprocesadas: {len(norms) - failures}/{len(norms)} (sin red)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
