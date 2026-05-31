"""Parsea los XML raw locales de una norma BOE al modelo documental JSON.

Uso:
    uv run python scripts/parse_boe_raw.py BOE-A-2015-10565

No llama a internet: lee el raw de `data/raw/boe/<norm_id>/` y el manifest de
`data/manifests/<norm_id>.json`, y escribe `data/processed/documents/<norm_id>.json`.
Devuelve exit code != 0 si falla.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Permite `import src...` al ejecutar el script directamente.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.boe.parser import parse_boe_document, save_processed_document  # noqa: E402
from src.core.exceptions import ParsingError  # noqa: E402

RAW_DIR = Path("data/raw/boe")
MANIFEST_DIR = Path("data/manifests")
OUTPUT_DIR = Path("data/processed/documents")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(
            "Uso: python scripts/parse_boe_raw.py <BOE-ID>\n"
            "Ejemplo: python scripts/parse_boe_raw.py BOE-A-2015-10565",
            file=sys.stderr,
        )
        return 2

    norm_id = argv[0]
    manifest_path = MANIFEST_DIR / f"{norm_id}.json"

    try:
        document = parse_boe_document(norm_id, RAW_DIR, manifest_path)
        out_path = save_processed_document(document, OUTPUT_DIR)
    except (ParsingError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    checks = document["quality_checks"]
    print(f"Documento procesado: {out_path}")
    print(
        f"  bloques: {len(document['blocks'])} | índice: {checks['index_blocks_count']} | "
        f"texto: {checks['text_blocks_count']} | warnings: {len(checks['warnings'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
