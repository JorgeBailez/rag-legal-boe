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

from src.boe.parser import build_processed_bundle, save_processed_bundle  # noqa: E402
from src.core.exceptions import ParsingError  # noqa: E402

RAW_DIR = Path("data/raw/boe")
MANIFEST_DIR = Path("data/manifests")
DOCUMENTS_DIR = Path("data/processed/documents")
HISTORIES_DIR = Path("data/processed/histories")
PARENTS_DIR = Path("data/processed/parents")


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
        bundle = build_processed_bundle(norm_id, RAW_DIR, manifest_path)
        paths = save_processed_bundle(bundle, DOCUMENTS_DIR, HISTORIES_DIR, PARENTS_DIR)
    except (ParsingError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Documento procesado: {paths['document']}")
    print(
        f"  bloques: {len(bundle.document['blocks'])} | "
        f"history: {len(bundle.history['blocks'])} | parents: {len(bundle.parents['parents'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
