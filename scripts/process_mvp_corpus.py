"""Reprocesa localmente un corpus (parser + chunker) desde el raw ya descargado.

Uso:
    uv run python scripts/process_mvp_corpus.py --seed data/corpus/seed_corpus_ampliado.json

No llama a internet ni re-descarga: por cada norma del catálogo indicado regenera
`data/processed/documents/<id>.json`, `histories`, `parents` y `chunks`. Devuelve exit code != 0 si
falla alguna.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.boe.corpus import SEED_CORPUS_PATH, load_seed_corpus  # noqa: E402
from src.boe.parser import build_processed_bundle, save_processed_bundle  # noqa: E402
from src.core.exceptions import ParsingError  # noqa: E402
from src.preprocessing.chunker import create_chunks, save_chunks  # noqa: E402

RAW_DIR = Path("data/raw/boe")
MANIFEST_DIR = Path("data/manifests")
DOCUMENTS_DIR = Path("data/processed/documents")
HISTORIES_DIR = Path("data/processed/histories")
PARENTS_DIR = Path("data/processed/parents")
CHUNKS_DIR = Path("data/processed/chunks")


def main(seed: Path = SEED_CORPUS_PATH) -> int:
    """Regenera offline el corpus a los contratos v2 (document + history + parents + chunks)."""
    norms = load_seed_corpus(seed)
    failures = 0
    for norm in norms:
        norm_id = norm["norm_id"]
        try:
            bundle = build_processed_bundle(norm_id, RAW_DIR, MANIFEST_DIR / f"{norm_id}.json")
            save_processed_bundle(bundle, DOCUMENTS_DIR, HISTORIES_DIR, PARENTS_DIR)
            chunks_document = create_chunks(bundle.document, bundle.parents)
            save_chunks(chunks_document, CHUNKS_DIR)
        except (ParsingError, FileNotFoundError) as exc:
            failures += 1
            print(f"  [WARN] {norm_id}: {exc}", file=sys.stderr)
            continue
        print(
            f"  {norm_id}: bloques={len(bundle.document['blocks'])} "
            f"parents={len(bundle.parents['parents'])} chunks={len(chunks_document['chunks'])}"
        )

    print(f"\nNormas reprocesadas: {len(norms) - failures}/{len(norms)} (sin red)")
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reprocesa offline el corpus (parser + chunker).")
    parser.add_argument(
        "--seed",
        type=Path,
        default=SEED_CORPUS_PATH,
        help="catálogo de normas (default: seed MVP-10 data/corpus/seed_corpus.json).",
    )
    args = parser.parse_args()
    raise SystemExit(main(seed=args.seed))
