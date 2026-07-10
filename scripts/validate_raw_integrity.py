"""Valida la integridad del raw descargado contra los manifests (sin red).

Uso:
    uv run python scripts/validate_raw_integrity.py --seed data/corpus/seed_corpus_ampliado.json

Por cada norma del catálogo indicado recomputa el `sha256` y el `size_bytes` de cada fichero listado
en su manifest y los compara con el raw en disco.
Devuelve exit code != 0 si falta algún fichero o si algún hash/tamaño no coincide.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.boe.corpus import SEED_CORPUS_PATH, load_seed_corpus  # noqa: E402
from src.quality.corpus_audit import raw_integrity, verify_manifest  # noqa: E402

MANIFEST_DIR = Path("data/manifests")


def main(seed: Path = SEED_CORPUS_PATH) -> int:
    norms = load_seed_corpus(seed)
    norm_ids = [n["norm_id"] for n in norms]

    for norm_id in norm_ids:
        r = verify_manifest(norm_id, MANIFEST_DIR)
        problems = r["missing_files"] + r["size_mismatches"] + r["sha256_mismatches"]
        status = "OK" if not problems else f"{len(problems)} problema(s)"
        print(f"  {norm_id}: {r['files_checked']} ficheros · {status}")
        for p in problems[:5]:
            print(f"      - {p}", file=sys.stderr)

    agg = raw_integrity(norm_ids, MANIFEST_DIR)
    print(
        f"\nraw_integrity: ready={agg['ready']} | files_checked={agg['files_checked']} | "
        f"missing={len(agg['missing_files'])} size={len(agg['size_mismatches'])} "
        f"sha256={len(agg['sha256_mismatches'])}"
    )
    print(json.dumps({"raw_integrity": agg}, ensure_ascii=False))
    return 0 if agg["ready"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valida la integridad del raw vs los manifests.")
    parser.add_argument(
        "--seed",
        type=Path,
        default=SEED_CORPUS_PATH,
        help="catálogo de normas (default: seed MVP-10 data/corpus/seed_corpus.json).",
    )
    args = parser.parse_args()
    raise SystemExit(main(seed=args.seed))
