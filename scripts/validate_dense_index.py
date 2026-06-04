"""Revalida un bundle de índice denso ya publicado (checksums + Gate B).

Uso:
    uv run python scripts/validate_dense_index.py --bundle data/indexes/dense/<bundle_id>

Recalcula los checksums de los artefactos contra el manifest y vuelve a pasar Gate B sobre la
matriz (dtype, dimensión, NaN/Inf, vectores nulos, norma L2, ids/row_index, etc.). Exit≠0 si hay
errores.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings.bundle import revalidate_bundle  # noqa: E402
from src.embeddings.corpus_loader import load_processed_corpus  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Revalida un bundle de índice denso.")
    parser.add_argument("--bundle", required=True, help="ruta al directorio del bundle.")
    args = parser.parse_args()

    bundle_dir = Path(args.bundle)
    if not (bundle_dir / "manifest.json").is_file():
        print(f"No es un bundle válido (falta manifest.json): {bundle_dir}", file=sys.stderr)
        return 1

    corpus = load_processed_corpus()
    report = revalidate_bundle(bundle_dir, corpus=corpus)
    print(f"bundle: {report['bundle_id']}")
    print(f"vectores: {report['n_rows']} | dimensión: {report['embedding_dimension']}")
    print(f"resumen: {report['summary']}")
    for f in report["findings"]:
        if f["severity"] != "INFO":
            ev = f" [{f['evidence']}]" if f.get("evidence") else ""
            print(f"  {f['severity']:7} {f['gate']}.{f['check']}: {f['message']}{ev}")
    ok = report["gate_b_passed"]
    print(f"\nGate B: {'OK' if ok else 'FALLO'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
