"""Valida en local (sin red) la integridad estructural del corpus MVP reprocesado.

Uso:
    uv run python scripts/validate_mvp_corpus.py

Por cada norma del catálogo canónico (`data/corpus/seed_corpus.json`) carga el documento y
los chunks y ejecuta los checks de contrato (`check_document`/`check_chunks`). Devuelve exit
code != 0 si hay algún hallazgo ERROR.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.boe.corpus import load_seed_corpus  # noqa: E402
from src.quality.corpus_audit import check_chunks, check_document  # noqa: E402

DOCUMENTS_DIR = Path("data/processed/documents")
CHUNKS_DIR = Path("data/processed/chunks")


def main(strict: bool = False) -> int:
    norms = load_seed_corpus()
    total_errors = 0
    total_warnings = 0
    for norm in norms:
        norm_id = norm["norm_id"]
        doc_path = DOCUMENTS_DIR / f"{norm_id}.json"
        chunks_path = CHUNKS_DIR / f"{norm_id}.json"
        if not doc_path.is_file() or not chunks_path.is_file():
            print(
                f"  ⚠ {norm_id}: faltan artefactos (ejecuta process_mvp_corpus.py)", file=sys.stderr
            )
            total_errors += 1
            continue
        doc = json.loads(doc_path.read_text(encoding="utf-8"))
        chunks_doc = json.loads(chunks_path.read_text(encoding="utf-8"))
        findings = check_document(doc) + check_chunks(chunks_doc, doc)
        errors = [f for f in findings if f["severity"] == "ERROR"]
        warns = [f for f in findings if f["severity"] == "WARN"]
        total_errors += len(errors)
        total_warnings += len(warns)
        status = "OK" if not errors else f"{len(errors)} ERROR"
        if warns:
            status += f" ({len(warns)} WARN)"
        print(f"  {norm_id}: {status}")
        for f in errors[:5]:
            print(f"      - {f['check']} ({f['ref']}): {f['message']}", file=sys.stderr)

    print(
        f"\nValidación: {total_errors} errores de integridad, {total_warnings} avisos en "
        f"{len(norms)} normas"
    )
    # Los ERROR siempre fallan; en modo estricto, también fallan los WARN (cierre/CI).
    if total_errors:
        return 1
    if strict and total_warnings:
        print("[--strict] hay avisos WARN → exit 1", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validación local de integridad del corpus MVP.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit != 0 también ante avisos WARN (cierre/CI local).",
    )
    args = parser.parse_args()
    raise SystemExit(main(strict=args.strict))
