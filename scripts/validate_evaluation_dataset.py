"""Valida el dataset de evaluación de retrieval denso (contrato + reglas + Gate C).

Uso:
    uv run python scripts/validate_evaluation_dataset.py
    uv run python scripts/validate_evaluation_dataset.py --strict          # falla si hay errores
    uv run python scripts/validate_evaluation_dataset.py --require-gate-c   # falla si Gate C no ok

Gate C bloquea los benchmarks formales si el dataset no cumple el contrato requerido (sin errores
estructurales y con anotación revisada suficiente en development y test).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings.corpus_loader import load_processed_corpus  # noqa: E402
from src.evaluation.dataset import DATASET_DIR, GATE_C_LEVELS, load_and_validate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validación del dataset de evaluación denso.")
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument(
        "--gate-c-level",
        default="formal",
        choices=sorted(GATE_C_LEVELS),
        help="mínimos Gate C: checkpoint o formal (default formal).",
    )
    parser.add_argument("--strict", action="store_true", help="exit≠0 si hay errores.")
    parser.add_argument(
        "--require-gate-c", action="store_true", help="exit≠0 si Gate C no está listo."
    )
    args = parser.parse_args()

    corpus = load_processed_corpus()
    report = load_and_validate(
        Path(args.dataset_dir), corpus=corpus, gate_c_level=args.gate_c_level
    )
    print(f"dataset: {report['dataset_dir']}")
    print(f"questions: {report['n_questions']} | judgments: {report['n_judgments']}")
    print(f"por split: {report['by_split']}")
    print(f"queries revisadas con juicio relevante: {report['n_reviewed_relevant_queries']}")
    if report["errors"]:
        print(f"\nERRORES ({len(report['errors'])}):")
        for e in report["errors"]:
            print(f"  - {e}")
    if report["warnings"]:
        print(f"\nAVISOS ({len(report['warnings'])}):")
        for w in report["warnings"]:
            print(f"  - {w}")
    gate = report["gate_c"]
    print(
        f"\nGate C ({gate['level']}): {'LISTO' if gate['ready'] else 'NO LISTO'} "
        f"| mínimos: {gate['minimums']}"
    )
    for r in gate["reasons"]:
        print(f"  - {r}")

    if args.strict and report["errors"]:
        print("\n[--strict] hay errores estructurales → exit 1", file=sys.stderr)
        return 1
    if args.require_gate_c and not gate["ready"]:
        print("\n[--require-gate-c] Gate C no listo → exit 1", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
