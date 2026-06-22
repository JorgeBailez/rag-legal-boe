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

from pydantic import ValidationError  # noqa: E402

from src.boe.corpus import SEED_CORPUS_PATH, load_seed_corpus  # noqa: E402
from src.contracts.models import ChunksV2, DocumentV2, HistoryV2, ParentsV2  # noqa: E402
from src.quality.corpus_audit import (  # noqa: E402
    check_chunks,
    check_document,
    check_history,
    check_parents,
    check_relational,
    join_norm,
)

DOCUMENTS_DIR = Path("data/processed/documents")
HISTORIES_DIR = Path("data/processed/histories")
PARENTS_DIR = Path("data/processed/parents")
CHUNKS_DIR = Path("data/processed/chunks")

# Validación local por artefacto (contratos Pydantic = fuente única).
_CONTRACTS = {
    "document": DocumentV2,
    "history": HistoryV2,
    "parents": ParentsV2,
    "chunks": ChunksV2,
}


def _validate_contracts(norm_id: str, artifacts: dict) -> list[str]:
    """Valida cada artefacto contra su modelo Pydantic. Devuelve mensajes de error legibles."""
    errs: list[str] = []
    for name, model in _CONTRACTS.items():
        try:
            model.model_validate(artifacts[name])
        except ValidationError as exc:
            errs.append(f"{norm_id}/{name}: contrato inválido — {exc.error_count()} error(es)")
    return errs


def main(strict: bool = False, seed: Path = SEED_CORPUS_PATH) -> int:
    norms = load_seed_corpus(seed)
    total_errors = 0
    total_warnings = 0
    for norm in norms:
        norm_id = norm["norm_id"]
        paths = {
            "document": DOCUMENTS_DIR / f"{norm_id}.json",
            "history": HISTORIES_DIR / f"{norm_id}.json",
            "parents": PARENTS_DIR / f"{norm_id}.json",
            "chunks": CHUNKS_DIR / f"{norm_id}.json",
        }
        if not all(p.is_file() for p in paths.values()):
            print(
                f"  ⚠ {norm_id}: faltan artefactos (ejecuta process_mvp_corpus.py)", file=sys.stderr
            )
            total_errors += 1
            continue
        document = json.loads(paths["document"].read_text(encoding="utf-8"))
        history = json.loads(paths["history"].read_text(encoding="utf-8"))
        parents = json.loads(paths["parents"].read_text(encoding="utf-8"))
        chunks_doc = json.loads(paths["chunks"].read_text(encoding="utf-8"))

        # 1) Validación local de contratos (Pydantic); 2) auditoría relacional.
        contract_errs = _validate_contracts(
            norm_id,
            {"document": document, "history": history, "parents": parents, "chunks": chunks_doc},
        )
        joined = join_norm(document, history, parents)
        findings = (
            check_document(document, history, parents)
            + check_history(document, history)
            + check_parents(document, parents)
            + check_chunks(chunks_doc, joined)
            + check_relational(document, history, parents, chunks_doc)
        )
        errors = [f for f in findings if f["severity"] == "ERROR"]
        for ce in contract_errs:
            print(f"      - contract: {ce}", file=sys.stderr)
        total_errors += len(contract_errs)
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
    parser = argparse.ArgumentParser(description="Validación local de integridad del corpus.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit != 0 también ante avisos WARN (cierre/CI local).",
    )
    parser.add_argument(
        "--seed",
        type=Path,
        default=SEED_CORPUS_PATH,
        help="catálogo de normas (default: seed MVP-10 data/corpus/seed_corpus.json).",
    )
    args = parser.parse_args()
    raise SystemExit(main(strict=args.strict, seed=args.seed))
