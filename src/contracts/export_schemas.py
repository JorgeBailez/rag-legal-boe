"""Exporta los JSON Schema (Draft 2020-12) de los contratos Pydantic a `schemas/`.

La exportación es **determinista** (claves ordenadas, indentación fija) para que el test de
*drift* (`tests/test_contracts.py`) pueda comparar byte a byte el schema regenerado con el
versionado. Uso:

    uv run python -m src.contracts.export_schemas        # escribe schemas/*.json
    uv run python -m src.contracts.export_schemas --check # falla si hay drift (no escribe)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.contracts.embedding_models import EMBEDDING_ROOT_MODELS
from src.contracts.generation_models import GENERATION_ROOT_MODELS
from src.contracts.models import ROOT_MODELS

SCHEMAS_DIR = Path("schemas")

# Conjunto completo de contratos raíz exportables: jurídicos (Fase 1) + densos (Fase 2) +
# generación fundamentada (Fase 3).
ALL_ROOT_MODELS = {**ROOT_MODELS, **EMBEDDING_ROOT_MODELS, **GENERATION_ROOT_MODELS}


def schema_json(model) -> str:
    """JSON Schema determinista (sorted keys, indent 2, newline final) de un modelo raíz."""
    schema = model.model_json_schema(by_alias=True)
    return json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def export(out_dir: Path = SCHEMAS_DIR) -> dict[str, Path]:
    """Escribe `<name>.schema.json` por cada contrato raíz. Devuelve las rutas escritas."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, model in ALL_ROOT_MODELS.items():
        path = out_dir / f"{name}.schema.json"
        path.write_text(schema_json(model), encoding="utf-8")
        written[name] = path
    return written


def check(out_dir: Path = SCHEMAS_DIR) -> list[str]:
    """Devuelve la lista de contratos cuyo schema en disco difiere del regenerado (drift)."""
    drifted: list[str] = []
    for name, model in ALL_ROOT_MODELS.items():
        path = out_dir / f"{name}.schema.json"
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if current != schema_json(model):
            drifted.append(name)
    return drifted


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Exporta/verifica los JSON Schema de los contratos."
    )
    parser.add_argument("--check", action="store_true", help="falla si hay drift (no escribe).")
    parser.add_argument("--out", default=str(SCHEMAS_DIR))
    args = parser.parse_args()
    out_dir = Path(args.out)

    if args.check:
        drifted = check(out_dir)
        if drifted:
            print(f"[drift] schemas desincronizados: {drifted}", file=sys.stderr)
            return 1
        print("schemas sincronizados (sin drift).")
        return 0

    written = export(out_dir)
    print(f"schemas escritos: {[p.name for p in written.values()]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
