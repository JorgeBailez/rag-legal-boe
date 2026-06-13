"""Reconocimiento (read-only) del XML raw del BOE ANTES de tocar el parser.

Inspecciona los `data/raw/boe/<norm_id>/{texto,indice}.xml` descargados y reporta, sin parsear a
contratos (no falla aunque el documento sea raro): histograma de **clases CSS** de los párrafos,
**fechas** que `normalize_date` no acepta (→ riesgo de cuarentena temporal), **clases estructurales
no reconocidas** (candidatas a endurecer el parser si son recurrentes), tipos de bloque y nº de
versiones por bloque. Es la evidencia con la que se decide, por ola, qué endurecer y qué dejar en
cuarentena. Reutiliza los helpers y los conjuntos de clases del parser para un contraste fiel.

Uso:
    uv run python scripts/recon_raw_corpus.py --norm-ids BOE-A-2017-12902,BOE-A-1985-5392
    uv run python scripts/recon_raw_corpus.py   # por defecto: todas las normas descargadas
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.boe.parser import (  # noqa: E402
    NOTE_CLASSES,
    STRUCTURAL_LABEL_CLASSES,
    is_table_class,
    load_xml,
    parse_index,
    validate_response,
)
from src.core.exceptions import ParsingError  # noqa: E402
from src.evaluation.reports import new_run_id  # noqa: E402

RAW_DIR = Path("data/raw/boe")
OUTPUT_ROOT = Path("data/processed/reports/recon")
# Cuerpo normativo típico (no exhaustivo; cualquier clase no-nota se conserva como cuerpo).
BODY_CLASSES = {"articulo", "parrafo"}
# Una clase "huele" a estructural si encaja este patrón (para detectar las no reconocidas).
_STRUCTURAL_HINT = re.compile(
    r"(_num$|_tit$|disposic|transitori|adicional|derogatori|^final|secc|libro|titulo|cap[ií]tulo|anexo|parte)",
    re.IGNORECASE,
)


def _classify(css: str) -> str:
    """Clasifica una clase CSS de párrafo según cómo la trata el parser."""
    if not css:
        return "vacia"
    if css in NOTE_CLASSES:
        return "nota"
    if is_table_class(css):
        return "tabla"
    if css in STRUCTURAL_LABEL_CLASSES:
        return "estructural"
    if css in BODY_CLASSES:
        return "cuerpo"
    return "otra"  # se conserva como cuerpo recuperable; no se pierde


def recon_norm(raw_dir: Path, norm_id: str) -> dict:
    """Reconoce una norma a partir de su raw. No lanza: los errores se devuelven en el dict."""
    base = Path(raw_dir) / norm_id
    out: dict = {"norm_id": norm_id}

    try:
        texto = validate_response(load_xml(base / "texto.xml"), base / "texto.xml")
    except ParsingError as exc:
        return {"norm_id": norm_id, "error": f"texto.xml: {exc}"}

    classes = Counter((p.get("class") or "") for p in texto.iter("p"))
    block_types = Counter((b.get("tipo") or "") for b in texto.findall("bloque"))
    versions_per_block = [len(b.findall("version")) for b in texto.findall("bloque")]
    by_kind: Counter[str] = Counter()
    for css, n in classes.items():
        by_kind[_classify(css)] += n
    unknown_structural = sorted(
        css for css in classes if _classify(css) == "otra" and _STRUCTURAL_HINT.search(css)
    )

    out.update(
        {
            "n_blocks": int(sum(block_types.values())),
            "block_types": dict(block_types),
            "classes_by_kind": dict(by_kind),
            "unknown_structural_candidates": unknown_structural,
            "max_versions_per_block": max(versions_per_block, default=0),
            "multiversion_blocks": sum(1 for v in versions_per_block if v > 1),
        }
    )

    try:
        idx = validate_response(load_xml(base / "indice.xml"), base / "indice.xml")
        blocks = parse_index(idx)
        invalid = [
            b for b in blocks if b["index_last_update_date_raw"] and not b["index_last_update_date"]
        ]
        out["index_blocks"] = len(blocks)
        out["invalid_date_count"] = len(invalid)
        out["invalid_date_samples"] = sorted({b["index_last_update_date_raw"] for b in invalid})[:5]
    except ParsingError as exc:
        out["index_error"] = str(exc)

    return out


def _aggregate(per_norm: list[dict]) -> dict:
    """Agrega el recon de la ola: clases estructurales no reconocidas por nº de normas, etc."""
    unknown_by_norm: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    for r in per_norm:
        for css in r.get("unknown_structural_candidates", []):
            unknown_by_norm[css] += 1
        kinds.update(r.get("classes_by_kind", {}))
    return {
        "n_norms": len(per_norm),
        "norms_with_error": [r["norm_id"] for r in per_norm if "error" in r],
        "total_invalid_dates": sum(r.get("invalid_date_count", 0) for r in per_norm),
        "norms_with_invalid_dates": [
            r["norm_id"] for r in per_norm if r.get("invalid_date_count", 0) > 0
        ],
        # clase -> nº de normas en que aparece (≥ varias ⇒ candidata a endurecer el parser).
        "unknown_structural_classes": dict(unknown_by_norm.most_common()),
        "classes_by_kind_total": dict(kinds),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconocimiento read-only del XML raw del BOE.")
    parser.add_argument("--norm-ids", help="lista por comas; por defecto, todas las descargadas.")
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    if args.norm_ids:
        norm_ids = [n.strip() for n in args.norm_ids.split(",") if n.strip()]
    elif raw_dir.is_dir():
        norm_ids = sorted(p.name for p in raw_dir.iterdir() if p.is_dir())
    else:
        norm_ids = []
    if not norm_ids:
        print(f"No hay normas en {raw_dir} (¿descargadas?).", file=sys.stderr)
        return 1

    per_norm = [recon_norm(raw_dir, nid) for nid in norm_ids]
    aggregate = _aggregate(per_norm)

    run_id = new_run_id("recon")
    out_dir = Path(args.output_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "recon.json").write_text(
        json.dumps({"aggregate": aggregate, "per_norm": per_norm}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )

    print(f"\nRecon: {out_dir}")
    print(f"normas: {aggregate['n_norms']} · con error: {len(aggregate['norms_with_error'])}")
    print(
        f"fechas no normalizables: {aggregate['total_invalid_dates']} "
        f"(en {len(aggregate['norms_with_invalid_dates'])} normas)"
    )
    unknown = aggregate["unknown_structural_classes"]
    if unknown:
        print("clases estructurales NO reconocidas (clase → nº normas):")
        for css, n in unknown.items():
            print(f"  {css}: {n}")
    else:
        print("clases estructurales: todas reconocidas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
