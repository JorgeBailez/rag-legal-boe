"""Diff de respuestas de generación entre dos reports (por `query_id`).

Uso principal: tras re-correr generación con otro perfil (p. ej. I1 vs I2), saber qué respuestas
(`answer_text`) **cambian** — y por tanto qué anotación humana L3/L5 (fidelidad/corrección, atada a
la respuesta concreta) deja de aplicar y hay que re-mirar. Las que no cambian conservan su etiqueta.

Compara, por `query_id` presente en ambos:
  - cambio de estado answered/abstención,
  - cambio del texto de la respuesta (normalizado: espacios colapsados).

Uso:
    uv run python -m scripts.diff_generation_answers \
        --old data/processed/reports/generation/gen_20260628T122113Z_b3c260cd \
        --new data/processed/reports/generation_i1/<dev_run_id> \
        --annotations data/evaluation/corpus92_v1/judge_validation/anotacion_corpus92_dev.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_per_query(report_dir: Path) -> dict[str, dict]:
    rows = (report_dir / "per_query.jsonl").read_text(encoding="utf-8").splitlines()
    return {json.loads(x)["query_id"]: json.loads(x) for x in rows if x.strip()}


def _norm(text: str | None) -> str:
    return " ".join((text or "").split())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old", type=Path, required=True, help="report previo (I2).")
    ap.add_argument("--new", type=Path, required=True, help="report nuevo (I1).")
    ap.add_argument("--annotations", type=Path, default=None, help="anotación L3/L5 (para cruzar).")
    args = ap.parse_args()

    old = _load_per_query(args.old)
    new = _load_per_query(args.new)
    annotated = set()
    if args.annotations and args.annotations.is_file():
        annotated = {
            json.loads(x)["query_id"]
            for x in args.annotations.read_text(encoding="utf-8").splitlines()
            if x.strip()
        }

    common = sorted(set(old) & set(new))
    identical, changed_text, changed_status = [], [], []
    for qid in common:
        o, n = old[qid], new[qid]
        if bool(o.get("answered")) != bool(n.get("answered")):
            changed_status.append(qid)
        elif _norm(o.get("answer_text")) != _norm(n.get("answer_text")):
            changed_text.append(qid)
        else:
            identical.append(qid)

    print(
        f"comunes={len(common)}  idénticas={len(identical)}  "
        f"cambió_texto={len(changed_text)}  cambió_estado={len(changed_status)}"
    )
    only_old = sorted(set(old) - set(new))
    only_new = sorted(set(new) - set(old))
    if only_old:
        print(f"solo en OLD ({len(only_old)}): {only_old}")
    if only_new:
        print(f"solo en NEW ({len(only_new)}): {only_new}")

    to_recheck = sorted(set(changed_text) | set(changed_status))
    print("\n--- RESPUESTAS QUE CAMBIAN (anotación L3/L5 a re-mirar) ---")
    if not to_recheck:
        print("  ninguna — toda la anotación humana se conserva tal cual.")
    for qid in to_recheck:
        tag = " [ANOTADA]" if qid in annotated else ""
        kind = "estado" if qid in changed_status else "texto"
        print(f"  {qid} ({kind}){tag}")
    if annotated:
        recheck_annotated = [q for q in to_recheck if q in annotated]
        print(
            f"\nDe las {len(annotated)} anotadas, hay que re-mirar {len(recheck_annotated)}: "
            f"{recheck_annotated or 'ninguna'}"
        )


if __name__ == "__main__":
    main()
