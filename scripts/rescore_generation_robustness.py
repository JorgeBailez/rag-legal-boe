"""Robustez de las métricas deterministas de generación combinando *development* + *test*.

Motivación (cap. de experimentos): las métricas de generación se reportan sobre `test` (n pequeño:
18 respondidas). Este script NO regenera nada: reutiliza las respuestas ya guardadas en el
`per_query.jsonl` de reports existentes y **re-puntúa** (funciones puras, sin LLM) contra el
`answer_keys.jsonl` actual, para reportar dev+test como **robustez** (más n, IC más estrechos).

Requisito de validez: los reports combinados deben compartir configuración de generación (mismo
`prompt_fingerprint`, bundle, k, contexto, temperatura). Con temperatura 0 las respuestas son
deterministas, así que re-puntuar equivale a una re-corrida. Es un análisis de robustez, NO held-out
(dev es el split de ajuste): sesgo optimista leve, se declara como tal.

Métricas deterministas (sin juez): key-fact recall, citas P/R/F1 (endurecidas, frontera de palabra),
y sobre-abstención. Cada una con IC 95 % por bootstrap.

Uso:
    uv run python -m scripts.rescore_generation_robustness \
        --report test=data/processed/reports/generation/gen_20260630T091038Z_6fe3b021 \
        --report dev=data/processed/reports/generation/gen_20260628T122113Z_b3c260cd \
        --dataset-dir data/evaluation/corpus92_v1 \
        --out data/processed/reports/generation/robustez_gen_devtest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.generation_metrics import citation_attribution, key_fact_recall
from src.evaluation.metrics import bootstrap_ci


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _rescore(rows: list[dict], ak: dict[str, dict]) -> dict:
    """Re-puntúa una lista de per_query contra el gold actual (funciones puras)."""
    kf: list[float] = []
    cp: list[float] = []
    cr: list[float] = []
    cf: list[float] = []
    answerable = answered = 0
    for r in rows:
        if r.get("answerable"):
            answerable += 1
            if r.get("answered"):
                answered += 1
        if not (r.get("answered") and r.get("answerable")):
            continue
        a = ak.get(r["query_id"])
        if not a:
            continue
        if a.get("key_facts"):
            v = key_fact_recall(r.get("answer_text", ""), a["key_facts"])["key_fact_recall"]
            if v is not None:
                kf.append(v)
        if a.get("expected_citation_parents"):
            c = citation_attribution(r.get("cited_parents", []), a["expected_citation_parents"])
            if c["citation_f1"] is not None:
                cp.append(c["citation_precision"])
                cr.append(c["citation_recall"])
                cf.append(c["citation_f1"])
    return {
        "answerable": answerable,
        "answered": answered,
        "over_abstention_rate": (answerable - answered) / answerable if answerable else None,
        "key_fact": {"n": len(kf), "mean": _mean(kf), "ci": bootstrap_ci(kf) if kf else None},
        "citation_precision": {"n": len(cp), "mean": _mean(cp)},
        "citation_recall": {"n": len(cr), "mean": _mean(cr)},
        "citation_f1": {"n": len(cf), "mean": _mean(cf), "ci": bootstrap_ci(cf) if cf else None},
    }


def _mean(xs: list[float]) -> float | None:
    return (sum(xs) / len(xs)) if xs else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--report", action="append", required=True, help="etiqueta=ruta_report (repetible)."
    )
    ap.add_argument("--dataset-dir", type=Path, default=Path("data/evaluation/corpus92_v1"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ak = {a["query_id"]: a for a in _load_jsonl(args.dataset_dir / "answer_keys.jsonl")}
    sources: dict[str, list[dict]] = {}
    fingerprints: dict[str, str] = {}
    for spec in args.report:
        label, path = spec.split("=", 1)
        d = Path(path)
        sources[label] = _load_jsonl(d / "per_query.jsonl")
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        fingerprints[label] = cfg.get("prompt_fingerprint", "?")

    # comprobación de validez: mismo prompt_fingerprint en todas las fuentes
    uniq_fp = set(fingerprints.values())
    same_config = len(uniq_fp) == 1

    result: dict = {
        "kind": "generation_robustness_rescore",
        "prompt_fingerprints": fingerprints,
        "same_config": same_config,
        "per_source": {label: _rescore(rows, ak) for label, rows in sources.items()},
    }
    combined_rows = [r for rows in sources.values() for r in rows]
    result["combined"] = _rescore(combined_rows, ak)

    # salida legible
    def _fmt(block: dict) -> str:
        m = block["mean"]
        ci = block.get("ci")
        s = f"n={block['n']} mean={m:.3f}" if m is not None else f"n={block['n']} mean=None"
        if ci:
            s += f" IC=[{ci['ci_low']:.3f},{ci['ci_high']:.3f}]"
        return s

    print(f"prompt_fingerprints: {fingerprints}  | config idéntica: {same_config}")
    if not same_config:
        print("  ¡AVISO! Los reports NO comparten prompt_fingerprint: la combinación NO es válida.")
    for label in [*sources.keys(), "combined"]:
        r = result["per_source"][label] if label in sources else result["combined"]
        print(f"\n### {label}")
        oa = r["over_abstention_rate"]
        print(f"  answerable={r['answerable']} answered={r['answered']} "
              f"over_abstention={oa:.3f}" if oa is not None else "  (sin answerable)")
        print(f"  key_fact   {_fmt(r['key_fact'])}")
        print(f"  citas P    n={r['citation_precision']['n']} mean="
              f"{r['citation_precision']['mean']:.3f}")
        print(f"  citas R    n={r['citation_recall']['n']} mean={r['citation_recall']['mean']:.3f}")
        print(f"  citas F1   {_fmt(r['citation_f1'])}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[guardado] {args.out}")


if __name__ == "__main__":
    main()
