"""Análisis de robustez de recuperación combinando *development* + *test* (n=81).

Motivación (memoria, cap. de experimentos): la comparación confirmatoria de recuperadores se hace
sobre `test` held-out (n=28), que solo tiene potencia para efectos grandes (efecto mínimo detectable
≈0.07). Este script NO re-ejecuta ningún modelo: reutiliza los `per_query` de dos reports ya
versionados —la ablación de fusión sobre *development* y el flagship sobre *test*— y recompone el
análisis sobre el conjunto in-corpus completo (n=81) como **robustez** (no held-out: incluye el
split de ajuste, con un sesgo optimista leve que además favorece al híbrido, no al denso). Reporta:

  * media + IC BCa por estrategia (denso, BM25, RRF, ponderada α0.7) a n=81,
  * bootstrap **pareado** vs denso + corrección **Holm-Bonferroni** (familia confirmatoria),
  * **TOST** (equivalencia) de ponderada/RRF vs denso contra un margen de negligibilidad,
  * el desglose por estilo de consulta con n por estrato ya no minúsculo.

Uso:
    uv run python -m scripts.combine_retrieval_robustness \
        --dev-report data/processed/reports/dense/benchmarks/fusionabl_20260625T094731Z \
        --test-report data/processed/reports/dense/benchmarks/retrieval_20260625T111234Z \
        --questions data/evaluation/corpus92_v1/questions.jsonl \
        --margin 0.02
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.metrics import (
    DEFAULT_BOOTSTRAP_SEED,
    PRIMARY_METRIC,
    bca_ci,
    holm_correction,
    paired_bootstrap,
    paired_equivalence_tost,
)

# Nombre de estrategia en cada report -> nombre canónico. La ablación de fusión (dev) y el flagship
# (test) nombran distinto la misma estrategia; solo se conservan las cuatro del experimento central.
STRATEGY_ALIASES = {
    "dense": "dense",
    "bm25": "bm25",
    "rrf": "rrf",
    "hybrid_rrf": "rrf",
    "conv_a0.7": "weighted",
    "hybrid_weighted": "weighted",
}
CANONICAL = ("dense", "bm25", "rrf", "weighted")
BASELINE = "dense"


def _load_per_query(report_dir: Path, primary: str) -> dict[str, dict[str, float]]:
    """Devuelve {estrategia_canónica: {query_id: valor_primario}} de un report."""
    out: dict[str, dict[str, float]] = {s: {} for s in CANONICAL}
    path = report_dir / "query_results.jsonl"
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            canon = STRATEGY_ALIASES.get(row["strategy"])
            if canon is None:
                continue
            val = row["metrics"].get(primary)
            if val is not None:
                out[canon][row["query_id"]] = float(val)
    return out


def _style_map(questions_path: Path) -> dict[str, str]:
    styles: dict[str, str] = {}
    with questions_path.open(encoding="utf-8") as fh:
        for line in fh:
            q = json.loads(line)
            styles[q["query_id"]] = q.get("query_style", "?")
    return styles


def _aligned_vectors(
    combined: dict[str, dict[str, float]], query_ids: list[str]
) -> dict[str, list[float]]:
    """Vectores por estrategia alineados por query_id (para el pareado)."""
    return {s: [combined[s][qid] for qid in query_ids] for s in CANONICAL}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev-report", type=Path, required=True)
    ap.add_argument("--test-report", type=Path, required=True)
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument(
        "--margin", type=float, default=0.02, help="Margen TOST (unidades de la métrica)."
    )
    ap.add_argument("--primary", default=PRIMARY_METRIC)
    ap.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    ap.add_argument("--out", type=Path, default=None, help="Ruta JSON de salida (opcional).")
    args = ap.parse_args()

    dev = _load_per_query(args.dev_report, args.primary)
    test = _load_per_query(args.test_report, args.primary)
    styles = _style_map(args.questions)

    # Fusión: cada query_id es único y disjunto entre splits, así que combinar = unión.
    combined: dict[str, dict[str, float]] = {s: {**dev[s], **test[s]} for s in CANONICAL}
    # query_ids donde TODAS las estrategias tienen valor (garantiza el pareado)
    qids = sorted(set.intersection(*[set(combined[s]) for s in CANONICAL]))
    n = len(qids)
    vecs = _aligned_vectors(combined, qids)

    # --- agregados por estrategia (n=81) ---
    per_strategy = {s: {"n": n, "ci": bca_ci(vecs[s], seed=args.seed)} for s in CANONICAL}
    # --- pareado vs denso + Holm ---
    paired = {
        s: paired_bootstrap(vecs[s], vecs[BASELINE], seed=args.seed)
        for s in CANONICAL
        if s != BASELINE
    }
    holm = holm_correction({s: paired[s]["p_value"] for s in paired})
    # --- TOST equivalencia vs denso (para los nulos: ponderada y RRF) ---
    tost = {
        s: paired_equivalence_tost(vecs[s], vecs[BASELINE], margin=args.margin, seed=args.seed)
        for s in ("weighted", "rrf")
    }
    # --- estratificado por estilo (n por estrato ya no diminuto) ---
    by_style: dict[str, dict] = {}
    style_set = sorted({styles.get(q, "?") for q in qids})
    for st in style_set:
        st_qids = [q for q in qids if styles.get(q) == st]
        by_style[st] = {
            "n": len(st_qids),
            "means": {
                s: round(sum(combined[s][q] for q in st_qids) / len(st_qids), 4) for s in CANONICAL
            },
        }

    result = {
        "kind": "retrieval_robustness_devtest",
        "primary_metric": args.primary,
        "n_combined": n,
        "n_dev": len(dev["dense"]),
        "n_test": len(test["dense"]),
        "seed": args.seed,
        "margin_tost": args.margin,
        "baseline": BASELINE,
        "sources": {"dev": str(args.dev_report), "test": str(args.test_report)},
        "per_strategy": per_strategy,
        "paired_vs_dense": paired,
        "holm": holm,
        "tost_vs_dense": tost,
        "by_query_style": by_style,
    }

    # --- salida legible ---
    print(f"\n=== Robustez in-corpus (dev+test), n={n} · métrica={args.primary} ===")
    print(f"{'estrategia':<12}{'media':>8}{'IC95 BCa':>22}")
    for s in CANONICAL:
        c = per_strategy[s]["ci"]
        print(f"{s:<12}{c['mean']:>8.3f}   [{c['ci_low']:.3f}, {c['ci_high']:.3f}]")
    print(f"\n--- Pareado vs {BASELINE} (d = estrategia - denso) + Holm ---")
    for s in paired:
        p = paired[s]
        h = holm[s]
        print(
            f"{s:<12}d={p['mean_diff']:+.3f}  [{p['ci_low']:+.3f}, {p['ci_high']:+.3f}]  "
            f"p={p['p_value']:.3f}  p_holm={h['p_holm']:.3f}  sig={h['significant']}"
        )
    print(f"\n--- TOST equivalencia vs denso (margen +/-{args.margin}) ---")
    for s, t in tost.items():
        print(
            f"{s:<12}d={t['mean_diff']:+.3f}  IC90=[{t['ci_low']:+.3f}, {t['ci_high']:+.3f}]  "
            f"p_tost={t['p_tost']:.3f}  equivalente={t['equivalent']}"
        )
    print("\n--- Por estilo de consulta (medias) ---")
    print(f"{'estilo':<18}{'n':>3}  " + "".join(f"{s:>10}" for s in CANONICAL))
    for st, d in by_style.items():
        print(f"{st:<18}{d['n']:>3}  " + "".join(f"{d['means'][s]:>10.3f}" for s in CANONICAL))

    if args.out:
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[guardado] {args.out}")


if __name__ == "__main__":
    main()
