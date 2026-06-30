"""Ablación de BM25 (OFAT) sobre un split del banco: stopwords · stemming · heading_boost · k1 · b.

Compara cada config de BM25 como una «estrategia», reusando `evaluate_retrieval_strategies`
(mismas métricas L1 que el flagship: ParentnDCG@10 + IC bootstrap + **pareado vs baseline** +
estratificación por `query_style`). BM25 no carga el modelo de embeddings (solo tokeniza la query),
así que la ablación es barata en CPU. Un solo report versionado con una fila por configuración.

Diseño OFAT (one-factor-at-a-time) desde la baseline `base` (stopwords+stemming ON, heading_boost=0,
k1=1.5, b=0.75 = defaults de rank_bm25): se varía un knob cada vez. Tras leer el report se elige la
mejor combinación para una corrida de confirmación.

Uso:
    uv run python scripts/experiments/ablate_bm25.py --bundle data/indexes/dense/<bundle_id> \
      --split development --gate-c-level checkpoint --threads 24
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tqdm import tqdm  # noqa: E402

from src.config.settings import get_settings  # noqa: E402
from src.embeddings.corpus_loader import load_processed_corpus  # noqa: E402
from src.embeddings.encoder import set_cpu_threads  # noqa: E402
from src.evaluation.dataset import DATASET_DIR, load_and_validate, load_jsonl  # noqa: E402
from src.evaluation.metrics import PRIMARY_METRIC  # noqa: E402
from src.evaluation.reports import new_run_id, write_benchmark_report  # noqa: E402
from src.evaluation.retrieval_eval import (  # noqa: E402
    DEFAULT_RETRIEVE_DEPTH,
    evaluate_retrieval_strategies,
)
from src.retrieval.lexical_retriever import LexicalRetriever  # noqa: E402
from src.retrieval.text_analysis import SpanishAnalyzer  # noqa: E402

BASE = {"remove_stopwords": True, "stem": True, "heading_boost": 0, "k1": 1.5, "b": 0.75}
CONFIGS: list[tuple[str, dict]] = [
    ("base", BASE),
    ("no_stopwords", {**BASE, "remove_stopwords": False}),
    ("no_stem", {**BASE, "stem": False}),
    ("hb1", {**BASE, "heading_boost": 1}),
    ("hb2", {**BASE, "heading_boost": 2}),
    ("hb3", {**BASE, "heading_boost": 3}),
    ("k1_0.9", {**BASE, "k1": 0.9}),
    ("k1_1.2", {**BASE, "k1": 1.2}),
    ("k1_2.0", {**BASE, "k1": 2.0}),
    ("b_0.3", {**BASE, "b": 0.3}),
    ("b_0.5", {**BASE, "b": 0.5}),
    ("b_0.9", {**BASE, "b": 0.9}),
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablación OFAT de BM25 (L1).")
    p.add_argument("--bundle", help="ruta al bundle; fallback a GENERATION_DENSE_BUNDLE.")
    p.add_argument("--dataset-dir", default=str(DATASET_DIR))
    p.add_argument(
        "--split", default="development", choices=["development", "test", "out_of_corpus"]
    )
    p.add_argument("--retrieve-depth", type=int, default=DEFAULT_RETRIEVE_DEPTH)
    p.add_argument("--gate-c-level", default="formal", choices=["checkpoint", "formal"])
    p.add_argument("--allow-incomplete-dataset", action="store_true")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output-root", default="data/processed/reports/dense")
    return p.parse_args()


def _build(bundle: str, corpus: dict, cfg: dict) -> LexicalRetriever:
    analyzer = SpanishAnalyzer(remove_stopwords=cfg["remove_stopwords"], stem=cfg["stem"])
    return LexicalRetriever.from_bundle(
        bundle,
        corpus=corpus,
        analyzer=analyzer,
        heading_boost=cfg["heading_boost"],
        k1=cfg["k1"],
        b=cfg["b"],
    )


def main() -> int:
    args = _parse_args()
    settings = get_settings()
    bundle = args.bundle if args.bundle is not None else settings.generation_dense_bundle
    if not bundle:
        print("Falta el bundle. Indica --bundle o GENERATION_DENSE_BUNDLE.", file=sys.stderr)
        return 2

    set_cpu_threads(args.threads)
    corpus = load_processed_corpus()
    ds = load_and_validate(Path(args.dataset_dir), corpus=corpus, gate_c_level=args.gate_c_level)
    if not ds["gate_c"]["ready"] and not args.allow_incomplete_dataset:
        print("Gate C no listo: el banco no está revisado.", file=sys.stderr)
        for r in ds["gate_c"]["reasons"]:
            print(f"  - {r}", file=sys.stderr)
        return 1

    questions = load_jsonl(Path(args.dataset_dir) / "questions.jsonl")
    judgments = load_jsonl(Path(args.dataset_dir) / "judgments.jsonl")
    by_q: dict[str, list[dict]] = {}
    for j in judgments:
        by_q.setdefault(j["query_id"], []).append(j)
    split_qs = [q for q in questions if q["split"] == args.split]
    if args.limit is not None:
        split_qs = split_qs[: args.limit]
    if not split_qs:
        print(f"No hay preguntas en el split {args.split!r}.", file=sys.stderr)
        return 1

    print(f"Construyendo {len(CONFIGS)} configuraciones BM25 sobre {len(split_qs)} preguntas…")
    strategies = {label: _build(bundle, corpus, cfg) for label, cfg in CONFIGS}

    bar = tqdm(total=len(strategies) * len(split_qs), desc="ablación BM25", unit="q")

    def on_progress(ev: dict) -> None:
        if ev.get("event") == "query":
            bar.set_postfix_str(ev["strategy"])
            bar.update(1)

    try:
        result = evaluate_retrieval_strategies(
            strategies=strategies,
            split_questions=split_qs,
            judgments_by_query=by_q,
            retrieve_depth=args.retrieve_depth,
            baseline="base",
            seed=args.seed,
            on_progress=on_progress,
        )
    finally:
        bar.close()

    run_id = new_run_id("bm25abl")
    summary = {
        "split": args.split,
        "gate_c_ready": ds["gate_c"]["ready"],
        "seed": args.seed,
        "ablation": "bm25_ofat",
        "configs": {label: cfg for label, cfg in CONFIGS},
        **result["summary"],
    }
    out = write_benchmark_report(
        run_id,
        summary=summary,
        metrics_rows=result["metrics_rows"],
        query_results=result["query_results"],
        context_results=[],
        reports_root=Path(args.output_root),
    )
    _print_table(out, result["metrics_rows"], result["summary"])
    return 0


def _print_table(out: Path, rows: list[dict], summary: dict) -> None:
    by_style = summary.get("stratified", {}).get("by_query_style", {})

    def style_cell(label: str, style: str) -> str:
        g = by_style.get(label, {}).get(style)
        return f"{g[PRIMARY_METRIC]:.3f}" if g else "-"

    print(f"\nReport: {out}")
    print(f"métrica {PRIMARY_METRIC} · baseline {summary['baseline']} · n={rows[0]['n_queries']}\n")
    print(f"  {'config':14}{'nDCG@10':>9}{'Rec@10':>9}{'directa':>9}{'lexica':>8}{'lat':>7}")
    for r in sorted(rows, key=lambda x: -x.get(PRIMARY_METRIC, 0.0)):
        lbl = r["strategy"]
        print(
            f"  {lbl:14}{r.get(PRIMARY_METRIC, 0.0):>9.3f}{r.get('ParentRecall@10', 0.0):>9.3f}"
            f"{style_cell(lbl, 'directa_articulo'):>9}{style_cell(lbl, 'lexica'):>8}"
            f"{r.get('retrieve_latency_p50_ms', 0.0):>7.1f}"
        )
    print("\n  Δ vs base (pareado, IC95%):")
    for pv in summary["paired_vs_baseline"]:
        d = pv["diff"]
        sig = "SIG" if (d["ci_low"] > 0 or d["ci_high"] < 0) else "n.s."
        print(
            f"    {pv['strategy']:14} {d['mean_diff']:+.4f} "
            f"[{d['ci_low']:+.4f},{d['ci_high']:+.4f}] {sig}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
