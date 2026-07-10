"""Ablación de la FUSIÓN híbrida sobre un split: RRF (k=60) vs convexa min-max (barrido de α).

Compara denso (e5-large/I1) · BM25 (config ganadora, heading_boost) · híbrido RRF · híbrido convexo
para varios α, todo como "estrategias" vía `evaluate_retrieval_strategies` (ParentnDCG@10 + IC +
**pareado vs denso** + estratificación por estilo). Baseline = denso, para responder "¿el híbrido
bate al denso y por cuánto?". Construye denso y BM25 UNA vez y los reutiliza en todos los híbridos.

Literatura: RRF es robusto y no normaliza escalas (Cormack 2009); la combinación convexa bien
afinada puede batir a RRF (Bruch et al. 2023) → se prueban las dos.

Uso:
    uv run python scripts/experiments/ablate_fusion.py --bundle data/indexes/dense/<bundle_id> \
      --split development --gate-c-level checkpoint --bm25-heading-boost 3 --threads 24
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
from src.retrieval.dense_retriever import DenseRetriever  # noqa: E402
from src.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from src.retrieval.lexical_retriever import LexicalRetriever  # noqa: E402
from src.retrieval.text_analysis import SpanishAnalyzer  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablación de la fusión híbrida (RRF vs convexa).")
    p.add_argument("--bundle", help="ruta al bundle; fallback a GENERATION_DENSE_BUNDLE.")
    p.add_argument("--dataset-dir", default=str(DATASET_DIR))
    p.add_argument(
        "--split", default="development", choices=["development", "test", "out_of_corpus"]
    )
    p.add_argument(
        "--query-profile-id", default="I1_LEGAL", help="perfil del denso (ganador OE-03)."
    )
    p.add_argument("--bm25-heading-boost", type=int, default=3, help="config BM25 ganadora.")
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument(
        "--alphas", default="0.3,0.4,0.5,0.6,0.7", help="α de la convexa (peso del denso)."
    )
    p.add_argument("--hybrid-candidates", type=int, default=100)
    p.add_argument("--retrieve-depth", type=int, default=DEFAULT_RETRIEVE_DEPTH)
    p.add_argument("--gate-c-level", default="formal", choices=["checkpoint", "formal"])
    p.add_argument("--allow-incomplete-dataset", action="store_true")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output-root", default="data/processed/reports/dense")
    return p.parse_args()


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

    print(f"Construyendo denso + BM25(hb={args.bm25_heading_boost}) y los híbridos…")
    dense = DenseRetriever.from_bundle(bundle, corpus=corpus, batch_size=args.batch_size)
    bm25 = LexicalRetriever.from_bundle(
        bundle, corpus=corpus, analyzer=SpanishAnalyzer(), heading_boost=args.bm25_heading_boost
    )

    def conv(alpha: float) -> HybridRetriever:
        return HybridRetriever(
            dense=dense,
            lexical=bm25,
            fusion="weighted",
            alpha=alpha,
            candidates=args.hybrid_candidates,
        )

    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    strategies: dict[str, object] = {
        "dense": dense,
        "bm25": bm25,
        "rrf": HybridRetriever(
            dense=dense,
            lexical=bm25,
            fusion="rrf",
            rrf_k=args.rrf_k,
            candidates=args.hybrid_candidates,
        ),
    }
    for a in alphas:
        strategies[f"conv_a{a:g}"] = conv(a)

    bar = tqdm(total=len(strategies) * len(split_qs), desc="ablación fusión", unit="q")

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
            query_profile_id=args.query_profile_id,
            baseline="dense",
            seed=args.seed,
            on_progress=on_progress,
        )
    finally:
        bar.close()

    run_id = new_run_id("fusionabl")
    summary = {
        "split": args.split,
        "gate_c_ready": ds["gate_c"]["ready"],
        "seed": args.seed,
        "ablation": "fusion",
        "bm25_heading_boost": args.bm25_heading_boost,
        "rrf_k": args.rrf_k,
        "alphas": alphas,
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

    def cell(label: str, style: str) -> str:
        g = by_style.get(label, {}).get(style)
        return f"{g[PRIMARY_METRIC]:.3f}" if g else "-"

    print(f"\nReport: {out}")
    print(f"métrica {PRIMARY_METRIC} · baseline {summary['baseline']} · n={rows[0]['n_queries']}\n")
    print(f"  {'estrategia':14}{'nDCG@10':>9}{'Rec@10':>9}{'directa':>9}{'lexica':>8}{'lat':>8}")
    for r in sorted(rows, key=lambda x: -x.get(PRIMARY_METRIC, 0.0)):
        lbl = r["strategy"]
        print(
            f"  {lbl:14}{r.get(PRIMARY_METRIC, 0.0):>9.3f}{r.get('ParentRecall@10', 0.0):>9.3f}"
            f"{cell(lbl, 'directa_articulo'):>9}{cell(lbl, 'lexica'):>8}"
            f"{r.get('retrieve_latency_p50_ms', 0.0):>8.1f}"
        )
    print("\n  Δ vs denso (pareado, IC95%):")
    for pv in summary["paired_vs_baseline"]:
        d = pv["diff"]
        sig = "SIG" if (d["ci_low"] > 0 or d["ci_high"] < 0) else "n.s."
        print(
            f"    {pv['strategy']:14} {d['mean_diff']:+.4f} "
            f"[{d['ci_low']:+.4f},{d['ci_high']:+.4f}] {sig}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
