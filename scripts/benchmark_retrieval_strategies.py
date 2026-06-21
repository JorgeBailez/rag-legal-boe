"""Comparación de estrategias de retrieval: denso vs BM25 vs híbrido sobre un bundle y un split.

Es el experimento central del TFG (retrieval, nivel L1 → no usa el juez). Construye los tres
recuperadores UNA vez sobre el mismo bundle/corpus (comparten encoder e índices) y evalúa el split
del banco con las mismas métricas que el benchmark denso, escribiendo un report versionado con el IC
por estrategia y el bootstrap pareado de cada estrategia frente al denso.

Uso:
    uv run python scripts/benchmark_retrieval_strategies.py \
      --bundle data/indexes/dense/<bundle_id> --split development \
      --strategies dense,bm25,hybrid_rrf --allow-incomplete-dataset

Gate C bloquea por defecto si el banco no está revisado; `--allow-incomplete-dataset` permite la
corrida informativa (de-riesgo) sobre el banco borrador. La corrida formal exige Gate C.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

KNOWN_STRATEGIES = ("dense", "bm25", "hybrid_rrf", "hybrid_weighted")
_DENSE_STRATEGIES = ("dense", "hybrid_rrf", "hybrid_weighted")
_LEXICAL_STRATEGIES = ("bm25", "hybrid_rrf", "hybrid_weighted")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comparación denso vs BM25 vs híbrido (L1).")
    parser.add_argument("--bundle", help="ruta al bundle; fallback a GENERATION_DENSE_BUNDLE.")
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument(
        "--split", default="development", choices=["development", "test", "out_of_corpus"]
    )
    parser.add_argument(
        "--strategies",
        default="dense,bm25,hybrid_rrf",
        help=f"lista por comas; opciones: {', '.join(KNOWN_STRATEGIES)}.",
    )
    parser.add_argument("--query-profile-id", default=None)
    parser.add_argument("--retrieve-depth", type=int, default=DEFAULT_RETRIEVE_DEPTH)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--hybrid-candidates", type=int, default=100)
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="peso del denso en la fusión ponderada."
    )
    parser.add_argument("--bm25-no-stem", action="store_true")
    parser.add_argument("--bm25-no-stopwords", action="store_true")
    parser.add_argument(
        "--bm25-heading-boost",
        type=int,
        default=0,
        help="copias EXTRA de los tokens de la cabecera al indexar BM25 (0 = sin boost).",
    )
    parser.add_argument("--gate-c-level", default="formal", choices=["checkpoint", "formal"])
    parser.add_argument(
        "--allow-incomplete-dataset",
        action="store_true",
        help="permite la comparación sin Gate C (informativa, no formal).",
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--limit", type=int, default=None, help="evalúa solo N preguntas (smoke).")
    parser.add_argument("--output-root", default="data/processed/reports/dense")
    return parser.parse_args()


def _build_strategies(args: argparse.Namespace, bundle: str, corpus: dict) -> tuple[dict, dict]:
    """Construye los recuperadores pedidos (denso/léxico una vez, reutilizados por los híbridos)."""
    requested = [s.strip() for s in args.strategies.split(",") if s.strip()]
    unknown = [s for s in requested if s not in KNOWN_STRATEGIES]
    if unknown:
        raise ValueError(f"estrategias desconocidas: {unknown}; válidas: {KNOWN_STRATEGIES}")

    analyzer = SpanishAnalyzer(
        remove_stopwords=not args.bm25_no_stopwords, stem=not args.bm25_no_stem
    )
    need_dense = any(s in _DENSE_STRATEGIES for s in requested)
    need_lexical = any(s in _LEXICAL_STRATEGIES for s in requested)
    dense = (
        DenseRetriever.from_bundle(bundle, corpus=corpus, batch_size=args.batch_size)
        if need_dense
        else None
    )
    lexical = (
        LexicalRetriever.from_bundle(
            bundle, corpus=corpus, analyzer=analyzer, heading_boost=args.bm25_heading_boost
        )
        if need_lexical
        else None
    )

    builders = {
        "dense": lambda: dense,
        "bm25": lambda: lexical,
        "hybrid_rrf": lambda: HybridRetriever(
            dense=dense,
            lexical=lexical,
            fusion="rrf",
            rrf_k=args.rrf_k,
            candidates=args.hybrid_candidates,
        ),
        "hybrid_weighted": lambda: HybridRetriever(
            dense=dense,
            lexical=lexical,
            fusion="weighted",
            alpha=args.alpha,
            candidates=args.hybrid_candidates,
        ),
    }
    strategies = {name: builders[name]() for name in requested}
    meta = {
        "bundle_id": getattr(dense or lexical, "bundle_id", ""),
        "model_alias": getattr(dense, "model_alias", "bm25") if dense else "bm25",
        "resolved_query_profile_id": (
            dense.resolved_query_profile_id(args.query_profile_id) if dense else "lexical"
        ),
        "analyzer": analyzer.signature(),
        "bm25_heading_boost": args.bm25_heading_boost,
    }
    return strategies, meta


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
        print("Anota el banco o usa --allow-incomplete-dataset (no formal).", file=sys.stderr)
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

    try:
        strategies, meta = _build_strategies(args, bundle, corpus)
    except Exception as exc:  # noqa: BLE001 - fallo de construcción (bundle/modelo/estrategia)
        print(f"No se pudieron construir las estrategias: {exc}", file=sys.stderr)
        return 1

    bar = tqdm(total=len(strategies) * len(split_qs), desc="comparación", unit="q")

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
            seed=args.seed,
            on_progress=on_progress,
        )
    finally:
        bar.close()

    run_id = new_run_id("retrieval")
    summary = {
        "split": args.split,
        "gate_c_ready": ds["gate_c"]["ready"],
        "seed": args.seed,
        **meta,
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
    _print_summary(out, result["metrics_rows"], result["summary"])
    return 0


def _print_summary(out: Path, metrics_rows: list[dict], summary: dict) -> None:
    print(f"\nReport: {out}")
    print(f"métrica primaria: {PRIMARY_METRIC} · baseline: {summary['baseline']}")
    for row in metrics_rows:
        print(
            f"  {row['strategy']:>16}: {PRIMARY_METRIC}={row.get(PRIMARY_METRIC, 0.0):.3f} "
            f"ParentRecall@5={row.get('ParentRecall@5', 0.0):.3f} "
            f"lat_p50={row.get('retrieve_latency_p50_ms', 0.0):.1f}ms"
        )
    for paired in summary["paired_vs_baseline"]:
        d = paired["diff"]
        print(
            f"  Δ {paired['strategy']} − {paired['baseline']}: "
            f"{d['mean_diff']:+.3f} IC95%=[{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"
        )
    by_style = summary.get("stratified", {}).get("by_query_style", {})
    if by_style:
        styles = sorted({s for groups in by_style.values() for s in groups})
        print(f"\n  {PRIMARY_METRIC} por query_style:")
        print("    " + "estrategia".ljust(16) + "".join(s[:14].rjust(16) for s in styles))
        for name, groups in by_style.items():
            cells = "".join(
                (f"{groups[s][PRIMARY_METRIC]:.3f}(n{groups[s]['n']})" if s in groups else "-")
                .rjust(16)
                for s in styles
            )
            print("    " + name.ljust(16) + cells)


if __name__ == "__main__":
    raise SystemExit(main())
