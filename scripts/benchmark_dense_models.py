"""Benchmark / smoke test de los modelos densos (CPU).

Modos:
- `--smoke-test`: codifica una muestra pequeña con cada modelo y mide dimensión, throughput,
  latencia y RAM pico (cuando está disponible). Sirve para validar el flujo y el coste por modelo.
- (por defecto) benchmark formal: recupera sobre los bundles publicados usando el dataset de
  evaluación y calcula métricas. **Gate C** bloquea si el dataset no está anotado/revisado.

No se ejecuta automáticamente en los tests: requiere pesos reales y/o bundles publicados.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings import model_registry as reg  # noqa: E402
from src.embeddings.corpus_loader import load_processed_corpus  # noqa: E402
from src.embeddings.encoder import DenseEncoder, load_tokenizer, set_cpu_threads  # noqa: E402
from src.embeddings.input_preparation import prepare_inputs  # noqa: E402
from src.embeddings.model_registry import (  # noqa: E402
    assert_bundle_compatible,
    default_query_profile_id_for_contract,
    effective_query_profile_ids,
    query_profile_metadata,
)
from src.evaluation.dataset import (  # noqa: E402
    DATASET_DIR,
    load_and_validate,
    load_jsonl,
)
from src.evaluation.metrics import (  # noqa: E402
    CONTEXT_KS,
    PRIMARY_METRIC,
    RETRIEVAL_KS,
    abstention_threshold_analysis,
    aggregate_metric_groups,
    aggregate_metrics,
    bootstrap_ci,
    compute_query_retrieval_metrics,
    context_metrics,
    paired_bootstrap,
    paired_vs_baseline,
    pareto_front,
)
from src.evaluation.reports import (  # noqa: E402
    new_run_id,
    write_benchmark_report,
    write_smoke_report,
)
from src.indexing.vector_index import ExactDenseIndex  # noqa: E402
from src.retrieval.context_assembler import (  # noqa: E402
    BUDGETS,
    K_ONLY,
    P_EXPAND_BOUNDED,
    P_EXPAND_FULL,
    assemble_context,
)

SAMPLE_QUERIES = [
    "¿Cuánto tiempo tiene la Administración para resolver mi solicitud?",
    "¿Qué sistemas de identificación electrónica puedo usar?",
    "¿Qué efectos tiene el silencio administrativo?",
]


def _peak_ram_mb() -> float | None:
    try:
        import resource  # Linux/macOS; ausente en Windows
    except ImportError:
        return None
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(ru / 1024, 1)  # En Linux, ru_maxrss se reporta en KB.


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


def _latency_summary_ms(query_latencies_ms: list[float], search_latencies_ms: list[float]) -> dict:
    return {
        "query_embedding_latency_p50_ms": round(_percentile(query_latencies_ms, 0.50), 3),
        "query_embedding_latency_p95_ms": round(_percentile(query_latencies_ms, 0.95), 3),
        "exact_search_latency_p50_ms": round(_percentile(search_latencies_ms, 0.50), 3),
        "exact_search_latency_p95_ms": round(_percentile(search_latencies_ms, 0.95), 3),
        "latency_sample_count": len(query_latencies_ms),
    }


def _dir_size_bytes(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    total = 0
    try:
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    except OSError:
        return None
    return total


def _hf_cache_size_bytes(model_id: str) -> int | None:
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = cache_root / f"models--{model_id.replace('/', '--')}"
    return _dir_size_bytes(repo_dir)


def _select_representative_chunks(chunks: list[dict], n_docs: int) -> list[dict]:
    """Muestra pequeña, determinista y más útil que `chunks[:n]`."""
    if n_docs <= 0:
        return []
    ordered = sorted(chunks, key=lambda c: c.get("chunk_id", ""))
    by_len = sorted(
        ordered, key=lambda c: (len(c.get("retrieval_text", "")), c.get("chunk_id", ""))
    )
    picks: list[dict] = []

    def add(chunk: dict | None) -> None:
        if chunk is not None and chunk not in picks:
            picks.append(chunk)

    if by_len:
        add(by_len[0])
        add(by_len[len(by_len) // 2])
        add(by_len[-1])
    wanted_roles = ("preamble", "precept", "annex")
    for role in wanted_roles:
        add(
            next(
                (c for c in ordered if (c.get("filters") or {}).get("semantic_role") == role),
                None,
            )
        )
    add(next((c for c in ordered if (c.get("filters") or {}).get("table")), None))
    for chunk in by_len:
        add(chunk)
        if len(picks) >= n_docs:
            break
    return picks[:n_docs]


def _prepare_smoke_documents(
    contract, corpus: dict, tokenizer, n_docs: int
) -> tuple[list[str], dict]:
    sample_chunks = _select_representative_chunks(corpus["chunks"], n_docs)
    prepared = prepare_inputs(
        "J1",
        chunks=sample_chunks,
        parents_by_id=corpus["parents_by_id"],
        contract=contract,
        tokenizer=tokenizer,
    )
    return prepared.texts, prepared.report


def _resolve_hit_text(hit: dict, corpus: dict) -> str:
    source = hit.get("source") or {}
    if source.get("kind") == "derived_text" and source.get("text") is not None:
        return source["text"]
    chunk_id = source.get("chunk_id")
    if chunk_id:
        chunk = {c["chunk_id"]: c for c in corpus["chunks"]}.get(chunk_id, {})
        return chunk.get("text", "")
    parent = corpus["parents_by_id"].get(hit["parent_id"], {})
    return source.get("text") or parent.get("text", "")


def _dedupe_hits_by_parent(hits: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for hit in hits:
        parent_id = hit["parent_id"]
        if parent_id in seen:
            continue
        seen.add(parent_id)
        out.append(hit)
    return out


def _hits_for_context_strategy(strategy: str, hits: list[dict]) -> list[dict]:
    return hits if strategy == K_ONLY else _dedupe_hits_by_parent(hits)


def _budget_runs_for_strategy(strategy: str) -> list[tuple[str | None, int]]:
    if strategy in (K_ONLY, P_EXPAND_FULL):
        return [(None, BUDGETS["B8K"])]
    return list(BUDGETS.items())


def _query_result_hit(hit: dict) -> dict:
    return {
        "rank": hit["rank"],
        "row_index": hit["row_index"],
        "embedding_input_id": hit["embedding_input_id"],
        "parent_id": hit["parent_id"],
        "source_chunk_id": (hit.get("source") or {}).get("chunk_id"),
        "context_anchor": hit.get("context_anchor"),
        "score": round(hit["score"], 4),
    }


def _evidence_by_parent(judgments: list[dict]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for j in judgments:
        if j.get("relevance", 0) >= 1:
            out.setdefault(j["parent_id"], []).extend(
                (j.get("evidence") or {}).get("paragraph_orders", [])
            )
    return out


def run_smoke(args: argparse.Namespace) -> int:
    set_cpu_threads(args.threads)
    corpus = load_processed_corpus()
    if not corpus["chunks"]:
        print("No hay corpus procesado (ejecuta el pipeline de Fase 1).", file=sys.stderr)
        return 1

    model_rows: list[dict] = []
    for name in args.models:
        contract = reg.get_contract(name)
        print(f"[smoke] {contract.alias} ({contract.model_id}) …")
        try:
            tokenizer = load_tokenizer(
                contract, allow_unpinned_revision=args.allow_unpinned_revision
            )
            sample_docs, sample_report = _prepare_smoke_documents(
                contract, corpus, tokenizer, args.n_docs
            )
            enc = DenseEncoder(
                contract,
                batch_size=args.batch_size,
                allow_unpinned_revision=args.allow_unpinned_revision,
            )
            t0 = time.perf_counter()
            emb = enc.encode_documents(sample_docs, show_progress=not args.no_progress)
            doc_secs = time.perf_counter() - t0
            lat = []
            for q in SAMPLE_QUERIES:
                t = time.perf_counter()
                enc.encode_queries([q], show_progress=False)
                lat.append(time.perf_counter() - t)
            model_rows.append(
                {
                    "model_alias": contract.alias,
                    "model_id": contract.model_id,
                    "model_revision": contract.model_revision,
                    "tokenizer_revision": contract.tokenizer_revision,
                    "exploratory_unpinned": args.allow_unpinned_revision,
                    "embedding_dimension": int(emb.shape[1]),
                    "observed_dimension": int(emb.shape[1]),
                    "n_source_chunks": sample_report["n_source_chunks"],
                    "n_docs": len(sample_docs),
                    "doc_encode_seconds": round(doc_secs, 3),
                    "doc_throughput_per_s": round(len(sample_docs) / doc_secs, 2)
                    if doc_secs
                    else 0.0,
                    "query_latency_p50_s": round(_percentile(lat, 0.50), 4),
                    "query_latency_p95_s": round(_percentile(lat, 0.95), 4),
                    "peak_ram_mb": _peak_ram_mb(),
                    "download_size_bytes": None,
                    "cache_size_bytes": _hf_cache_size_bytes(contract.model_id),
                    "warning": "",
                }
            )
        except Exception as exc:  # noqa: BLE001 - se registra el fallo por modelo, no aborta el lote
            model_rows.append(
                {
                    "model_alias": contract.alias,
                    "model_id": contract.model_id,
                    "warning": f"{type(exc).__name__}: {exc}",
                }
            )

    run_id = new_run_id("smoke")
    out = write_smoke_report(
        run_id,
        meta={
            "device": "cpu",
            "threads": args.threads,
            "batch_size": args.batch_size,
            "exploratory_unpinned": args.allow_unpinned_revision,
            "n_docs_requested": args.n_docs,
        },
        model_rows=model_rows,
        reports_root=Path(args.output_root),
    )
    print(f"\nSmoke report: {out}")
    return 0


def _baseline_run_key(metrics_rows: list[dict], baseline_alias: str) -> str | None:
    """run_key del baseline: su perfil por defecto, o el primero del modelo si no aparece."""
    rows = [r for r in metrics_rows if r["model_alias"] == baseline_alias]
    if not rows:
        return None
    try:
        default_profile = default_query_profile_id_for_contract(reg.get_contract(baseline_alias))
    except KeyError:
        return rows[0]["run_key"]
    for r in rows:
        if r["query_profile_id"] == default_profile:
            return r["run_key"]
    return rows[0]["run_key"]


def run_benchmark(args: argparse.Namespace) -> int:
    if args.allow_unpinned_revision:
        print("--allow-unpinned-revision solo se permite en smoke exploratorio.", file=sys.stderr)
        return 1
    corpus = load_processed_corpus()
    ds = load_and_validate(Path(args.dataset_dir), corpus=corpus, gate_c_level=args.gate_c_level)
    if not ds["gate_c"]["ready"] and not args.allow_incomplete_dataset:
        print("Gate C no listo: el dataset no tiene anotación revisada.", file=sys.stderr)
        for r in ds["gate_c"]["reasons"]:
            print(f"  - {r}", file=sys.stderr)
        print("Anota el dataset o usa --allow-incomplete-dataset (no formal).", file=sys.stderr)
        return 1

    questions = load_jsonl(Path(args.dataset_dir) / "questions.jsonl")
    judgments = load_jsonl(Path(args.dataset_dir) / "judgments.jsonl")
    by_q: dict[str, list[dict]] = {}
    for j in judgments:
        by_q.setdefault(j["query_id"], []).append(j)
    split_qs = [q for q in questions if q["split"] == args.split]
    ooc_qs = [q for q in questions if q["split"] == "out_of_corpus"]
    run_abstention = bool(ooc_qs) and not args.skip_abstention

    bundle_dirs = (
        [Path(b) for b in args.bundle]
        if args.bundle
        else sorted(p for p in Path(args.bundles_root).glob("*") if (p / "manifest.json").is_file())
    )
    if not bundle_dirs:
        print("No hay bundles que evaluar (genera alguno o pasa --bundle).", file=sys.stderr)
        return 1

    set_cpu_threads(args.threads)
    metrics_rows: list[dict] = []
    query_results: list[dict] = []
    summary_bundles: list[dict] = []
    context_results: list[dict] = []
    primary_values_by_run: dict[str, list[float]] = {}
    hits_by_run_query: dict[tuple[str, str], list[dict]] = {}
    stratified_by_run: dict[str, dict] = {}
    abstention_by_run: dict[str, dict] = {}

    for bundle_dir in bundle_dirs:
        index = ExactDenseIndex.from_bundle(bundle_dir, corpus=corpus)
        contract = reg.get_contract(index.manifest["bundle"]["model_alias"])
        assert_bundle_compatible(contract, index.manifest)
        enc = DenseEncoder(
            contract,
            batch_size=args.batch_size,
            allow_unpinned_revision=False,
        )
        bid = index.manifest["bundle"]["bundle_id"]
        try:
            profile_ids = effective_query_profile_ids(contract, args.query_profile_id)
        except (KeyError, ValueError) as exc:
            print(f"{bid}: {exc}", file=sys.stderr)
            return 1
        for profile_id in profile_ids:
            qp = query_profile_metadata(contract, profile_id)
            run_key = f"{bid}::{profile_id}"
            per_query: list[dict] = []
            query_latencies_ms: list[float] = []
            search_latencies_ms: list[float] = []
            top1_scores: list[float] = []
            if split_qs:
                warm_vec = enc.encode_queries(
                    [split_qs[0]["query"]],
                    query_profile_id=profile_id,
                    show_progress=False,
                )[0]
                index.search(warm_vec, k=1)
            for q in split_qs:
                t0 = time.perf_counter()
                qv = enc.encode_queries(
                    [q["query"]],
                    query_profile_id=profile_id,
                    show_progress=False,
                )[0]
                query_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                t1 = time.perf_counter()
                hits = index.search(qv, k=max(RETRIEVAL_KS + CONTEXT_KS))
                search_latencies_ms.append((time.perf_counter() - t1) * 1000.0)
                hits_by_run_query[(run_key, q["query_id"])] = hits
                top1_scores.append(hits[0]["score"] if hits else 0.0)
                m = compute_query_retrieval_metrics(hits, by_q.get(q["query_id"], []))
                per_query.append(m)
                query_results.append(
                    {
                        "bundle_id": bid,
                        "query_profile_id": profile_id,
                        "query_profile_fingerprint": qp["query_profile_fingerprint"],
                        "query_id": q["query_id"],
                        "split": q["split"],
                        "hits": [_query_result_hit(h) for h in hits],
                        "metrics": m,
                    }
                )
            agg = aggregate_metrics(per_query)
            metrics_rows.append(
                {
                    "run_key": run_key,
                    "bundle_id": bid,
                    "view": index.manifest["bundle"]["view"],
                    "model_alias": contract.alias,
                    "query_profile_id": profile_id,
                    "query_profile_fingerprint": qp["query_profile_fingerprint"],
                    **_latency_summary_ms(query_latencies_ms, search_latencies_ms),
                    **agg,
                }
            )
            primary_vals = [d[PRIMARY_METRIC] for d in per_query if PRIMARY_METRIC in d]
            primary_values_by_run[run_key] = primary_vals
            summary_bundles.append(
                {
                    "run_key": run_key,
                    "bundle_id": bid,
                    "query_profile_id": profile_id,
                    "n_queries": len(per_query),
                    "primary_metric": PRIMARY_METRIC,
                    "primary_ci": bootstrap_ci(primary_vals, seed=args.seed),
                }
            )
            # Cortes por tipo de pregunta y dificultad: dónde gana/pierde (no solo la media global).
            style_groups: dict[str, list[dict]] = {}
            diff_groups: dict[str, list[dict]] = {}
            for q, m in zip(split_qs, per_query, strict=True):
                style_groups.setdefault(q.get("query_style", "?"), []).append(m)
                diff_groups.setdefault(q.get("difficulty", "?"), []).append(m)
            stratified_by_run[run_key] = {
                "by_query_style": aggregate_metric_groups(style_groups, seed=args.seed),
                "by_difficulty": aggregate_metric_groups(diff_groups, seed=args.seed),
            }
            # Abstención (L6): ¿el score top-1 separa in-corpus de out_of_corpus?
            # Se desglosa el OOC en dos subconjuntos: 'far_domain' (materia ajena al corpus,
            # negativos fáciles) y 'near_miss' (misma materia pero respuesta ausente; query_id con
            # prefijo 'q92nm_'). El AUC global mezcla ambos y sobreestima la abstención, porque el
            # near-miss es el caso difícil. Se guardan además los scores top-1 por pregunta.
            if run_abstention:
                ooc_scored: list[tuple[str, float]] = []
                for q in ooc_qs:
                    qv_ooc = enc.encode_queries(
                        [q["query"]], query_profile_id=profile_id, show_progress=False
                    )[0]
                    h = index.search(qv_ooc, k=1)
                    ooc_scored.append((q["query_id"], h[0]["score"] if h else 0.0))
                far = [s for qid, s in ooc_scored if not qid.startswith("q92nm_")]
                near = [s for qid, s in ooc_scored if qid.startswith("q92nm_")]
                abst: dict = {
                    "all": abstention_threshold_analysis(top1_scores, [s for _, s in ooc_scored]),
                    "ooc_top1_scores": dict(ooc_scored),
                }
                if far:
                    abst["far_domain"] = abstention_threshold_analysis(top1_scores, far)
                if near:
                    abst["near_miss"] = abstention_threshold_analysis(top1_scores, near)
                abstention_by_run[run_key] = abst

    ranked_runs = sorted(
        metrics_rows, key=lambda r: float(r.get(PRIMARY_METRIC, 0.0)), reverse=True
    )
    paired = None
    if len(ranked_runs) >= 2:
        a_key = ranked_runs[0]["run_key"]
        b_key = ranked_runs[1]["run_key"]
        paired = {
            "a": a_key,
            "b": b_key,
            PRIMARY_METRIC: paired_bootstrap(
                primary_values_by_run[a_key], primary_values_by_run[b_key], seed=args.seed
            ),
        }

    # Pareado de CADA run contra el baseline (no solo top-1 vs top-2) → "¿X mejora al baseline?".
    baseline_vs = None
    base_key = _baseline_run_key(metrics_rows, args.baseline_alias)
    if base_key and len(primary_values_by_run) >= 2:
        baseline_vs = {
            "baseline_run_key": base_key,
            "diffs": paired_vs_baseline(primary_values_by_run, base_key, seed=args.seed),
        }

    # Frontera calidad/coste: ParentnDCG@10 vs latencia de embedding de query (despliegue CPU-only).
    cost_key = "query_embedding_latency_p50_ms"
    points = [
        {
            "run_key": r["run_key"],
            "model_alias": r["model_alias"],
            PRIMARY_METRIC: float(r.get(PRIMARY_METRIC, 0.0)),
            cost_key: float(r.get(cost_key, 0.0)),
        }
        for r in metrics_rows
    ]
    front = pareto_front(points, quality_key=PRIMARY_METRIC, cost_key=cost_key)
    efficiency = {
        "cost_metric": cost_key,
        "points": points,
        "pareto_front_run_keys": [p["run_key"] for p in front],
    }

    if args.context_ablations and ranked_runs:
        wanted_bundles = set(args.finalist_bundle or [])
        finalist_runs = [
            r for r in ranked_runs if not wanted_bundles or r["bundle_id"] in wanted_bundles
        ][:2]
        for run in finalist_runs:
            run_key = run["run_key"]
            for q in split_qs:
                judgments_for_q = by_q.get(q["query_id"], [])
                relevant = {j["parent_id"] for j in judgments_for_q if j.get("relevance", 0) >= 1}
                evidence = _evidence_by_parent(judgments_for_q)
                hits = hits_by_run_query[(run_key, q["query_id"])]
                for k in CONTEXT_KS:
                    top_hits = hits[:k]
                    for strategy in (K_ONLY, P_EXPAND_FULL, P_EXPAND_BOUNDED):
                        context_hits = _hits_for_context_strategy(strategy, top_hits)
                        for budget_name, budget in _budget_runs_for_strategy(strategy):
                            contexts = []
                            for hit in context_hits:
                                parent = corpus["parents_by_id"][hit["parent_id"]]
                                ctx = assemble_context(
                                    strategy=strategy,
                                    parent=parent,
                                    anchor=hit.get("context_anchor"),
                                    retrieved_text=_resolve_hit_text(hit, corpus)
                                    if strategy in (K_ONLY, P_EXPAND_BOUNDED)
                                    else "",
                                    budget_chars=budget,
                                )
                                contexts.append(ctx.as_dict())
                            metrics = context_metrics(
                                contexts,
                                relevant_parents=relevant,
                                evidence_by_parent=evidence,
                            )
                            context_results.append(
                                {
                                    "run_key": run_key,
                                    "bundle_id": run["bundle_id"],
                                    "query_profile_id": run["query_profile_id"],
                                    "query_id": q["query_id"],
                                    "k": k,
                                    "strategy": strategy,
                                    "budget": budget_name,
                                    "items": [
                                        {
                                            "parent_id": c["parent_id"],
                                            "paragraph_orders": c["paragraph_orders"],
                                            "char_count": c["char_count"],
                                            "item_count": c["item_count"],
                                            "over_budget": c["over_budget"],
                                            "fallback_reason": c["fallback_reason"],
                                        }
                                        for c in contexts
                                    ],
                                    "metrics": metrics,
                                }
                            )

    run_id = new_run_id("bench")
    out = write_benchmark_report(
        run_id,
        summary={
            "split": args.split,
            "primary_metric": PRIMARY_METRIC,
            "seed": args.seed,
            "gate_c_ready": ds["gate_c"]["ready"],
            "bundles": summary_bundles,
            "paired_bootstrap_finalists": paired,
            "paired_vs_baseline": baseline_vs,
            "stratified_by_run": stratified_by_run,
            "abstention_by_run": abstention_by_run if run_abstention else None,
            "efficiency_frontier": efficiency,
        },
        metrics_rows=metrics_rows,
        query_results=query_results,
        context_results=context_results,
        reports_root=Path(args.output_root),
    )
    print(f"\nBenchmark report: {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark / smoke de modelos densos (CPU).")
    parser.add_argument("--smoke-test", action="store_true", help="muestra pequeña por modelo.")
    parser.add_argument("--models", nargs="*", default=reg.all_aliases())
    parser.add_argument(
        "--query-profile-id",
        action="append",
        help="perfil de query a comparar; repetible. Default: perfiles efectivos por modelo.",
    )
    parser.add_argument("--bundle", action="append", help="bundle concreto (repetible).")
    parser.add_argument("--bundles-root", default="data/indexes/dense")
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument(
        "--gate-c-level",
        default="formal",
        choices=["checkpoint", "formal"],
        help="mínimos Gate C para considerar formal el benchmark.",
    )
    parser.add_argument(
        "--split", default="development", choices=["development", "test", "out_of_corpus"]
    )
    parser.add_argument("--output-root", default="data/processed/reports/dense")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-docs", type=int, default=64, help="docs de muestra para el smoke.")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--finalist-bundle", action="append", help="bundle_id finalista para contexto."
    )
    parser.add_argument("--context-ablations", action="store_true")
    parser.add_argument(
        "--baseline-alias",
        default="e5-large-instruct",
        help="modelo de referencia para el bootstrap pareado vs baseline.",
    )
    parser.add_argument(
        "--skip-abstention",
        action="store_true",
        help="no calcular el experimento de abstención sobre el split out_of_corpus.",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--allow-unpinned-revision", action="store_true")
    parser.add_argument(
        "--allow-incomplete-dataset",
        action="store_true",
        help="permite benchmark sin Gate C (no formal).",
    )
    args = parser.parse_args()
    return run_smoke(args) if args.smoke_test else run_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
