"""Ablacion del ensamblado de contexto (L2): estrategia x k x presupuesto, SIN LLM ni juez.

Experimento E1 a nivel L2 de la hoja de ruta (`docs/hoja_de_ruta_experimental.md`). Sobre el
recuperador denso ganador (e5-large-instruct / I1_LEGAL), barre las tres estrategias de ensamblado
(K_ONLY, P_EXPAND_BOUNDED, P_EXPAND_FULL) x el numero de parents en contexto (k) x el presupuesto de
caracteres, y mide la calidad del CONTEXTO contra el gold de evidencia:

- ContextEvidenceRecall (primaria L2): fraccion de parrafos-evidencia cubiertos por el contexto.
- ContextPrecisionById / ContextRecallById: pureza / cobertura a nivel de parent.
- ContextCharacters, ContextItemCount, RedundantContextRate: coste y ruido del contexto.
- EvidenceDensity: fraccion de parrafos del contexto que son evidencia (senal/ruido intra-contexto;
  distingue BOUNDED de FULL, que ContextPrecisionById no separa por ser a nivel de articulo).

Regla de decision: elegir el (estrategia, k, presupuesto) con mayor ContextEvidenceRecall en la
RODILLA de coste (donde la cobertura satura y los caracteres/redundancia siguen creciendo). El
presupuesto solo afecta a P_EXPAND_BOUNDED; K_ONLY/P_EXPAND_FULL lo ignoran (una sola corrida).

La recuperacion se hace UNA vez por pregunta (top-`retrieve-depth`) y se reutiliza en todas las
configuraciones: solo cambia el ensamblado (barato, sin modelo). No reescribe el texto legal.

Uso:
    uv run python scripts/experiments/ablate_context.py --bundle data/indexes/dense/<bundle_id> \
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
from src.evaluation.dataset import load_and_validate, load_jsonl  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    aggregate_metrics,
    bootstrap_ci,
    context_metrics,
)
from src.evaluation.reports import new_run_id, write_benchmark_report  # noqa: E402
from src.evaluation.retrieval_eval import DEFAULT_RETRIEVE_DEPTH  # noqa: E402
from src.retrieval.context_assembler import (  # noqa: E402
    BUDGETS,
    K_ONLY,
    P_EXPAND_BOUNDED,
    P_EXPAND_FULL,
    assemble_context,
)
from src.retrieval.dense_retriever import DenseRetriever  # noqa: E402

CTX_PRIMARY = "ContextEvidenceRecall"
ALL_STRATEGIES = (K_ONLY, P_EXPAND_BOUNDED, P_EXPAND_FULL)
# Banco vigente del TFG (corpus-92), NO el MVP de 10 normas (DATASET_DIR del proyecto = MVP).
CORPUS92_DIR = "data/evaluation/corpus92_v1"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablacion del ensamblado de contexto (L2).")
    p.add_argument("--bundle", help="ruta al bundle denso; fallback a GENERATION_DENSE_BUNDLE.")
    p.add_argument("--dataset-dir", default=CORPUS92_DIR, help="banco; por defecto corpus-92.")
    p.add_argument(
        "--split", default="development", choices=["development", "test", "out_of_corpus"]
    )
    p.add_argument(
        "--query-profile-id", default="I1_LEGAL", help="perfil del denso (ganador OE-03)."
    )
    p.add_argument("--ks", default="1,3,5,8", help="numeros de parents en contexto a barrer.")
    p.add_argument(
        "--budgets", default="B4K,B8K,B12K", help="presupuestos (solo P_EXPAND_BOUNDED)."
    )
    p.add_argument(
        "--strategies",
        default="K_ONLY,P_EXPAND_BOUNDED,P_EXPAND_FULL",
        help="estrategias a barrer (coma).",
    )
    p.add_argument("--retrieve-depth", type=int, default=DEFAULT_RETRIEVE_DEPTH)
    p.add_argument("--gate-c-level", default="checkpoint", choices=["checkpoint", "formal"])
    p.add_argument("--allow-incomplete-dataset", action="store_true")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output-root", default="data/processed/reports/dense")
    return p.parse_args()


def _gold_for_query(judgments_q: list[dict]) -> tuple[set[str], dict[str, list[int]]]:
    """Parents relevantes (rel>=1) y parrafos-evidencia por parent, del gold de la pregunta."""
    relevant = {j["parent_id"] for j in judgments_q if j.get("relevance", 0) >= 1}
    evidence: dict[str, list[int]] = {}
    for j in judgments_q:
        if j.get("relevance", 0) < 1:
            continue
        orders = (j.get("evidence") or {}).get("paragraph_orders", []) or []
        if orders:
            evidence.setdefault(j["parent_id"], []).extend(orders)
    return relevant, evidence


def _unique_parent_hits(hits: list, k: int) -> list:
    """Primeros k hits de parents distintos, en orden de ranking (dedup parent-level)."""
    seen: set[str] = set()
    out: list = []
    for h in hits:
        if h.parent_id in seen:
            continue
        seen.add(h.parent_id)
        out.append(h)
        if len(out) >= k:
            break
    return out


def _assemble_for_query(
    hits_k: list, strategy: str, budget: int, parents_by_id: dict
) -> list[dict]:
    """Ensambla el contexto de los k parents segun la estrategia (ContextResult.as_dict())."""
    contexts: list[dict] = []
    for h in hits_k:
        parent = parents_by_id.get(h.parent_id)
        if not parent:
            continue
        # P_EXPAND_BOUNDED exige anchor; si falta, ese item cae a K_ONLY (sin romper la corrida).
        strat = strategy
        if strategy == P_EXPAND_BOUNDED and h.context_anchor is None:
            strat = K_ONLY
        retrieved_text = h.retrieval_text if strat in (K_ONLY, P_EXPAND_BOUNDED) else ""
        ctx = assemble_context(
            strategy=strat,
            parent=parent,
            anchor=h.context_anchor,
            retrieved_text=retrieved_text,
            budget_chars=budget,
        )
        contexts.append(ctx.as_dict())
    return contexts


def _evidence_density(contexts: list[dict], evidence_by_parent: dict[str, list[int]]) -> float:
    """Fraccion de parrafos del contexto que son evidencia anotada (senal/ruido intra-contexto).

    Distingue BOUNDED de FULL (que ContextPrecisionById no ve, por ser a nivel de articulo): FULL
    diluye la evidencia con parrafos no-evidencia del mismo articulo -> densidad menor.
    """
    covered: dict[str, set[int]] = {}
    for c in contexts:
        covered.setdefault(c["parent_id"], set()).update(c["paragraph_orders"])
    covered_ev = sum(
        len(set(orders) & covered.get(pid, set())) for pid, orders in evidence_by_parent.items()
    )
    total_paras = sum(len(c["paragraph_orders"]) for c in contexts)
    return covered_ev / total_paras if total_paras else 0.0


def _build_configs(
    strategies: list[str], budget_labels: list[str], ks: list[int]
) -> list[tuple[str, str, int, int]]:
    """Rejilla (estrategia, etiqueta, presupuesto, k); el presupuesto solo varia en BOUNDED."""
    configs: list[tuple[str, str, int, int]] = []
    for strat in strategies:
        if strat == P_EXPAND_BOUNDED:
            for blabel in budget_labels:
                for k in ks:
                    configs.append((strat, blabel, BUDGETS[blabel], k))
        else:
            for k in ks:
                configs.append((strat, "-", BUDGETS["B8K"], k))
    return configs


def main() -> int:
    args = _parse_args()
    settings = get_settings()
    bundle = args.bundle if args.bundle is not None else settings.generation_dense_bundle
    if not bundle:
        print("Falta el bundle. Indica --bundle o GENERATION_DENSE_BUNDLE.", file=sys.stderr)
        return 2

    set_cpu_threads(args.threads)
    corpus = load_processed_corpus()
    parents_by_id = corpus["parents_by_id"]
    ds = load_and_validate(Path(args.dataset_dir), corpus=corpus, gate_c_level=args.gate_c_level)
    if not ds["gate_c"]["ready"] and not args.allow_incomplete_dataset:
        print("Gate C no listo: el banco no esta revisado.", file=sys.stderr)
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

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    budget_labels = [b for b in args.budgets.split(",") if b.strip()]
    strategies = [s for s in args.strategies.split(",") if s.strip()]
    for s in strategies:
        if s not in ALL_STRATEGIES:
            print(f"Estrategia desconocida: {s!r} (esperado {ALL_STRATEGIES}).", file=sys.stderr)
            return 2

    print(f"Recuperando top-{args.retrieve_depth} por pregunta (una vez) y barriendo contexto...")
    dense = DenseRetriever.from_bundle(bundle, corpus=corpus, batch_size=args.batch_size)
    hits_by_qid: dict[str, list] = {}
    gold_by_qid: dict[str, tuple[set[str], dict[str, list[int]]]] = {}
    for q in tqdm(split_qs, desc="retrieval", unit="q"):
        hits_by_qid[q["query_id"]] = dense.retrieve(
            q["query"], query_profile_id=args.query_profile_id, top_k=args.retrieve_depth
        )
        gold_by_qid[q["query_id"]] = _gold_for_query(by_q.get(q["query_id"], []))

    configs = _build_configs(strategies, budget_labels, ks)
    metrics_rows: list[dict] = []
    bar = tqdm(total=len(configs), desc="configs de contexto", unit="cfg")
    for strat, blabel, budget, k in configs:
        per_query: list[dict] = []
        for q in split_qs:
            qid = q["query_id"]
            relevant, evidence = gold_by_qid[qid]
            hits_k = _unique_parent_hits(hits_by_qid[qid], k)
            contexts = _assemble_for_query(hits_k, strat, budget, parents_by_id)
            m = context_metrics(contexts, relevant_parents=relevant, evidence_by_parent=evidence)
            m["EvidenceDensity"] = _evidence_density(contexts, evidence)
            per_query.append(m)
        agg = aggregate_metrics(per_query)
        ci = bootstrap_ci([m[CTX_PRIMARY] for m in per_query], seed=args.seed)
        metrics_rows.append(
            {
                "config": f"{strat}|{blabel}|k{k}",
                "strategy": strat,
                "budget": blabel,
                "k": k,
                "n_queries": len(per_query),
                "EvRecall_ci_low": round(ci["ci_low"], 4),
                "EvRecall_ci_high": round(ci["ci_high"], 4),
                **agg,
            }
        )
        bar.update(1)
    bar.close()

    run_id = new_run_id("ctxabl")
    summary = {
        "split": args.split,
        "gate_c_ready": ds["gate_c"]["ready"],
        "seed": args.seed,
        "ablation": "context_assembly",
        "primary_metric": CTX_PRIMARY,
        "query_profile_id": args.query_profile_id,
        "retrieve_depth": args.retrieve_depth,
        "ks": ks,
        "budgets": budget_labels,
        "strategies": strategies,
        "bundle": str(bundle),
    }
    out = write_benchmark_report(
        run_id,
        summary=summary,
        metrics_rows=metrics_rows,
        query_results=[],
        context_results=[],
        reports_root=Path(args.output_root),
    )
    _print_table(out, metrics_rows)
    return 0


def _print_table(out: Path, rows: list[dict]) -> None:
    print(f"\nReport: {out}")
    print(f"metrica primaria L2: {CTX_PRIMARY} (cobertura de la evidencia por el contexto)\n")
    print(f"  {'config':24}{'EvRecall':>9}{'EvDens':>8}{'PrecById':>9}{'Chars':>8}{'Items':>7}")
    # Orden: mayor cobertura de evidencia y, a igualdad, menos caracteres (rodilla coste/cobertura).
    rows_sorted = sorted(
        rows, key=lambda x: (-x.get(CTX_PRIMARY, 0.0), x.get("ContextCharacters", 0.0))
    )
    for r in rows_sorted:
        print(
            f"  {r['config']:24}{r.get(CTX_PRIMARY, 0.0):>9.3f}"
            f"{r.get('EvidenceDensity', 0.0):>8.3f}{r.get('ContextPrecisionById', 0.0):>9.3f}"
            f"{r.get('ContextCharacters', 0.0):>8.0f}{r.get('ContextItemCount', 0.0):>7.1f}"
        )
    print(
        "\n  Rodilla = menor k/estrategia/presupuesto cuya EvRecall ya no mejora apreciablemente\n"
        "  (IC solapado) sin disparar Chars/Redund: esa es la config de contexto."
    )


if __name__ == "__main__":
    raise SystemExit(main())
