"""Métricas de evaluación del retrieval denso (parent-level + evidencia + contexto + bootstrap).

Métrica primaria: **ParentnDCG@10**. Controles obligatorios: ParentRecall@5, EvidenceRecall@5,
ParentHit@1. Todas son funciones puras y deterministas (el bootstrap usa una semilla fija que se
registra en el reporte).

Convenciones de relevancia (de los judgments): 2 = central/suficiente, 1 = apoyo/matiz, 0 =
descartado, ausencia = no juzgado. "Relevante" = relevancia ≥ 1.
"""

from __future__ import annotations

import math

import numpy as np

RETRIEVAL_KS = (1, 3, 5, 10)
CONTEXT_KS = (1, 3, 5, 8, 10)
PRIMARY_METRIC = "ParentnDCG@10"
CONTROL_METRICS = ("ParentRecall@5", "EvidenceRecall@5", "ParentHit@1")
DEFAULT_BOOTSTRAP_SEED = 12345


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #


def unique_in_order(seq: list[str]) -> list[str]:
    """Lista sin duplicados preservando el orden de primera aparición."""
    seen: set[str] = set()
    out: list[str] = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(rank + 1) for rank, g in enumerate(gains, start=1))


# --------------------------------------------------------------------------- #
# Métricas de retrieval (parent-level)
# --------------------------------------------------------------------------- #


def parent_hit_at_1(ranked_unique: list[str], relevant: set[str]) -> float:
    return 1.0 if ranked_unique and ranked_unique[0] in relevant else 0.0


def recall_at_k(ranked_unique: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked_unique[:k]) & relevant) / len(relevant)


def mrr_at_k(ranked_unique: list[str], relevant: set[str], k: int) -> float:
    for rank, pid in enumerate(ranked_unique[:k], start=1):
        if pid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_unique: list[str], relevance_by_parent: dict[str, int], k: int) -> float:
    gains = [(2 ** relevance_by_parent.get(p, 0)) - 1 for p in ranked_unique[:k]]
    ideal = sorted(((2**r) - 1 for r in relevance_by_parent.values()), reverse=True)[:k]
    idcg = _dcg([float(g) for g in ideal])
    return _dcg([float(g) for g in gains]) / idcg if idcg > 0 else 0.0


def unique_parents_at_k(ranked_parents: list[str], k: int) -> int:
    return len(set(ranked_parents[:k]))


def duplicate_parent_rate_at_k(ranked_parents: list[str], k: int) -> float:
    top = ranked_parents[:k]
    return 0.0 if not top else 1.0 - len(set(top)) / len(top)


def evidence_metrics_at_k(
    hits: list[dict], evidence_set: set[tuple[str, int]], k: int
) -> tuple[float, float]:
    """(EvidenceHit@k, EvidenceRecall@k): cobertura de párrafos-evidencia por los hits top-k.

    Un hit cubre (parent_id, paragraph_order) si comparte parent y su `context_anchor` cubre ese
    párrafo. Sin anchor no se imputa cobertura de evidencia.
    """
    covered: set[tuple[str, int]] = set()
    for h in hits[:k]:
        pid = h.get("parent_id")
        anchor = h.get("context_anchor")
        for ep, eo in evidence_set:
            if ep != pid:
                continue
            if anchor is not None and anchor["paragraph_start"] <= eo <= anchor["paragraph_end"]:
                covered.add((ep, eo))
    hit = 1.0 if covered else 0.0
    recall = len(covered) / len(evidence_set) if evidence_set else 0.0
    return hit, recall


def compute_query_retrieval_metrics(
    hits: list[dict], judgments: list[dict], *, ks: tuple[int, ...] = RETRIEVAL_KS
) -> dict:
    """Métricas de retrieval de UNA query. `hits` ordenados; `judgments` de esa query."""
    relevance_by_parent = {
        j["parent_id"]: j["relevance"] for j in judgments if j.get("relevance", 0) >= 1
    }
    relevant = set(relevance_by_parent)
    evidence_set = {
        (j["parent_id"], o)
        for j in judgments
        if j.get("relevance", 0) >= 1
        for o in (j.get("evidence") or {}).get("paragraph_orders", [])
    }
    ranked_parents = [h["parent_id"] for h in hits]
    ranked_unique = unique_in_order(ranked_parents)

    out: dict[str, float] = {
        "ParentHit@1": parent_hit_at_1(ranked_unique, relevant),
        "ParentMRR@10": mrr_at_k(ranked_unique, relevant, 10),
        "ParentnDCG@10": ndcg_at_k(ranked_unique, relevance_by_parent, 10),
    }
    for k in ks:
        out[f"ParentRecall@{k}"] = recall_at_k(ranked_unique, relevant, k)
        out[f"UniqueParents@{k}"] = float(unique_parents_at_k(ranked_parents, k))
        out[f"DuplicateParentRate@{k}"] = duplicate_parent_rate_at_k(ranked_parents, k)
        hit, rec = evidence_metrics_at_k(hits, evidence_set, k)
        out[f"EvidenceHit@{k}"] = hit
        out[f"EvidenceRecall@{k}"] = rec
    return out


def aggregate_metrics(per_query: list[dict]) -> dict:
    """Media por métrica sobre todas las queries (ignora claves ausentes)."""
    if not per_query:
        return {}
    keys = sorted({k for d in per_query for k in d})
    return {k: float(np.mean([d[k] for d in per_query if k in d])) for k in keys}


# --------------------------------------------------------------------------- #
# Métricas de contexto
# --------------------------------------------------------------------------- #


def context_metrics(
    context_results: list[dict],
    *,
    relevant_parents: set[str],
    evidence_by_parent: dict[str, list[int]],
) -> dict:
    """Métricas de un contexto ensamblado (lista de ContextResult.as_dict())."""
    parents = [c["parent_id"] for c in context_results]
    uniq = set(parents)
    total_chars = sum(c["char_count"] for c in context_results)
    item_count = sum(c["item_count"] for c in context_results)
    base = sum(c["base_char_count"] for c in context_results)

    covered_orders: dict[str, set[int]] = {}
    for c in context_results:
        covered_orders.setdefault(c["parent_id"], set()).update(c["paragraph_orders"])
    total_ev = sum(len(v) for v in evidence_by_parent.values())
    covered_ev = sum(
        len(set(orders) & covered_orders.get(pid, set()))
        for pid, orders in evidence_by_parent.items()
    )

    seen: set[tuple[str, int]] = set()
    dup = tot = 0
    for c in context_results:
        for o in c["paragraph_orders"]:
            key = (c["parent_id"], o)
            tot += 1
            if key in seen:
                dup += 1
            seen.add(key)

    return {
        "ContextEvidenceRecall": covered_ev / total_ev if total_ev else 0.0,
        "ContextPrecisionById": len(uniq & relevant_parents) / len(uniq) if uniq else 0.0,
        "ContextRecallById": (
            len(uniq & relevant_parents) / len(relevant_parents) if relevant_parents else 0.0
        ),
        "ContextCharacters": float(total_chars),
        "ContextItemCount": float(item_count),
        "ExpansionRatio": total_chars / base if base > 0 else 0.0,
        "RedundantContextRate": dup / tot if tot else 0.0,
    }


# --------------------------------------------------------------------------- #
# Bootstrap (IC 95 %, semilla fija registrada)
# --------------------------------------------------------------------------- #


def bootstrap_ci(
    values: list[float],
    *,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    n_resamples: int = 1000,
    ci: float = 0.95,
) -> dict:
    """IC por bootstrap (percentil) de la media de una métrica por query."""
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {
            "mean": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "seed": seed,
            "n_resamples": n_resamples,
        }
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_resamples, a.size))
    boot = a[idx].mean(axis=1)
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return {
        "mean": float(a.mean()),
        "ci_low": float(np.percentile(boot, lo)),
        "ci_high": float(np.percentile(boot, hi)),
        "seed": seed,
        "n_resamples": n_resamples,
    }


def paired_bootstrap(
    a_values: list[float],
    b_values: list[float],
    *,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    n_resamples: int = 1000,
    ci: float = 0.95,
) -> dict:
    """IC por bootstrap **pareado** de la diferencia de medias (a − b), por query."""
    a = np.asarray(a_values, dtype=float)
    b = np.asarray(b_values, dtype=float)
    if a.size != b.size:
        raise ValueError("bootstrap pareado requiere vectores del mismo tamaño")
    if a.size == 0:
        return {
            "mean_diff": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "seed": seed,
            "n_resamples": n_resamples,
        }
    diffs = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_resamples, diffs.size))
    boot = diffs[idx].mean(axis=1)
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(np.percentile(boot, lo)),
        "ci_high": float(np.percentile(boot, hi)),
        "seed": seed,
        "n_resamples": n_resamples,
    }
