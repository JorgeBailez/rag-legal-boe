"""Tests de las métricas de retrieval/contexto y del bootstrap reproducible."""

import math

from src.evaluation.metrics import (
    aggregate_metrics,
    bootstrap_ci,
    compute_query_retrieval_metrics,
    context_metrics,
    duplicate_parent_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    paired_bootstrap,
    parent_hit_at_1,
    recall_at_k,
    unique_in_order,
)


def _hit(parent_id, anchor=None):
    return {"parent_id": parent_id, "context_anchor": anchor}


# --- métricas básicas -------------------------------------------------------


def test_unique_in_order_and_duplicate_rate() -> None:
    assert unique_in_order(["a", "b", "a", "c"]) == ["a", "b", "c"]
    assert duplicate_parent_rate_at_k(["a", "a", "b"], 3) == 1 - 2 / 3


def test_parent_hit_recall_mrr() -> None:
    ranked = ["p1", "p2", "p3"]
    relevant = {"p2"}
    assert parent_hit_at_1(ranked, relevant) == 0.0
    assert parent_hit_at_1(["p2", "p1"], relevant) == 1.0
    assert recall_at_k(ranked, {"p2", "p9"}, 5) == 0.5
    assert mrr_at_k(ranked, relevant, 10) == 0.5


def test_ndcg_perfect_and_imperfect() -> None:
    rel = {"p1": 2, "p2": 1}
    # ranking ideal → nDCG 1.0
    assert math.isclose(ndcg_at_k(["p1", "p2"], rel, 10), 1.0, rel_tol=1e-9)
    # ranking invertido → < 1
    assert ndcg_at_k(["p2", "p1"], rel, 10) < 1.0


def test_query_metrics_with_evidence() -> None:
    hits = [_hit("p1", {"paragraph_start": 1, "paragraph_end": 3}), _hit("p2")]
    judgments = [
        {"parent_id": "p1", "relevance": 2, "evidence": {"paragraph_orders": [2]}},
        {"parent_id": "p9", "relevance": 1, "evidence": {"paragraph_orders": [1]}},
    ]
    m = compute_query_retrieval_metrics(hits, judgments)
    assert m["ParentHit@1"] == 1.0  # p1 relevante en top-1
    assert m["EvidenceHit@5"] == 1.0  # anchor de p1 cubre el párrafo 2
    assert 0.0 < m["EvidenceRecall@5"] <= 1.0
    assert m["ParentRecall@5"] == 0.5  # de {p1, p9} solo p1 recuperado


def test_evidence_recall_does_not_treat_missing_anchor_as_full_parent() -> None:
    hits = [_hit("p1", None)]
    judgments = [{"parent_id": "p1", "relevance": 2, "evidence": {"paragraph_orders": [2]}}]
    m = compute_query_retrieval_metrics(hits, judgments)
    assert m["EvidenceHit@5"] == 0.0
    assert m["EvidenceRecall@5"] == 0.0


def test_evidence_recall_uses_refined_segment_anchor_only() -> None:
    hits = [_hit("p1", {"paragraph_start": 1, "paragraph_end": 1})]
    judgments = [{"parent_id": "p1", "relevance": 2, "evidence": {"paragraph_orders": [2]}}]
    m = compute_query_retrieval_metrics(hits, judgments)
    assert m["EvidenceHit@5"] == 0.0
    assert m["EvidenceRecall@5"] == 0.0


def test_aggregate_metrics_mean() -> None:
    agg = aggregate_metrics([{"ParentHit@1": 1.0}, {"ParentHit@1": 0.0}])
    assert agg["ParentHit@1"] == 0.5


# --- contexto ---------------------------------------------------------------


def test_context_metrics_basic() -> None:
    ctx = [
        {
            "parent_id": "p1",
            "paragraph_orders": [1, 2],
            "char_count": 100,
            "item_count": 2,
            "base_char_count": 50,
        },
        {
            "parent_id": "p2",
            "paragraph_orders": [1],
            "char_count": 40,
            "item_count": 1,
            "base_char_count": 40,
        },
    ]
    m = context_metrics(ctx, relevant_parents={"p1"}, evidence_by_parent={"p1": [2]})
    assert m["ContextEvidenceRecall"] == 1.0  # párrafo 2 de p1 presente
    assert m["ContextPrecisionById"] == 0.5  # 1 de 2 parents relevante
    assert m["ContextRecallById"] == 1.0
    assert m["ContextCharacters"] == 140.0
    assert m["ExpansionRatio"] == 140 / 90


# --- bootstrap reproducible -------------------------------------------------


def test_bootstrap_ci_is_reproducible_and_bracketed() -> None:
    vals = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    a = bootstrap_ci(vals, seed=12345, n_resamples=500)
    b = bootstrap_ci(vals, seed=12345, n_resamples=500)
    assert a == b  # misma semilla → mismo resultado
    assert a["ci_low"] <= a["mean"] <= a["ci_high"]
    assert a["seed"] == 12345


def test_paired_bootstrap_detects_positive_difference() -> None:
    a = [1.0] * 10
    b = [0.0] * 10
    res = paired_bootstrap(a, b, seed=7, n_resamples=300)
    assert res["mean_diff"] == 1.0
    assert res["ci_low"] == 1.0 and res["ci_high"] == 1.0
