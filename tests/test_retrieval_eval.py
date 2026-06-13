"""Tests del núcleo de comparación de estrategias de retrieval (offline, con fakes)."""

from src.evaluation.retrieval_eval import evaluate_retrieval_strategies
from tests.generation_fakes import FakeRetriever, make_corpus_for_parents, make_hit

_P1 = "BOE-A-0001__a1"  # parent relevante
_P2 = "BOE-A-0001__a2"  # parent no relevante
_CORPUS = make_corpus_for_parents([_P1, _P2])


def _retriever(first: str, second: str) -> FakeRetriever:
    hits = [
        make_hit(rank=1, parent_id=first, block_id=first.split("__")[-1]),
        make_hit(rank=2, parent_id=second, block_id=second.split("__")[-1]),
    ]
    return FakeRetriever(hits, _CORPUS)


def _questions() -> list[dict]:
    return [
        {"query_id": "q1", "query": "¿plazo?", "split": "development"},
        {"query_id": "q2", "query": "¿silencio?", "split": "development"},
    ]


def _judgments() -> dict[str, list[dict]]:
    rel = lambda qid: [{"query_id": qid, "parent_id": _P1, "relevance": 2}]  # noqa: E731
    return {"q1": rel("q1"), "q2": rel("q2")}


def test_compara_estrategias_y_calcula_pareado() -> None:
    result = evaluate_retrieval_strategies(
        strategies={"dense": _retriever(_P1, _P2), "bm25": _retriever(_P2, _P1)},
        split_questions=_questions(),
        judgments_by_query=_judgments(),
    )
    rows = {r["strategy"]: r for r in result["metrics_rows"]}
    assert rows["dense"]["ParentHit@1"] == 1.0  # el denso pone el relevante primero
    assert rows["bm25"]["ParentHit@1"] == 0.0  # bm25 lo pone segundo
    assert rows["dense"]["ParentnDCG@10"] > rows["bm25"]["ParentnDCG@10"]

    summary = result["summary"]
    assert summary["baseline"] == "dense"
    assert [p["strategy"] for p in summary["paired_vs_baseline"]] == ["bm25"]
    assert all("mean" in s["primary_ci"] for s in summary["strategies"])
    assert len(result["query_results"]) == 4  # 2 estrategias × 2 queries


def test_baseline_cae_a_la_primera_si_no_hay_dense() -> None:
    result = evaluate_retrieval_strategies(
        strategies={"bm25": _retriever(_P1, _P2), "hybrid": _retriever(_P2, _P1)},
        split_questions=_questions(),
        judgments_by_query=_judgments(),
        baseline="dense",
    )
    assert result["summary"]["baseline"] == "bm25"
    assert [p["strategy"] for p in result["summary"]["paired_vs_baseline"]] == ["hybrid"]
