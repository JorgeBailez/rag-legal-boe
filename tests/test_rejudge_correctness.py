"""Test del re-juzgado de solo corrección reutilizando un report previo (offline, con fakes)."""

import pytest

from src.core.exceptions import RagLegalBoeError
from src.evaluation.generation_eval import rejudge_correctness
from tests.generation_fakes import FakeJudge


def _question(qid="q1", failure_mode="numeric_threshold"):
    return {
        "query_id": qid,
        "query": "¿Qué plazo tengo?",
        "split": "development",
        "issue_family_id": f"fam_{qid}",
        "query_style": "ciudadana",
        "answer_scope": "single_parent",
        "failure_mode": failure_mode,
        "difficulty": "media",
    }


def _prior_row(qid="q1", *, answered=True, faithfulness=1.0, correctness=0.0):
    """Fila de un per_query.jsonl previo (respuesta + fidelidad ya guardadas)."""
    return {
        "query_id": qid,
        "split": "development",
        "query_style": "ciudadana",
        "failure_mode": "numeric_threshold",
        "difficulty": "media",
        "answered": answered,
        "answerable": True,
        "abstention_outcome": "answered" if answered else "over_abstention",
        "abstention_point": "answered" if answered else "llm_decided",
        "faithfulness": faithfulness,
        "correctness": correctness,
        "cited_parents": ["BOE-A-0001__a1"] if answered else [],
        "delivered_parents": ["BOE-A-0001__a1"] if answered else [],
        "retrieved_parents": ["BOE-A-0001__a1"],
        "omitted_evidences": [],
        "expected_citation_parents": ["BOE-A-0001__a1"],
        "answer_text": "El plazo es de un mes con el dato clave." if answered else "",
        "abstention_reason": "" if answered else "sin evidencia útil",
        "eval_count": 10 if answered else 8,
        "latency_s": 3.0,
    }


def _answer_key(qid="q1", *, answerable=True, reference="El plazo es de un mes."):
    return {
        "query_id": qid,
        "answerable": answerable,
        "reference_answer": reference,
        "key_facts": ["dato clave"],
        "forbidden_facts": [],
        "expected_citation_parents": ["BOE-A-0001__a1"],
    }


def test_rejudge_reuses_faithfulness_and_rejudges_correctness() -> None:
    judge = FakeJudge(correctness="correct")  # el veredicto previo era 0.0 (incorrect)
    per_query, metrics_rows, agg = rejudge_correctness(
        prior_per_query=[_prior_row(correctness=0.0)],
        answer_keys=[_answer_key()],
        questions=[_question()],
        judge=judge,
    )
    row = per_query[0]
    assert row["correctness"] == 1.0  # re-juzgada (cambia respecto al 0.0 previo)
    assert row["faithfulness"] == 1.0  # REUSADA del report previo
    assert row["faithfulness_source"] == "reused"
    assert row["key_fact_recall"] == 1.0  # recalculada sobre la respuesta guardada
    assert row["citation_recall"] == 1.0
    assert row["rejudged"] is True
    # solo se llamó al juez de CORRECCIÓN, nunca al de fidelidad
    assert len(judge.correctness_calls) == 1
    assert len(judge.faithfulness_calls) == 0
    assert agg["correctness_mean"] == 1.0 and agg["faithfulness_mean"] == 1.0
    assert metrics_rows[0]["query_id"] == "q1"


def test_rejudge_skips_abstained_and_marks_no_faithfulness() -> None:
    judge = FakeJudge(correctness="correct")
    per_query, _, agg = rejudge_correctness(
        prior_per_query=[_prior_row(answered=False, faithfulness=None)],
        answer_keys=[_answer_key()],
        questions=[_question()],
        judge=judge,
    )
    row = per_query[0]
    assert row["abstention_outcome"] == "over_abstention"
    assert "correctness" not in row  # no se juzga una abstención
    assert "faithfulness" not in row  # no había fidelidad previa que reutilizar
    assert row["faithfulness_source"] == "none"
    assert len(judge.correctness_calls) == 0
    assert agg["abstention"]["over_abstention_rate"] == 1.0


def test_rejudge_raises_on_report_dataset_mismatch() -> None:
    with pytest.raises(RagLegalBoeError):
        rejudge_correctness(
            prior_per_query=[_prior_row(qid="q9")],
            answer_keys=[_answer_key(qid="q1")],
            questions=[_question(qid="q1")],
            judge=FakeJudge(),
        )


def test_rejudge_limit_truncates() -> None:
    prior = [_prior_row(qid="q1"), _prior_row(qid="q2")]
    per_query, _, _ = rejudge_correctness(
        prior_per_query=prior,
        answer_keys=[_answer_key(qid="q1"), _answer_key(qid="q2")],
        questions=[_question(qid="q1"), _question(qid="q2")],
        judge=FakeJudge(correctness="partial"),
        limit=1,
    )
    assert len(per_query) == 1 and per_query[0]["query_id"] == "q1"
    assert per_query[0]["correctness"] == 0.5
