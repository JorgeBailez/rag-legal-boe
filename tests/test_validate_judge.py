"""Tests de los helpers puros del CLI de validación del juez (κ vs anotación humana)."""

from scripts.validate_judge import (
    compute_agreement,
    correctness_label_from_score,
    is_faithful,
    scaffold_rows,
)


def test_correctness_label_from_score() -> None:
    assert correctness_label_from_score(1.0) == "correct"
    assert correctness_label_from_score(0.5) == "partial"
    assert correctness_label_from_score(0.0) == "incorrect"
    assert correctness_label_from_score(None) is None


def test_is_faithful() -> None:
    assert is_faithful(1.0) is True
    assert is_faithful(0.8) is False
    assert is_faithful(None) is None


def test_compute_agreement_correctness_and_faithfulness() -> None:
    per_query = [
        {"query_id": "q1", "correctness": 1.0, "faithfulness": 1.0},
        {"query_id": "q2", "correctness": 0.0, "faithfulness": 1.0},
        {"query_id": "q3", "correctness": 0.5, "faithfulness": 0.6},
    ]
    human_rows = [
        {"query_id": "q1", "human_correctness": "correct", "human_faithful": True},
        {"query_id": "q2", "human_correctness": "correct", "human_faithful": True},
        {"query_id": "q3", "human_correctness": "partial", "human_faithful": False},
    ]
    res = compute_agreement(human_rows, per_query)

    corr = res["correctness"]
    assert corr["n"] == 3
    # q2: humano 'correct' vs juez 'incorrect' → único desacuerdo.
    assert len(corr["disagreements"]) == 1
    assert corr["disagreements"][0]["query_id"] == "q2"

    faith = res["faithfulness"]
    assert faith["n"] == 3
    assert faith["percent_agreement"] == 1.0  # q1/q2 faithful, q3 unfaithful → todos coinciden


def test_compute_agreement_ignores_unannotated() -> None:
    per_query = [{"query_id": "q1", "correctness": 1.0, "faithfulness": 1.0}]
    human_rows = [{"query_id": "q1", "human_correctness": "", "human_faithful": None}]
    res = compute_agreement(human_rows, per_query)
    assert res["correctness"]["n"] == 0
    assert res["faithfulness"]["n"] == 0


def test_scaffold_only_answered_with_judge_labels() -> None:
    per_query = [
        {
            "query_id": "q1",
            "answered": True,
            "correctness": 1.0,
            "faithfulness": 1.0,
            "answer_text": "resp",
            "evidences_block": "[E1] art. 1: texto de evidencia",
        },
        {"query_id": "q2", "answered": False},
    ]
    rows = scaffold_rows(per_query, {"q1": "¿pregunta?"}, {"q1": "referencia"})
    assert len(rows) == 1
    row = rows[0]
    assert row["query_id"] == "q1"
    assert row["question"] == "¿pregunta?"
    assert row["answer_text"] == "resp"
    assert row["reference_answer"] == "referencia"
    assert row["evidences_block"] == "[E1] art. 1: texto de evidencia"
    assert row["judge_correctness"] == "correct"
    assert row["judge_faithful"] is True
    assert row["human_correctness"] == "" and row["human_faithful"] is None


def test_scaffold_missing_evidence_is_empty_string() -> None:
    # Report previo al guardado de evidencias: la fila sale con evidences_block vacío (no None).
    per_query = [
        {
            "query_id": "q1",
            "answered": True,
            "correctness": 0.5,
            "faithfulness": 1.0,
            "answer_text": "resp",
        },
    ]
    rows = scaffold_rows(per_query, {"q1": "¿pregunta?"}, {"q1": "referencia"})
    assert rows[0]["evidences_block"] == ""
