"""Tests de las métricas de generación (L3–L6): puras, deterministas, sin red ni LLM."""

import pytest

from src.evaluation.generation_metrics import (
    abstention_outcome,
    abstention_point,
    aggregate_generation_metrics,
    citation_attribution,
    compute_query_generation_metrics,
    correctness_score,
    faithfulness_score,
    forbidden_fact_hits,
    key_fact_recall,
    normalize_text,
)


def test_normalize_text_strips_accents_and_case() -> None:
    assert normalize_text("  Un MES  ") == "un mes"
    assert normalize_text("Artículo  122") == normalize_text("articulo 122")


def test_key_fact_recall_present_and_missing() -> None:
    out = key_fact_recall("El plazo es de un mes si el acto es expreso.", ["un mes", "tres meses"])
    assert out["key_fact_recall"] == 0.5
    assert out["key_facts_present"] == ["un mes"]
    assert out["key_facts_missing"] == ["tres meses"]


def test_key_fact_recall_none_when_no_facts() -> None:
    assert key_fact_recall("texto", [])["key_fact_recall"] is None


def test_key_fact_recall_respects_numeric_boundaries() -> None:
    # "40.000" NO debe casar embebido en "140.000" (falso positivo del match por subcadena).
    embedded = key_fact_recall("La cuantía asciende a 140.000 euros.", ["40.000"])
    assert embedded["key_fact_recall"] == 0.0
    # …pero SÍ casa como número independiente.
    standalone = key_fact_recall("La sanción es de 40.000 euros.", ["40.000"])
    assert standalone["key_fact_recall"] == 1.0


def test_key_fact_recall_respects_word_boundaries() -> None:
    # "mes" NO debe casar dentro de "mesón"; "días" NO dentro de "adjudicación".
    out = key_fact_recall("Se celebró en el mesón tras la adjudicación.", ["mes", "dias"])
    assert out["key_fact_recall"] == 0.0


def test_forbidden_fact_hits_respects_boundaries() -> None:
    # No debe marcar alucinación por un solapamiento de subcadena espurio.
    assert forbidden_fact_hits("la cuantía es de 140.000 euros", ["40.000"]) == []
    assert forbidden_fact_hits("la cuantía es de 40.000 euros", ["40.000"]) == ["40.000"]


def test_forbidden_fact_hits_flags_hallucination() -> None:
    assert forbidden_fact_hits("el plazo es de tres meses", ["tres meses"]) == ["tres meses"]
    assert forbidden_fact_hits("el plazo es de un mes", ["tres meses"]) == []


def test_citation_attribution_precision_recall_f1() -> None:
    out = citation_attribution(["p1", "p2"], ["p1", "p3"])
    assert out["citation_precision"] == 0.5
    assert out["citation_recall"] == 0.5
    assert out["citation_f1"] == pytest.approx(0.5)


def test_citation_attribution_none_when_no_expected() -> None:
    out = citation_attribution(["p1"], [])
    assert out["citation_precision"] is None


def test_abstention_outcome_four_cases() -> None:
    assert abstention_outcome(answered=True, answerable=True) == "answered"
    assert abstention_outcome(answered=False, answerable=True) == "over_abstention"
    assert abstention_outcome(answered=False, answerable=False) == "correct_abstention"
    assert abstention_outcome(answered=True, answerable=False) == "false_answer"


def test_abstention_point() -> None:
    assert abstention_point(answered=True, has_generation_metrics=True) == "answered"
    assert abstention_point(answered=False, has_generation_metrics=False) == "pre_llm"
    assert abstention_point(answered=False, has_generation_metrics=True) == "llm_decided"


def test_faithfulness_score() -> None:
    assert faithfulness_score([True, True, False, True]) == 0.75
    assert faithfulness_score([]) is None


def test_correctness_score_mapping_and_error() -> None:
    assert correctness_score("correct") == 1.0
    assert correctness_score("partial") == 0.5
    assert correctness_score("incorrect") == 0.0
    with pytest.raises(ValueError):
        correctness_score("nope")


def test_compute_query_metrics_answered_computes_content() -> None:
    m = compute_query_generation_metrics(
        answered=True,
        answer_text="El plazo es de un mes.",
        has_generation_metrics=True,
        cited_parents=["p1"],
        answerable=True,
        key_facts=["un mes"],
        forbidden_facts=["tres meses"],
        expected_citation_parents=["p1"],
        faithfulness_claims=[True, True],
        correctness_label="correct",
    )
    assert m["abstention_outcome"] == "answered"
    assert m["key_fact_recall"] == 1.0
    assert m["faithfulness"] == 1.0
    assert m["correctness"] == 1.0
    assert m["citation_recall"] == 1.0
    assert m["hallucinated_forbidden"] is False


def test_compute_query_metrics_abstention_skips_content() -> None:
    m = compute_query_generation_metrics(
        answered=False,
        answer_text="",
        has_generation_metrics=False,
        cited_parents=[],
        answerable=False,
        key_facts=[],
        forbidden_facts=[],
        expected_citation_parents=[],
    )
    assert m["abstention_outcome"] == "correct_abstention"
    assert m["abstention_point"] == "pre_llm"
    assert "faithfulness" not in m
    assert "key_fact_recall" not in m


def test_aggregate_balanced_accuracy_and_means() -> None:
    per_query = [
        compute_query_generation_metrics(
            answered=True,
            answer_text="un mes",
            has_generation_metrics=True,
            cited_parents=["p1"],
            answerable=True,
            key_facts=["un mes"],
            forbidden_facts=[],
            expected_citation_parents=["p1"],
            faithfulness_claims=[True, False],
            correctness_label="correct",
        ),
        compute_query_generation_metrics(
            answered=False,
            answer_text="",
            has_generation_metrics=False,
            cited_parents=[],
            answerable=False,
            key_facts=[],
            forbidden_facts=[],
            expected_citation_parents=[],
        ),
    ]
    agg = aggregate_generation_metrics(per_query)
    assert agg["n_queries"] == 2
    assert agg["faithfulness_mean"] == 0.5
    assert agg["faithfulness_n"] == 1
    ab = agg["abstention"]
    # 1 answerable respondida (acc 1.0) + 1 no respondible abstenida (acc 1.0) → balanced 1.0
    assert ab["balanced_accuracy"] == 1.0
    assert ab["answer_rate_on_answerable"] == 1.0
    assert ab["abstention_rate_on_unanswerable"] == 1.0
    assert ab["abstention_points"]["pre_llm"] == 1
