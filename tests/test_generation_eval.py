"""Test de orquestación del bucle de evaluación de generación (offline, con fakes)."""

from src.contracts.generation_models import RagLlmAnswerV1
from src.core.exceptions import OllamaApiError
from src.evaluation.generation_eval import evaluate_generation
from src.generation.answer_generator import AnswerGenerator
from tests.generation_fakes import (
    FakeJudge,
    FakeLlmClient,
    FakeRetriever,
    make_corpus_for_parents,
    make_hit,
)


def _generator(answer: RagLlmAnswerV1) -> AnswerGenerator:
    hits = [
        make_hit(rank=1, parent_id="BOE-A-0001__a1", block_id="a1"),
        make_hit(rank=2, parent_id="BOE-A-0001__a2", block_id="a2", label="Ley 1/2000, artículo 2"),
    ]
    corpus = make_corpus_for_parents(["BOE-A-0001__a1", "BOE-A-0001__a2"])
    retriever = FakeRetriever(hits, corpus)
    return AnswerGenerator(retriever=retriever, llm_client=FakeLlmClient(answer))


def _question(qid="q1", split="development", scope="single_parent"):
    return {
        "query_id": qid,
        "query": "¿Qué plazo tengo?",
        "split": split,
        "issue_family_id": f"fam_{qid}",
        "query_style": "ciudadana",
        "answer_scope": scope,
    }


_ANSWERED = RagLlmAnswerV1(
    answered=True,
    answer="Una respuesta con el dato clave.",
    citation_ids=["E1"],
    abstention_reason="",
)
_ABSTAINED = RagLlmAnswerV1(
    answered=False, answer="", citation_ids=[], abstention_reason="sin evidencia útil"
)


def test_evaluate_answered_with_judge() -> None:
    generator = _generator(_ANSWERED)
    judge = FakeJudge(faithfulness_claims=[True, True], correctness="correct")
    questions = [_question()]
    answer_keys = [
        {
            "query_id": "q1",
            "answerable": True,
            "reference_answer": "El plazo es de un mes.",
            "key_facts": ["dato clave"],
            "expected_citation_parents": ["BOE-A-0001__a1"],
        }
    ]
    per_query, metrics_rows, agg = evaluate_generation(
        questions=questions, answer_keys=answer_keys, generator=generator, judge=judge
    )
    assert per_query[0]["abstention_outcome"] == "answered"
    assert per_query[0]["faithfulness"] == 1.0
    assert per_query[0]["correctness"] == 1.0
    assert per_query[0]["key_fact_recall"] == 1.0
    assert per_query[0]["citation_recall"] == 1.0
    assert len(judge.faithfulness_calls) == 1 and len(judge.correctness_calls) == 1
    assert agg["n_queries"] == 1
    assert metrics_rows[0]["query_id"] == "q1"


def test_evaluate_without_judge_skips_judge_metrics() -> None:
    generator = _generator(_ANSWERED)
    questions = [_question()]
    answer_keys = [{"query_id": "q1", "answerable": True, "key_facts": ["dato clave"]}]
    per_query, _, agg = evaluate_generation(
        questions=questions, answer_keys=answer_keys, generator=generator, judge=None
    )
    assert "faithfulness" not in per_query[0]
    assert per_query[0]["key_fact_recall"] == 1.0
    assert agg["faithfulness_mean"] is None


def test_evaluate_correct_abstention_on_out_of_corpus() -> None:
    generator = _generator(_ABSTAINED)
    judge = FakeJudge()
    questions = [_question(qid="ooc1", split="out_of_corpus", scope="none")]
    answer_keys = [{"query_id": "ooc1", "answerable": False}]
    per_query, _, agg = evaluate_generation(
        questions=questions, answer_keys=answer_keys, generator=generator, judge=judge
    )
    assert per_query[0]["abstention_outcome"] == "correct_abstention"
    assert per_query[0]["abstention_point"] == "llm_decided"
    assert len(judge.faithfulness_calls) == 0  # no se juzga una abstención
    assert agg["abstention"]["abstention_rate_on_unanswerable"] == 1.0


class _RaisingJudge:
    """Juez que falla (p. ej. JSON truncado): no debe abortar la corrida."""

    model_label = "raising"

    def judge_faithfulness(self, *, answer, evidences_block):  # noqa: ANN001, ANN204
        raise OllamaApiError("la salida del juez no es JSON válido")

    def judge_correctness(self, *, question, answer, reference):  # noqa: ANN001, ANN204
        raise OllamaApiError("boom")


def test_evaluate_judge_error_does_not_abort() -> None:
    generator = _generator(_ANSWERED)
    questions = [_question()]
    answer_keys = [{"query_id": "q1", "answerable": True, "key_facts": ["dato clave"]}]
    per_query, _, agg = evaluate_generation(
        questions=questions, answer_keys=answer_keys, generator=generator, judge=_RaisingJudge()
    )
    assert per_query[0]["judge_error"]  # se registró el fallo del juez
    assert per_query[0].get("faithfulness") is None  # no se calculó L3
    assert per_query[0]["key_fact_recall"] == 1.0  # las métricas sin juez SÍ se calcularon
    assert agg["n_queries"] == 1  # la corrida NO abortó


def test_evaluate_limit_truncates() -> None:
    generator = _generator(_ANSWERED)
    questions = [_question(qid=f"q{i}") for i in range(3)]
    answer_keys = [{"query_id": f"q{i}", "answerable": True} for i in range(3)]
    per_query, _, _ = evaluate_generation(
        questions=questions, answer_keys=answer_keys, generator=generator, judge=None, limit=1
    )
    assert len(per_query) == 1
