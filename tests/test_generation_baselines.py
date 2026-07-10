"""Tests de los baselines de generación (closed-book y oracle), offline y con fakes."""

from src.contracts.generation_models import RagLlmAnswerV1
from src.evaluation.generation_eval import evaluate_generation
from src.generation.answer_generator import AnswerGenerator
from src.retrieval.evidence_builder import EvidenceSelection, build_oracle_evidences
from tests.generation_fakes import FakeLlmClient, FakeRetriever, make_corpus_for_parents

_PARENTS = ["BOE-A-0001__a1", "BOE-A-0001__a2"]
_ANSWERED_E1 = RagLlmAnswerV1(
    answered=True,
    answer="Una respuesta con el dato clave.",
    citation_ids=["E1"],
    abstention_reason="",
)


def _corpus() -> dict:
    return make_corpus_for_parents(_PARENTS)


def _question(qid: str = "q1", split: str = "test") -> dict:
    return {
        "query_id": qid,
        "query": "¿Qué plazo tengo?",
        "split": split,
        "issue_family_id": f"fam_{qid}",
        "query_style": "ciudadana",
        "answer_scope": "single_parent",
    }


def _gold_a1() -> list[dict]:
    return [
        {
            "query_id": "q1",
            "parent_id": "BOE-A-0001__a1",
            "relevance": 2,
            "evidence": {"paragraph_orders": [1]},
        }
    ]


# --------------------------------------------------------------------------- #
# build_oracle_evidences (evidencia gold inyectada)
# --------------------------------------------------------------------------- #


def test_build_oracle_evidences_injects_gold_parents() -> None:
    corpus = _corpus()
    gold = [
        {"parent_id": "BOE-A-0001__a1", "paragraph_orders": [1], "relevance": 2},
        {"parent_id": "BOE-A-0001__a2", "paragraph_orders": [2], "relevance": 1},
    ]
    selection = build_oracle_evidences(gold, parents_by_id=corpus["parents_by_id"])
    assert [ev.evidence_id for ev in selection.evidences] == ["E1", "E2"]
    assert [ev.parent_id for ev in selection.evidences] == ["BOE-A-0001__a1", "BOE-A-0001__a2"]
    assert all(ev.text.strip() for ev in selection.evidences)  # contexto no vacío


def test_build_oracle_evidences_skips_unknown_parent() -> None:
    corpus = _corpus()
    gold = [{"parent_id": "NO-EXISTE", "paragraph_orders": [1], "relevance": 2}]
    selection = build_oracle_evidences(gold, parents_by_id=corpus["parents_by_id"])
    assert selection.evidences == []


# --------------------------------------------------------------------------- #
# AnswerGenerator.answer_with_evidences (ruta oracle)
# --------------------------------------------------------------------------- #


def test_answer_with_evidences_uses_injected_and_skips_retrieval() -> None:
    corpus = _corpus()
    selection = build_oracle_evidences(
        [{"parent_id": "BOE-A-0001__a1", "paragraph_orders": [1], "relevance": 2}],
        parents_by_id=corpus["parents_by_id"],
    )
    retriever = FakeRetriever([], corpus)
    llm = FakeLlmClient(_ANSWERED_E1)
    gen = AnswerGenerator(retriever=retriever, llm_client=llm)

    ans = gen.answer_with_evidences("¿Qué plazo tengo?", selection)

    assert ans.answered is True
    assert [c.parent_id for c in ans.citations] == ["BOE-A-0001__a1"]
    assert retriever.retrieve_calls == 0  # el oracle NO recupera
    assert len(llm.calls) == 1
    assert ans.retrieval_trace.selected_evidences == 1


def test_answer_with_evidences_abstains_without_evidence() -> None:
    corpus = _corpus()
    llm = FakeLlmClient(_ANSWERED_E1)
    gen = AnswerGenerator(retriever=FakeRetriever([], corpus), llm_client=llm)

    ans = gen.answer_with_evidences("¿Qué plazo tengo?", EvidenceSelection())

    assert ans.answered is False
    assert ans.abstention_reason
    assert len(llm.calls) == 0  # sin evidencia no se llama al LLM


# --------------------------------------------------------------------------- #
# closed-book
# --------------------------------------------------------------------------- #


def test_generate_closed_book_answered() -> None:
    corpus = _corpus()
    llm = FakeLlmClient(
        json_payload={"answered": True, "answer": "El plazo es de un mes.", "abstention_reason": ""}
    )
    gen = AnswerGenerator(retriever=FakeRetriever([], corpus), llm_client=llm)

    res = gen.generate_closed_book("¿Qué plazo tengo?")

    assert res.answered is True
    assert "mes" in res.answer
    assert len(llm.json_calls) == 1 and len(llm.calls) == 0  # usa chat_json, no chat


def test_generate_closed_book_abstains() -> None:
    corpus = _corpus()
    llm = FakeLlmClient(
        json_payload={"answered": False, "answer": "", "abstention_reason": "no lo sé"}
    )
    gen = AnswerGenerator(retriever=FakeRetriever([], corpus), llm_client=llm)

    res = gen.generate_closed_book("¿Qué plazo tengo?")

    assert res.answered is False
    assert res.abstention_reason == "no lo sé"


# --------------------------------------------------------------------------- #
# evaluate_generation con modos
# --------------------------------------------------------------------------- #


def test_evaluate_generation_oracle_end_to_end() -> None:
    corpus = _corpus()
    gen = AnswerGenerator(
        retriever=FakeRetriever([], corpus), llm_client=FakeLlmClient(_ANSWERED_E1)
    )
    questions = [_question()]
    answer_keys = [
        {
            "query_id": "q1",
            "answerable": True,
            "key_facts": ["dato clave"],
            "expected_citation_parents": ["BOE-A-0001__a1"],
        }
    ]
    per_query, _, agg = evaluate_generation(
        questions=questions,
        answer_keys=answer_keys,
        generator=gen,
        judge=None,
        mode="oracle",
        judgments=_gold_a1(),
    )
    assert per_query[0]["mode"] == "oracle"
    assert per_query[0]["abstention_outcome"] == "answered"
    assert per_query[0]["key_fact_recall"] == 1.0
    assert per_query[0]["citation_recall"] == 1.0  # citó el parent gold inyectado
    assert agg["n_queries"] == 1


def test_evaluate_generation_closed_book_end_to_end() -> None:
    corpus = _corpus()
    llm = FakeLlmClient(
        json_payload={
            "answered": True,
            "answer": "Respuesta con el dato clave.",
            "abstention_reason": "",
        }
    )
    gen = AnswerGenerator(retriever=FakeRetriever([], corpus), llm_client=llm)
    questions = [_question()]
    answer_keys = [{"query_id": "q1", "answerable": True, "key_facts": ["dato clave"]}]

    per_query, _, agg = evaluate_generation(
        questions=questions,
        answer_keys=answer_keys,
        generator=gen,
        judge=None,
        mode="closed_book",
    )
    assert per_query[0]["mode"] == "closed_book"
    assert per_query[0]["abstention_outcome"] == "answered"
    assert per_query[0]["key_fact_recall"] == 1.0
    assert per_query[0]["cited_parents"] == []  # closed-book no cita
    assert agg["n_queries"] == 1
