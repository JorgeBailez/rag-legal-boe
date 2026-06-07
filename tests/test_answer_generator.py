"""Tests del orquestador de respuesta (enriquecimiento, abstención, fail-closed, contrato)."""

import argparse

import pytest

from src.contracts.generation_models import OllamaMetricsV1, RagLlmAnswerV1
from src.core.exceptions import ConfigurationError, GenerationContractError
from src.generation.answer_generator import DISCLAIMER, AnswerGenerator, GenerationConfig
from tests.generation_fakes import (
    FakeLlmClient,
    FakeRetriever,
    make_corpus_for_parents,
    make_hit,
)


def _two_hit_setup(answer: RagLlmAnswerV1) -> tuple[AnswerGenerator, FakeLlmClient, FakeRetriever]:
    corpus = make_corpus_for_parents(["d__a1", "d__a2"])
    hits = [
        make_hit(rank=1, parent_id="d__a1", label="Ley X, art. 1", url="https://boe/d#a1"),
        make_hit(rank=2, parent_id="d__a2", label="Ley X, art. 2", url="https://boe/d#a2"),
    ]
    retriever = FakeRetriever(hits, corpus)
    llm = FakeLlmClient(answer)
    return AnswerGenerator(retriever=retriever, llm_client=llm), llm, retriever


def test_grounded_answer_enriched_with_authoritative_url() -> None:
    gen, llm, _ = _two_hit_setup(
        RagLlmAnswerV1(
            answered=True,
            answer="El plazo es de un mes.",
            citation_ids=["E1"],
            abstention_reason="",
        )
    )
    ans = gen.answer("¿plazo?")
    assert ans.answered
    assert ans.answer == "El plazo es de un mes."
    assert len(ans.citations) == 1
    cite = ans.citations[0]
    # Etiqueta y URL provienen del corpus (autoritativas), no del texto del LLM.
    assert cite.evidence_id == "E1"
    assert cite.label == "Ley X, art. 1"
    assert cite.url == "https://boe/d#a1"
    assert ans.disclaimer == DISCLAIMER
    assert ans.generation_metrics is not None and ans.generation_metrics.eval_count == 10
    assert ans.retrieval_trace.selected_evidences == 2
    assert llm.calls  # se llamó al LLM


def test_llm_abstention_is_propagated() -> None:
    gen, llm, _ = _two_hit_setup(
        RagLlmAnswerV1(
            answered=False,
            answer="",
            citation_ids=[],
            abstention_reason="El contexto no contiene la respuesta.",
        )
    )
    ans = gen.answer("¿plazo?")
    assert not ans.answered
    assert ans.abstention_reason == "El contexto no contiene la respuesta."
    assert ans.citations == []
    assert ans.disclaimer == DISCLAIMER
    assert ans.generation_metrics is not None  # hubo llamada al LLM
    assert llm.calls


def test_no_hits_abstains_without_calling_llm() -> None:
    corpus = make_corpus_for_parents(["d__a1"])
    retriever = FakeRetriever([], corpus)
    llm = FakeLlmClient(
        RagLlmAnswerV1(answered=True, answer="x", citation_ids=["E1"], abstention_reason="")
    )
    gen = AnswerGenerator(retriever=retriever, llm_client=llm)
    ans = gen.answer("¿plazo?")
    assert not ans.answered
    assert ans.generation_metrics is None
    assert ans.disclaimer == DISCLAIMER
    assert ans.retrieval_trace.returned_hits == 0
    assert llm.calls == []  # fail closed: no se llamó al LLM


def test_no_usable_evidences_abstains_without_calling_llm() -> None:
    # Hay hits, pero sus parents no están en el corpus → no se puede construir evidencia.
    corpus = make_corpus_for_parents(["d__a1"])
    hits = [make_hit(rank=1, parent_id="d__missing")]
    retriever = FakeRetriever(hits, corpus)
    llm = FakeLlmClient(
        RagLlmAnswerV1(answered=True, answer="x", citation_ids=["E1"], abstention_reason="")
    )
    gen = AnswerGenerator(retriever=retriever, llm_client=llm)
    ans = gen.answer("¿plazo?")
    assert not ans.answered
    assert ans.generation_metrics is None
    assert llm.calls == []


def test_unknown_citation_id_raises_contract_error() -> None:
    gen, _, _ = _two_hit_setup(
        RagLlmAnswerV1(
            answered=True, answer="Respuesta.", citation_ids=["E9"], abstention_reason=""
        )
    )
    with pytest.raises(GenerationContractError, match="no entregados"):
        gen.answer("¿plazo?")


def test_trace_propagates_evidence_diagnostics() -> None:
    # Un parent duplicado (rank 1 y 2) + un parent ausente (rank 3): la traza debe reflejarlo.
    corpus = make_corpus_for_parents(["d__a1"])
    hits = [
        make_hit(rank=1, parent_id="d__a1"),
        make_hit(rank=2, parent_id="d__a1"),  # duplicado
        make_hit(rank=3, parent_id="d__missing"),  # ausente
    ]
    retriever = FakeRetriever(hits, corpus)
    llm = FakeLlmClient(
        RagLlmAnswerV1(answered=True, answer="ok", citation_ids=["E1"], abstention_reason="")
    )
    gen = AnswerGenerator(retriever=retriever, llm_client=llm)
    ans = gen.answer("¿plazo?")

    t = ans.retrieval_trace
    assert t.returned_hits == 3
    assert t.selected_evidences == 1
    assert t.duplicate_parents_removed == 1
    assert t.total_context_chars > 0
    assert any(
        o.reason == "parent_not_found" and o.parent_id == "d__missing" for o in t.omitted_evidences
    )


def test_only_winning_hit_is_marked_selected() -> None:
    # Dos hits del mismo parent: solo el ganador (rank 1) debe quedar marcado en la traza.
    corpus = make_corpus_for_parents(["d__a1"])
    hits = [
        make_hit(rank=1, parent_id="d__a1"),
        make_hit(rank=2, parent_id="d__a1"),  # duplicado del mismo parent
    ]
    retriever = FakeRetriever(hits, corpus)
    llm = FakeLlmClient(
        RagLlmAnswerV1(answered=True, answer="ok", citation_ids=["E1"], abstention_reason="")
    )
    ans = AnswerGenerator(retriever=retriever, llm_client=llm).answer("¿plazo?")

    t = ans.retrieval_trace
    assert t.duplicate_parents_removed == 1
    selected = [h for h in t.hits if h.selected]
    assert len(selected) == 1
    assert selected[0].rank == 1 and selected[0].evidence_id == "E1"
    non_selected = [h for h in t.hits if not h.selected]
    assert non_selected and all(h.evidence_id is None for h in non_selected)


def test_generation_config_rejects_non_positive_top_k() -> None:
    with pytest.raises(ConfigurationError, match="top_k"):
        GenerationConfig(top_k=0)


def test_generation_config_rejects_unknown_strategy() -> None:
    with pytest.raises(ConfigurationError, match="context_strategy"):
        GenerationConfig(context_strategy="NOPE")


def test_cli_positive_int_helper_accepts_and_rejects() -> None:
    from scripts.answer_question import _non_blank, _positive_int

    assert _positive_int("5") == 5
    for bad in ("0", "-3", "abc"):
        with pytest.raises(argparse.ArgumentTypeError):
            _positive_int(bad)

    assert _non_blank("¿Qué plazo tengo?") == "¿Qué plazo tengo?"
    for bad in ("", "   "):
        with pytest.raises(argparse.ArgumentTypeError):
            _non_blank(bad)


def test_metrics_propagated_with_custom_values() -> None:
    corpus = make_corpus_for_parents(["d__a1"])
    hits = [make_hit(rank=1, parent_id="d__a1")]
    retriever = FakeRetriever(hits, corpus)
    metrics = OllamaMetricsV1(total_duration_ns=7_000_000_000, eval_count=99, eval_duration_ns=1)
    llm = FakeLlmClient(
        RagLlmAnswerV1(answered=True, answer="ok", citation_ids=["E1"], abstention_reason=""),
        metrics=metrics,
    )
    gen = AnswerGenerator(retriever=retriever, llm_client=llm)
    ans = gen.answer("¿plazo?")
    assert ans.generation_metrics is not None and ans.generation_metrics.eval_count == 99
