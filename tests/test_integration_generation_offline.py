"""Integración offline del flujo de generación de Fase 3 (bundle real + fakes, sin red ni Ollama).

Ejercita extremo a extremo: corpus sintético → bundle temporal real → ExactDenseIndex →
DenseRetriever (con FakeEncoder) → evidencias acotadas → AnswerGenerator con un LlmClient falso →
RagAnswerV1 trazable.
"""

from src.contracts.generation_models import OllamaMetricsV1, RagLlmAnswerV1
from src.generation.answer_generator import DISCLAIMER, AnswerGenerator, GenerationConfig
from tests.dense_fakes import FakeEncoder
from tests.generation_fakes import FakeLlmClient, build_bundle_retriever
from tests.test_bundle import TEST_CONTRACT


def test_end_to_end_offline_grounded_answer(tmp_path) -> None:
    encoder = FakeEncoder(dimension=8, contract=TEST_CONTRACT)
    retriever = build_bundle_retriever(tmp_path, TEST_CONTRACT, encoder)

    # El LLM (falso) cita E1, que el orquestador debe enriquecer con datos autoritativos del corpus.
    llm = FakeLlmClient(
        RagLlmAnswerV1(
            answered=True,
            answer="Según la evidencia, el plazo general es de tres meses.",
            citation_ids=["E1"],
            abstention_reason="",
        ),
        metrics=OllamaMetricsV1(total_duration_ns=5_000_000_000, eval_count=20, eval_duration_ns=1),
    )
    config = GenerationConfig(query_profile_id="BASELINE", top_k=5, max_evidences=3)
    gen = AnswerGenerator(retriever=retriever, llm_client=llm, config=config)

    ans = gen.answer("plazo administrativo", query_profile_id="BASELINE")

    assert ans.answered
    assert ans.answer
    assert ans.citations, "esperaba citas enriquecidas"
    cite = ans.citations[0]
    assert cite.evidence_id == "E1"
    assert cite.url and cite.url.startswith("https://")  # URL autoritativa del corpus
    assert cite.parent_id  # relación al parent jurídico
    assert ans.disclaimer == DISCLAIMER
    assert ans.retrieval_trace.bundle_id.startswith("fake-test__j1__")
    assert ans.retrieval_trace.returned_hits >= 1
    assert ans.retrieval_trace.selected_evidences >= 1
    assert ans.generation_metrics is not None and ans.generation_metrics.eval_count == 20
    # La traza marca qué hits se seleccionaron como evidencia.
    assert any(h.selected and h.evidence_id for h in ans.retrieval_trace.hits)


def test_end_to_end_offline_abstention_without_evidence(tmp_path) -> None:
    encoder = FakeEncoder(dimension=8, contract=TEST_CONTRACT)
    retriever = build_bundle_retriever(tmp_path, TEST_CONTRACT, encoder)
    llm = FakeLlmClient(
        RagLlmAnswerV1(
            answered=False,
            answer="",
            citation_ids=[],
            abstention_reason="La evidencia no contiene la respuesta.",
        )
    )
    config = GenerationConfig(query_profile_id="BASELINE")
    gen = AnswerGenerator(retriever=retriever, llm_client=llm, config=config)

    ans = gen.answer("pregunta sin respuesta en el corpus", query_profile_id="BASELINE")
    assert not ans.answered
    assert ans.abstention_reason
    assert ans.disclaimer == DISCLAIMER
