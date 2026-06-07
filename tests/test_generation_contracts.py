"""Tests de los contratos de generación (Fase 3) y de la ausencia de drift en sus JSON Schema."""

import pytest
from pydantic import ValidationError

from src.contracts.export_schemas import SCHEMAS_DIR, check, schema_json
from src.contracts.generation_models import (
    GENERATION_ROOT_MODELS,
    AnswerCitationV1,
    OllamaMetricsV1,
    RagAnswerV1,
    RagLlmAnswerV1,
    RetrievalTraceV1,
)


def _trace() -> RetrievalTraceV1:
    return RetrievalTraceV1(
        bundle_id="e5-large-instruct__j1__bc11142bdcc5",
        model_alias="e5-large-instruct",
        query_profile_id="I2_CITIZEN_LEGISLATION",
        top_k=5,
        returned_hits=0,
        selected_evidences=0,
    )


# --- rag_llm_answer_v1: invariantes ----------------------------------------


def test_llm_answer_answered_valid() -> None:
    ans = RagLlmAnswerV1(
        answered=True, answer="El plazo es de un mes.", citation_ids=["E1"], abstention_reason=""
    )
    assert ans.answered and ans.citation_ids == ["E1"]


def test_llm_answer_abstention_valid() -> None:
    ans = RagLlmAnswerV1(
        answered=False, answer="", citation_ids=[], abstention_reason="No hay evidencia suficiente."
    )
    assert not ans.answered and ans.abstention_reason


def test_llm_answer_answered_without_citations_rejected() -> None:
    with pytest.raises(ValidationError):
        RagLlmAnswerV1(answered=True, answer="Algo", citation_ids=[], abstention_reason="")


def test_llm_answer_answered_without_answer_rejected() -> None:
    with pytest.raises(ValidationError):
        RagLlmAnswerV1(answered=True, answer="   ", citation_ids=["E1"], abstention_reason="")


def test_llm_answer_abstention_with_answer_rejected() -> None:
    with pytest.raises(ValidationError):
        RagLlmAnswerV1(
            answered=False, answer="Respuesta", citation_ids=[], abstention_reason="motivo"
        )


def test_llm_answer_abstention_without_reason_rejected() -> None:
    with pytest.raises(ValidationError):
        RagLlmAnswerV1(answered=False, answer="", citation_ids=[], abstention_reason="")


def test_llm_answer_extra_field_forbidden() -> None:
    with pytest.raises(ValidationError):
        RagLlmAnswerV1.model_validate(
            {
                "answered": False,
                "answer": "",
                "citation_ids": [],
                "abstention_reason": "x",
                "unexpected": 1,
            }
        )


# --- métricas calculadas ----------------------------------------------------


def test_metrics_computed_properties_not_persisted() -> None:
    m = OllamaMetricsV1(
        total_duration_ns=2_000_000_000, eval_count=44, eval_duration_ns=20_000_000_000
    )
    assert m.total_duration_s == pytest.approx(2.0)
    assert m.tokens_per_second == pytest.approx(2.2)
    # Las propiedades calculadas NO se serializan (no persistidas).
    dumped = m.model_dump()
    assert "tokens_per_second" not in dumped and "total_duration_s" not in dumped


def test_metrics_tokens_per_second_zero_division_safe() -> None:
    assert OllamaMetricsV1().tokens_per_second == 0.0


# --- rag_answer_v1 ----------------------------------------------------------


def test_rag_answer_final_accepts_full_shape() -> None:
    ans = RagAnswerV1(
        answered=True,
        answer="El plazo es de un mes.",
        citations=[
            AnswerCitationV1(
                evidence_id="E1",
                parent_id="BOE-A-2015-10565__a122",
                document_id="BOE-A-2015-10565",
                block_id="a122",
                label="Ley 39/2015, artículo 122",
                url="https://www.boe.es/buscar/act.php?id=BOE-A-2015-10565#a122",
            )
        ],
        abstention_reason="",
        disclaimer="Aviso informativo.",
        retrieval_trace=_trace(),
        generation_metrics=OllamaMetricsV1(),
    )
    assert ans.schema_version == "rag_answer_v1"
    assert ans.citations[0].url.endswith("#a122")


def _citation() -> AnswerCitationV1:
    return AnswerCitationV1(
        evidence_id="E1",
        parent_id="BOE-A-2015-10565__a122",
        document_id="BOE-A-2015-10565",
        block_id="a122",
        label="Ley 39/2015, artículo 122",
        url="https://www.boe.es/buscar/act.php?id=BOE-A-2015-10565#a122",
    )


def test_rag_answer_answered_without_answer_rejected() -> None:
    with pytest.raises(ValidationError):
        RagAnswerV1(
            answered=True,
            answer="   ",
            citations=[_citation()],
            abstention_reason="",
            disclaimer="x",
            retrieval_trace=_trace(),
        )


def test_rag_answer_answered_without_citations_rejected() -> None:
    with pytest.raises(ValidationError):
        RagAnswerV1(
            answered=True,
            answer="Respuesta.",
            citations=[],
            abstention_reason="",
            disclaimer="x",
            retrieval_trace=_trace(),
        )


def test_rag_answer_abstention_with_answer_rejected() -> None:
    with pytest.raises(ValidationError):
        RagAnswerV1(
            answered=False,
            answer="Respuesta.",
            citations=[],
            abstention_reason="motivo",
            disclaimer="x",
            retrieval_trace=_trace(),
        )


def test_rag_answer_abstention_without_reason_rejected() -> None:
    with pytest.raises(ValidationError):
        RagAnswerV1(
            answered=False,
            answer="",
            citations=[],
            abstention_reason="",
            disclaimer="x",
            retrieval_trace=_trace(),
        )


def test_rag_answer_extra_field_forbidden() -> None:
    with pytest.raises(ValidationError):
        RagAnswerV1.model_validate(
            {
                "answered": False,
                "answer": "",
                "abstention_reason": "x",
                "disclaimer": "y",
                "retrieval_trace": _trace().model_dump(),
                "unexpected": True,
            }
        )


# --- drift / determinismo ---------------------------------------------------


def test_generation_schemas_exist() -> None:
    for name in GENERATION_ROOT_MODELS:
        assert (SCHEMAS_DIR / f"{name}.schema.json").is_file(), f"falta schema de {name}"


def test_no_generation_schema_drift() -> None:
    drifted = [d for d in check(SCHEMAS_DIR) if d in GENERATION_ROOT_MODELS]
    assert drifted == [], (
        f"hay drift en contratos de generación; regenera con export_schemas: {drifted}"
    )


def test_generation_schema_json_deterministic() -> None:
    a = schema_json(RagAnswerV1)
    b = schema_json(RagAnswerV1)
    assert a == b and a.endswith("\n")
