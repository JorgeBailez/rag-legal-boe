"""Tests del juez LLM (offline): contratos de veredicto, render de prompts y judge_agreement."""

import pytest

from src.contracts.generation_models import OllamaMetricsV1
from src.core.exceptions import GenerationContractError
from src.evaluation.judge import (
    CorrectnessVerdictV1,
    FaithfulnessVerdictV1,
    LlmJudge,
    judge_agreement,
)


class _FakeChatJsonClient:
    """Cliente fake: devuelve un dict fijo desde chat_json y registra lo recibido."""

    model = "fake-judge-model"

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def chat_json(self, messages, *, response_format, **kwargs):  # noqa: ANN001, ANN003
        self.calls.append({"messages": messages, "response_format": response_format, **kwargs})
        return self.payload, OllamaMetricsV1(eval_count=7)


def test_judge_faithfulness_parses_and_passes_evidence() -> None:
    client = _FakeChatJsonClient(
        {"claims": [{"claim": "x", "supported": True}, {"claim": "y", "supported": False}]}
    )
    judge = LlmJudge(client=client)
    verdict, metrics = judge.judge_faithfulness(answer="resp", evidences_block="EVIDENCIA-AQUI")
    assert isinstance(verdict, FaithfulnessVerdictV1)
    assert [c.supported for c in verdict.claims] == [True, False]
    assert metrics.eval_count == 7
    # El prompt del usuario debe contener la evidencia y la respuesta renderizadas.
    user_msg = client.calls[0]["messages"][1]["content"]
    assert "EVIDENCIA-AQUI" in user_msg and "resp" in user_msg


def test_judge_correctness_parses_label() -> None:
    client = _FakeChatJsonClient({"verdict": "partial", "rationale": "omite un matiz"})
    judge = LlmJudge(client=client)
    verdict, _ = judge.judge_correctness(question="¿?", answer="cand", reference="ref")
    assert isinstance(verdict, CorrectnessVerdictV1)
    assert verdict.verdict == "partial"


def test_judge_rejects_offcontract_output() -> None:
    client = _FakeChatJsonClient({"verdict": "maybe"})  # etiqueta fuera del Literal
    judge = LlmJudge(client=client)
    with pytest.raises(GenerationContractError):
        judge.judge_correctness(question="¿?", answer="c", reference="r")


def test_model_label_defaults_to_client_model() -> None:
    judge = LlmJudge(client=_FakeChatJsonClient({"claims": []}))
    assert judge.model_label == "fake-judge-model"


def test_judge_agreement_perfect() -> None:
    out = judge_agreement(["correct", "incorrect", "partial"], ["correct", "incorrect", "partial"])
    assert out["percent_agreement"] == 1.0
    assert out["cohens_kappa"] == 1.0
    assert out["n"] == 3


def test_judge_agreement_partial() -> None:
    out = judge_agreement(
        ["correct", "correct", "incorrect"], ["correct", "incorrect", "incorrect"]
    )
    assert out["percent_agreement"] == pytest.approx(2 / 3)
    assert out["cohens_kappa"] <= 1.0


def test_judge_agreement_length_mismatch() -> None:
    with pytest.raises(ValueError):
        judge_agreement(["a"], ["a", "b"])
