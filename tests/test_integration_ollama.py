"""Prueba REAL contra un Ollama local. Desactivada por defecto.

Activar con:
    RUN_OLLAMA_INTEGRATION=1 uv run --locked pytest tests/test_integration_ollama.py -q -s

No requiere bundle denso ni pesos de Hugging Face: usa un contexto jurídico breve controlado,
el JSON Schema del contrato y comprueba respuesta fundamentada y abstención. Solo depende de que
`ollama serve` esté en marcha con el modelo configurado (`OLLAMA_MODEL`).
"""

import os

import pytest

from src.config.settings import get_settings
from src.contracts.generation_models import RagLlmAnswerV1
from src.generation.ollama_client import OllamaClient
from src.generation.prompt import build_messages
from src.retrieval.evidence_builder import GenerationEvidence

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_OLLAMA_INTEGRATION") != "1",
    reason="prueba real de Ollama desactivada (exporta RUN_OLLAMA_INTEGRATION=1)",
)


def _evidence() -> GenerationEvidence:
    return GenerationEvidence(
        evidence_id="E1",
        parent_id="BOE-A-2015-10565__a122",
        document_id="BOE-A-2015-10565",
        block_id="a122",
        label="Ley 39/2015, artículo 122",
        url="https://www.boe.es/buscar/act.php?id=BOE-A-2015-10565#a122",
        score=0.9,
        retrieval_rank=1,
        context_strategy="P_EXPAND_BOUNDED",
        text=(
            "El plazo para la interposición del recurso de alzada será de un mes si el acto fuera "
            "expreso. Transcurrido dicho plazo sin haberse interpuesto, la resolución será firme."
        ),
    )


def _client() -> OllamaClient:
    s = get_settings()
    return OllamaClient(
        base_url=s.ollama_base_url,
        model=s.ollama_model,
        timeout=s.ollama_timeout_seconds,
        think=s.ollama_think,
    )


def test_ollama_grounded_answer_returns_valid_contract() -> None:
    client = _client()
    try:
        client.version()  # health check
        messages = build_messages(
            question="¿Qué plazo tengo para interponer un recurso de alzada si el acto es expreso?",
            evidences=[_evidence()],
        )
        answer, metrics = client.chat(messages, num_ctx=4096, num_predict=256)
        assert isinstance(answer, RagLlmAnswerV1)
        assert answer.answered is True
        assert answer.citation_ids == ["E1"]
        assert metrics.eval_count >= 0
    finally:
        _safe_cleanup(client)


def test_ollama_abstains_when_evidence_is_irrelevant() -> None:
    client = _client()
    try:
        client.version()
        messages = build_messages(
            question="¿Cuál es la cuantía máxima de una beca universitaria?",
            evidences=[_evidence()],  # evidencia no relacionada con becas
        )
        answer, _ = client.chat(messages, num_ctx=4096, num_predict=256)
        assert isinstance(answer, RagLlmAnswerV1)
        assert answer.answered is False
        assert answer.citation_ids == []
        assert answer.abstention_reason
    finally:
        _safe_cleanup(client)


def _safe_cleanup(client: OllamaClient) -> None:
    """Descarga y cierra el cliente sin enmascarar el error principal del test."""
    try:
        client.unload()
    except Exception:  # noqa: BLE001 - el cleanup no debe ocultar el fallo del test
        pass
    finally:
        client.close()
