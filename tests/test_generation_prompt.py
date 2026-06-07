"""Tests del prompt restrictivo (delimitación, IDs, schema, abstención, llaves en texto legal)."""

from src.generation.prompt import build_messages, build_user_prompt
from src.retrieval.evidence_builder import GenerationEvidence

_PLACEHOLDERS = ("{schema}", "{allowed_ids}", "{evidences}", "{question}")


def _evidence(
    evidence_id: str, text: str, *, label: str = "Ley 1/2000, art. 1"
) -> GenerationEvidence:
    return GenerationEvidence(
        evidence_id=evidence_id,
        parent_id="BOE-A-0001__a1",
        document_id="BOE-A-0001",
        block_id="a1",
        label=label,
        url="https://boe/BOE-A-0001#a1",
        score=0.9,
        retrieval_rank=1,
        context_strategy="P_EXPAND_BOUNDED",
        text=text,
    )


def test_user_prompt_contains_question_and_delimited_evidences() -> None:
    evs = [_evidence("E1", "El plazo general será de tres meses.")]
    prompt = build_user_prompt(question="¿Cuál es el plazo general?", evidences=evs)
    assert "¿Cuál es el plazo general?" in prompt
    assert "El plazo general será de tres meses." in prompt
    assert "--- E1" in prompt and "fin E1" in prompt  # evidencia delimitada


def test_user_prompt_lists_allowed_ids() -> None:
    evs = [_evidence("E1", "uno"), _evidence("E2", "dos")]
    prompt = build_user_prompt(question="x", evidences=evs)
    assert "E1, E2" in prompt


def test_user_prompt_embeds_serialized_schema() -> None:
    prompt = build_user_prompt(question="x", evidences=[_evidence("E1", "uno")])
    assert "RagLlmAnswerV1" in prompt
    assert '"citation_ids"' in prompt and '"answered"' in prompt


def test_system_prompt_states_abstention_and_untrusted_data_rules() -> None:
    messages = build_messages(question="x", evidences=[_evidence("E1", "uno")])
    assert messages[0]["role"] == "system" and messages[1]["role"] == "user"
    system = messages[0]["content"].lower()
    assert "abst" in system  # regla de abstención
    assert "datos" in system and "no instrucciones" in system  # datos no confiables
    assert "citation_ids" in messages[0]["content"]  # cita solo por IDs entregados


def test_no_residual_template_placeholders() -> None:
    prompt = build_user_prompt(question="pregunta normal", evidences=[_evidence("E1", "uno")])
    for token in _PLACEHOLDERS:
        assert token not in prompt, f"placeholder sin sustituir: {token}"


def test_legal_text_with_braces_does_not_break_rendering() -> None:
    # El texto jurídico contiene llaves y hasta un marcador que coincide con un placeholder.
    ev = _evidence("E1", "Artículo con {llaves} y un literal {question} embebido.")
    prompt = build_user_prompt(question="MARCA_PREGUNTA_UNICA", evidences=[ev])
    assert "{llaves}" in prompt  # las llaves del texto legal se conservan
    assert "{question}" in prompt  # el literal del texto legal NO se sustituye
    assert prompt.count("MARCA_PREGUNTA_UNICA") == 1  # solo la pregunta real se inyecta
