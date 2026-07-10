"""Tests del conversor scaffold ↔ Markdown de anotación del juez (offline, puro)."""

from scripts.annotation_worksheet import from_md, to_md


def test_to_md_hides_judge_verdict_and_flags_missing_reference() -> None:
    rows = [
        {
            "query_id": "q1",
            "question": "¿Pregunta?",
            "answer_text": "Respuesta",
            "reference_answer": "Referencia",
            "evidences_block": "E1 texto",
            "judge_correctness": "partial",
            "judge_faithful": True,
            "human_correctness": "",
            "human_faithful": None,
            "notes": "",
        },
        {
            "query_id": "q2",
            "question": "¿Otra?",
            "answer_text": "Resp2",
            "reference_answer": "",
            "evidences_block": "E1 texto",
            "human_correctness": "",
            "human_faithful": None,
        },
    ]
    md = to_md(rows)
    assert "## q1" in md and "## q2" in md
    # el veredicto del juez NO se vuelca a la hoja (anotación a ciegas):
    assert "judge_correctness" not in md and "judge_faithful" not in md
    assert "no anotes corrección" in md  # q2 sin referencia se marca


def test_from_md_parses_filled_annotation() -> None:
    md = (
        "# cabecera\n---\n"
        "## q1\n**>>> TU ANOTACIÓN <<<**\n"
        "FIDELIDAD: true\nCORRECCION: correct\nNOTAS: clara\n---\n"
        "## q2\n**>>> TU ANOTACIÓN <<<**\n"
        "FIDELIDAD: false\nCORRECCION:\nNOTAS:\n---\n"
    )
    rows, warnings = from_md(md)
    assert len(rows) == 2 and warnings == []
    assert rows[0]["query_id"] == "q1"
    assert rows[0]["human_faithful"] is True
    assert rows[0]["human_correctness"] == "correct"
    assert rows[0]["notes"] == "clara"
    assert rows[1]["human_faithful"] is False
    assert rows[1]["human_correctness"] == ""


def test_from_md_warns_on_unrecognized_values() -> None:
    md = "## q1\nFIDELIDAD: quizas\nCORRECCION: regular\nNOTAS:\n---\n"
    rows, warnings = from_md(md)
    assert rows[0]["human_faithful"] is None
    assert rows[0]["human_correctness"] == ""
    assert len(warnings) == 2


def test_blank_scaffold_round_trips_to_unannotated() -> None:
    rows = [
        {
            "query_id": "q1",
            "question": "¿P?",
            "answer_text": "R",
            "reference_answer": "Ref",
            "evidences_block": "E1",
            "human_correctness": "",
            "human_faithful": None,
        }
    ]
    parsed, _ = from_md(to_md(rows))
    assert parsed[0]["query_id"] == "q1"
    assert parsed[0]["human_faithful"] is None
    assert parsed[0]["human_correctness"] == ""
