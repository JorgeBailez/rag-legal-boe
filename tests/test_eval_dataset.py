"""Tests del validador del dataset de evaluación (contratos, reglas y Gate C)."""

from src.evaluation.dataset import DATASET_DIR, load_and_validate, validate_dataset
from tests.dense_fakes import synthetic_corpus


def _q(qid, split="development", family="fam_a", status="reviewed", scope="single_parent"):
    return {
        "query_id": qid,
        "query": "¿pregunta?",
        "split": split,
        "issue_family_id": family,
        "query_style": "ciudadana",
        "answer_scope": scope,
        "review_status": status,
    }


def _j(qid, parent="BOE-A-0001__a1", rel=2, status="reviewed", orders=None):
    row = {"query_id": qid, "parent_id": parent, "relevance": rel, "review_status": status}
    if rel >= 1:
        row["evidence"] = {"paragraph_orders": [1] if orders is None else orders}
        row["quote"] = "cita revisada"
    return row


def _formal_ready_dataset():
    questions = []
    judgments = []
    for i in range(40):
        qid = f"dev_{i:03d}"
        questions.append(_q(qid, "development", f"fam_dev_{i}"))
        judgments.append(_j(qid))
    for i in range(80):
        qid = f"test_{i:03d}"
        questions.append(_q(qid, "test", f"fam_test_{i}"))
        judgments.append(_j(qid))
    for i in range(20):
        questions.append(_q(f"ooc_{i:03d}", "out_of_corpus", f"fam_ooc_{i}", scope="none"))
    return questions, judgments


def test_versioned_scaffold_validates_structurally() -> None:
    # El dataset versionado (solo ejemplos) no tiene errores estructurales...
    report = load_and_validate(DATASET_DIR)
    assert report["errors"] == []
    # ...pero Gate C NO está listo (los ejemplos no son 'reviewed'/'final').
    assert report["gate_c"]["ready"] is False


def test_gate_c_ready_with_reviewed_annotation() -> None:
    questions, judgments = _formal_ready_dataset()
    report = validate_dataset(questions, judgments, corpus=synthetic_corpus())
    assert report["errors"] == []
    assert report["gate_c"]["ready"] is True


def test_cross_split_family_leakage_is_error() -> None:
    questions = [_q("q1", "development", "shared"), _q("q2", "test", "shared")]
    report = validate_dataset(questions, [])
    assert any("development y test" in e for e in report["errors"])
    assert report["gate_c"]["ready"] is False


def test_dangling_judgment_is_error() -> None:
    report = validate_dataset([_q("q1")], [_j("qZZ")])
    assert any("sin pregunta" in e for e in report["errors"])


def test_duplicate_query_id_is_error() -> None:
    report = validate_dataset([_q("q1"), _q("q1", family="fam_b")], [])
    assert any("duplicado" in e for e in report["errors"])


def test_bad_relevance_is_error() -> None:
    bad = {"query_id": "q1", "parent_id": "p", "relevance": 5, "review_status": "draft"}
    report = validate_dataset([_q("q1")], [bad])
    assert any("judgment[0]" in e for e in report["errors"])


def test_out_of_corpus_relevant_judgment_is_error() -> None:
    questions = [_q("q1", "out_of_corpus", "fam_ooc")]
    report = validate_dataset(questions, [_j("q1", rel=2)])
    assert any("out_of_corpus" in e for e in report["errors"])


def test_gate_c_rejects_tiny_reviewed_dataset() -> None:
    questions = [_q("q1", "development", "fam_dev"), _q("q2", "test", "fam_test")]
    judgments = [_j("q1"), _j("q2")]
    report = validate_dataset(questions, judgments, corpus=synthetic_corpus())
    assert report["errors"] == []
    assert report["gate_c"]["ready"] is False
    assert any("mínimo" in r for r in report["gate_c"]["reasons"])


def test_missing_parent_id_is_error() -> None:
    report = validate_dataset([_q("q1")], [_j("q1", parent="missing")], corpus=synthetic_corpus())
    assert any("parent_id inexistente" in e for e in report["errors"])


def test_invalid_paragraph_order_is_error() -> None:
    report = validate_dataset([_q("q1")], [_j("q1", orders=[99])], corpus=synthetic_corpus())
    assert any("paragraph_orders inexistentes" in e for e in report["errors"])


def test_duplicate_judgment_is_error() -> None:
    report = validate_dataset([_q("q1")], [_j("q1"), _j("q1")], corpus=synthetic_corpus())
    assert any("(query_id, parent_id) duplicado" in e for e in report["errors"])


def test_multi_parent_requires_two_relevant_parents() -> None:
    questions = [_q("q1", scope="multi_parent")]
    report = validate_dataset(questions, [_j("q1")], corpus=synthetic_corpus())
    assert any("multi_parent" in e for e in report["errors"])


def test_answer_scope_none_must_match_out_of_corpus_split() -> None:
    report = validate_dataset([_q("q1", split="development", scope="none")], [])
    assert any("answer_scope='none'" in e for e in report["errors"])

    report = validate_dataset([_q("q2", split="out_of_corpus", scope="single_parent")], [])
    assert any("answer_scope='none'" in e for e in report["errors"])


def test_empty_query_and_family_are_errors() -> None:
    q = _q("q1")
    q["query"] = " "
    q["issue_family_id"] = ""
    report = validate_dataset([q], [])
    assert any("query vacía" in e for e in report["errors"])
    assert any("issue_family_id vacío" in e for e in report["errors"])


# -- Honestidad: completitud solo en reviewed/final ------------------------------


def _draft_judgment(qid, parent="BOE-A-0001__a1", rel=2):
    # Borrador SIN evidencia ni cita: válido por ser draft (la revisión la completa después).
    return {"query_id": qid, "parent_id": parent, "relevance": rel, "review_status": "draft"}


def test_draft_judgment_without_evidence_is_not_error() -> None:
    q = _q("q1", status="draft")
    report = validate_dataset([q], [_draft_judgment("q1")], corpus=synthetic_corpus())
    assert report["errors"] == []


def test_reviewed_judgment_without_evidence_is_error() -> None:
    j = {
        "query_id": "q1",
        "parent_id": "BOE-A-0001__a1",
        "relevance": 2,
        "review_status": "reviewed",
    }
    report = validate_dataset([_q("q1")], [j], corpus=synthetic_corpus())
    assert any("evidence obligatoria" in e for e in report["errors"])


# -- answer_keys (gold de generación) -------------------------------------------


def _ak(qid, answerable=True, status="reviewed", **kw):
    row = {"query_id": qid, "answerable": answerable, "review_status": status}
    if answerable:
        row.setdefault("reference_answer", "respuesta de referencia")
        row.setdefault("expected_citation_parents", ["BOE-A-0001__a1"])
    row.update(kw)
    return row


def test_answer_key_dangling_query_is_error() -> None:
    report = validate_dataset([_q("q1")], [_j("q1")], answer_keys=[_ak("qZZ")])
    assert any("answer_key.query_id sin pregunta" in e for e in report["errors"])


def test_answer_key_answerable_in_out_of_corpus_is_error() -> None:
    q = _q("q1", "out_of_corpus", "fam_ooc", scope="none")
    report = validate_dataset([q], [], answer_keys=[_ak("q1", answerable=True)])
    assert any("answerable=true en out_of_corpus" in e for e in report["errors"])


def test_answer_key_expected_parent_must_exist_in_corpus() -> None:
    ak = _ak("q1", expected_citation_parents=["BOE-A-9999__zz"])
    report = validate_dataset([_q("q1")], [_j("q1")], answer_keys=[ak], corpus=synthetic_corpus())
    assert any("expected_citation_parents inexistentes" in e for e in report["errors"])


def test_draft_answer_key_incomplete_is_not_error() -> None:
    ak = {"query_id": "q1", "answerable": True, "review_status": "draft"}
    report = validate_dataset([_q("q1", status="draft")], [_draft_judgment("q1")], answer_keys=[ak])
    assert report["errors"] == []


def test_generation_ready_requires_reviewed_answer_keys() -> None:
    questions, judgments = _formal_ready_dataset()
    answer_keys = []
    for q in questions:
        if q["split"] == "out_of_corpus":
            answer_keys.append(_ak(q["query_id"], answerable=False))
        else:
            answer_keys.append(_ak(q["query_id"]))
    corpus = synthetic_corpus()
    # Sin answer_keys: retrieval listo pero generación NO.
    no_ak = validate_dataset(questions, judgments, corpus=corpus)
    assert no_ak["gate_c"]["ready"] is True
    assert no_ak["gate_c"]["generation_ready"] is False
    # Con answer_keys revisadas suficientes: generación lista.
    with_ak = validate_dataset(questions, judgments, answer_keys=answer_keys, corpus=corpus)
    assert with_ak["gate_c"]["generation_ready"] is True
