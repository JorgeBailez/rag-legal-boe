"""Tests relacionales del corpus v2 real + casos de error inyectado (auditoría)."""

import copy
import json
from pathlib import Path

import pytest

from src.quality.corpus_audit import (
    check_chunks,
    check_history,
    check_parents,
    check_relational,
    join_norm,
)

PROCESSED = Path("data/processed")
NORMS = ["BOE-A-1985-5392", "BOE-A-2003-20977"]


def _load(norm_id: str) -> tuple[dict, dict, dict, dict]:
    def rd(sub: str) -> dict:
        return json.loads((PROCESSED / sub / f"{norm_id}.json").read_text(encoding="utf-8"))

    return rd("documents"), rd("histories"), rd("parents"), rd("chunks")


def _errors(findings: list[dict]) -> list[str]:
    return [f["check"] for f in findings if f["severity"] == "ERROR"]


@pytest.fixture(params=NORMS)
def artifacts(request):
    if not (PROCESSED / "documents" / f"{request.param}.json").is_file():
        pytest.skip("corpus v2 no generado (ejecuta process_mvp_corpus.py)")
    return _load(request.param)


# --- corpus real limpio ------------------------------------------------------


def test_real_corpus_is_relationally_clean(artifacts) -> None:
    document, history, parents, chunks = artifacts
    joined = join_norm(document, history, parents)
    findings = (
        check_history(document, history)
        + check_parents(document, parents)
        + check_chunks(chunks, joined)
        + check_relational(document, history, parents, chunks)
    )
    assert _errors(findings) == []


def test_no_chunk_has_parent_text(artifacts) -> None:
    _, _, _, chunks = artifacts
    assert all("parent_text" not in c for c in chunks["chunks"])


def test_every_resolved_block_with_text_has_one_parent(artifacts) -> None:
    document, _, parents, _ = artifacts
    parent_ids = [p["parent_id"] for p in parents["parents"]]
    assert len(parent_ids) == len(set(parent_ids))  # sin duplicados
    with_parent = {b["block_id"] for b in document["blocks"] if b["parent_id"] is not None}
    parent_blocks = {p["block_id"] for p in parents["parents"]}
    assert with_parent == parent_blocks


# --- errores inyectados ------------------------------------------------------


def test_orphan_chunk_detected(artifacts) -> None:
    document, history, parents, chunks = artifacts
    bad = copy.deepcopy(chunks)
    bad["chunks"][0]["parent_id"] = "BOE-A-9999-1__zzz"
    checks = _errors(check_relational(document, history, parents, bad))
    assert "rel.chunk_parent_missing" in checks


def test_unknown_subject_code_detected(artifacts) -> None:
    document, history, parents, chunks = artifacts
    bad = copy.deepcopy(chunks)
    bad["chunks"][0]["filters"]["subject_codes"] = ["ZZZ-unknown"]
    checks = _errors(check_relational(document, history, parents, bad))
    assert "rel.subject_code_unknown" in checks


def test_missing_history_block_detected(artifacts) -> None:
    document, history, _, _ = artifacts
    bad = copy.deepcopy(history)
    bad["blocks"] = bad["blocks"][:-1]  # quita un block_id
    checks = _errors(check_history(document, bad))
    assert "history.coverage" in checks


def test_duplicate_parent_detected(artifacts) -> None:
    document, _, parents, _ = artifacts
    bad = copy.deepcopy(parents)
    bad["parents"].append(copy.deepcopy(bad["parents"][0]))
    checks = _errors(check_parents(document, bad))
    assert "parents.duplicate" in checks


def test_parent_with_indexable_field_detected(artifacts) -> None:
    document, _, parents, _ = artifacts
    bad = copy.deepcopy(parents)
    bad["parents"][0]["indexable"] = True
    checks = _errors(check_parents(document, bad))
    assert "parents.indexable_present" in checks


def test_current_version_mismatch_detected(artifacts) -> None:
    document, history, parents, chunks = artifacts
    bad = copy.deepcopy(parents)
    bad["parents"][0]["current_version"]["publication_date"] = "1900-01-01"
    checks = _errors(check_relational(document, history, bad, chunks))
    assert "rel.current_version_mismatch" in checks
