"""Tests de los contratos Pydantic v2 y de la ausencia de drift en los JSON Schema."""

import json

import pytest
from pydantic import ValidationError

from src.contracts.export_schemas import SCHEMAS_DIR, check, schema_json
from src.contracts.models import (
    ROOT_MODELS,
    ChunksV2,
    DocumentV2,
    HistoryV2,
    ParentsV2,
)


def _min_document() -> dict:
    return {
        "schema_version": "boe_legal_document_v2",
        "document_id": "BOE-A-2015-10565",
        "source": {"name": "BOE", "manifest_ref": "data/manifests/BOE-A-2015-10565.json"},
        "metadata": {},
        "analysis": {"subjects": [], "notes": [], "references": {"previous": [], "next": []}},
        "blocks": [],
        "generation_meta": {"generated_at": "2026-06-02T00:00:00Z", "generator": "test"},
    }


# --- validación local por artefacto -----------------------------------------


def test_valid_document_accepted() -> None:
    DocumentV2.model_validate(_min_document())


def test_extra_field_forbidden() -> None:
    doc = _min_document()
    doc["unexpected"] = 1
    with pytest.raises(ValidationError):
        DocumentV2.model_validate(doc)


def test_wrong_schema_version_rejected() -> None:
    doc = _min_document()
    doc["schema_version"] = "boe_legal_document_v1"
    with pytest.raises(ValidationError):
        DocumentV2.model_validate(doc)


def test_chunks_requires_position_and_filters() -> None:
    bad = {
        "schema_version": "boe_legal_chunks_v2",
        "document_id": "X",
        "source_refs": {"document": "a", "parents": "b"},
        "chunking_strategy": {
            "name": "s",
            "max_chars": 1800,
            "overlap_paragraphs": 1,
            "split_unit": "paragraphs",
            "parent_unit": "boe_block",
        },
        "chunks": [
            {
                "chunk_id": "X__a1__c001",
                "parent_id": "X__a1",
                "document_id": "X",
                "block_id": "a1",
                "text": "t",
                "retrieval_text": "t",
            }
        ],  # falta position/citation/filters
        "generation_meta": {"generated_at": "t", "generator": "g"},
    }
    with pytest.raises(ValidationError):
        ChunksV2.model_validate(bad)


def test_parent_record_rejects_indexable_field() -> None:
    parents = {
        "schema_version": "boe_legal_parents_v2",
        "document_id": "X",
        "parents": [
            {
                "parent_id": "X__a1",
                "document_id": "X",
                "block_id": "a1",
                "order": 1,
                "text": "t",
                "paragraphs": [],
                "citation": {"label": "L"},
                "current_version": {},
                "indexable": True,  # prohibido en parents
            }
        ],
        "generation_meta": {"generated_at": "t", "generator": "g"},
    }
    with pytest.raises(ValidationError):
        ParentsV2.model_validate(parents)


def test_parent_record_rejects_modification_notes() -> None:
    parents = {
        "schema_version": "boe_legal_parents_v2",
        "document_id": "X",
        "parents": [
            {
                "parent_id": "X__a1",
                "document_id": "X",
                "block_id": "a1",
                "order": 1,
                "text": "t",
                "paragraphs": [],
                "citation": {"label": "L"},
                "current_version": {},
                "modification_notes": [],  # propiedad de history, prohibido en parents
            }
        ],
        "generation_meta": {"generated_at": "t", "generator": "g"},
    }
    with pytest.raises(ValidationError):
        ParentsV2.model_validate(parents)


def test_history_minimal_valid() -> None:
    HistoryV2.model_validate(
        {
            "schema_version": "boe_legal_history_v2",
            "document_id": "X",
            "blocks": [{"block_id": "a1", "temporal_resolution": {"status": "resolved"}}],
            "generation_meta": {"generated_at": "t", "generator": "g"},
        }
    )


# --- drift de schemas --------------------------------------------------------


def test_schemas_exist_for_every_root_model() -> None:
    for name in ROOT_MODELS:
        assert (SCHEMAS_DIR / f"{name}.schema.json").is_file(), f"falta schema de {name}"


def test_no_schema_drift() -> None:
    # Los schemas versionados deben coincidir EXACTAMENTE con los regenerados desde Pydantic.
    assert check(SCHEMAS_DIR) == [], "hay drift; regenera con export_schemas"


def test_schema_json_is_deterministic() -> None:
    a = schema_json(DocumentV2)
    b = schema_json(DocumentV2)
    assert a == b and a.endswith("\n")
    parsed = json.loads(a)
    assert parsed["title"] == "DocumentV2"
