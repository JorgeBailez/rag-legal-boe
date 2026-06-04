"""Tests de los contratos densos de Fase 2 y de la ausencia de drift en sus JSON Schema."""

import pytest
from pydantic import ValidationError

from src.contracts.embedding_models import (
    EMBEDDING_ROOT_MODELS,
    DenseEmbeddingBundleV1,
    DenseEmbeddingRowV1,
    DenseEmbeddingValidationReportV1,
)
from src.contracts.export_schemas import SCHEMAS_DIR, check, schema_json


def _chunk_field_row() -> dict:
    return {
        "row_index": 0,
        "embedding_input_id": "ein_000000",
        "document_id": "BOE-A-2015-10565",
        "block_id": "a9",
        "parent_id": "BOE-A-2015-10565__a9",
        "source": {"kind": "chunk_field", "chunk_id": "X__a9__c001", "field": "retrieval_text"},
        "context_anchor": {"paragraph_start": 1, "paragraph_end": 4},
        "token_count": 186,
        "formatted_input_sha256": "0" * 64,
    }


def _derived_row() -> dict:
    return {
        "row_index": 1,
        "embedding_input_id": "ein_000001",
        "document_id": "BOE-A-2015-10565",
        "block_id": "a9",
        "parent_id": "BOE-A-2015-10565__a9",
        "source": {
            "kind": "derived_text",
            "origin": "overflow_repair",
            "chunk_id": "X__a9__c001",
            "text": "fragmento",
            "token_start": 0,
            "token_end": 410,
            "segment_index": 0,
            "segment_count": 2,
        },
        "token_count": 410,
        "formatted_input_sha256": "1" * 64,
    }


# --- rows -------------------------------------------------------------------


def test_row_chunk_field_accepted() -> None:
    row = DenseEmbeddingRowV1.model_validate(_chunk_field_row())
    assert row.source.kind == "chunk_field"
    assert row.context_anchor.paragraph_end == 4


def test_row_derived_text_accepted() -> None:
    row = DenseEmbeddingRowV1.model_validate(_derived_row())
    assert row.source.kind == "derived_text"
    assert row.source.origin == "overflow_repair"
    assert row.context_anchor is None


def test_row_extra_field_forbidden() -> None:
    bad = _chunk_field_row()
    bad["unexpected"] = 1
    with pytest.raises(ValidationError):
        DenseEmbeddingRowV1.model_validate(bad)


def test_row_source_discriminator_rejects_unknown_kind() -> None:
    bad = _chunk_field_row()
    bad["source"] = {"kind": "nope", "chunk_id": "x", "field": "text"}
    with pytest.raises(ValidationError):
        DenseEmbeddingRowV1.model_validate(bad)


def test_row_chunk_field_rejects_derived_fields() -> None:
    # Campos del derivado no pueden colarse en un source chunk_field (extra="forbid").
    bad = _chunk_field_row()
    bad["source"] = {
        "kind": "chunk_field",
        "chunk_id": "x",
        "field": "text",
        "token_start": 0,
    }
    with pytest.raises(ValidationError):
        DenseEmbeddingRowV1.model_validate(bad)


# --- bundle / validation report ---------------------------------------------


def _bundle() -> dict:
    return {
        "schema_version": "dense_embedding_bundle_v1",
        "bundle": {
            "bundle_id": "bge-m3__j1__8f31a9c42d7e",
            "model_alias": "bge-m3",
            "model_id": "BAAI/bge-m3",
            "view": "J1",
            "created_at": "2026-06-03T00:00:00Z",
            "overflow_policy": "repair",
        },
        "corpus": {
            "n_norms": 10,
            "n_source_chunks": 3272,
            "n_rows": 3272,
            "source_corpus_fingerprint": "a" * 64,
            "embedding_inputs_fingerprint": "b" * 64,
        },
        "document_embedding_contract": {
            "model_id": "BAAI/bge-m3",
            "model_revision": None,
            "tokenizer_id": "BAAI/bge-m3",
            "tokenizer_revision": None,
            "declared_max_tokens": 8192,
            "effective_max_tokens": 8192,
            "expected_embedding_dimension": 1024,
            "embedding_dimension": 1024,
            "document_template": "{text}",
            "pooling": "cls",
            "normalize_embeddings": True,
            "trust_remote_code": False,
            "remote_code_reviewed": False,
            "revision_pinned": False,
            "document_contract_fingerprint": "c" * 64,
            "overlap_tokens": 100,
        },
        "execution": {
            "device": "cpu",
            "threads": 8,
            "batch_size": 32,
            "duration_seconds": 12.5,
            "throughput_inputs_per_second": 261.7,
            "encoder_backend": "fake",
            "library_versions": {"numpy": "2.4.6"},
            "allow_unpinned_revision": True,
        },
        "artifacts": {
            "embeddings": {"path": "embeddings.npy", "sha256": "d" * 64, "size_bytes": 100},
            "rows": {"path": "rows.jsonl", "sha256": "e" * 64, "size_bytes": 200},
            "validation_report": {
                "path": "validation_report.json",
                "sha256": "f" * 64,
                "size_bytes": 50,
            },
            "n_rows": 3272,
            "embedding_dimension": 1024,
            "dtype": "float32",
        },
        "validation": {
            "gate_a_passed": True,
            "gate_b_passed": True,
            "n_errors": 0,
            "n_warnings": 0,
            "n_info": 1,
        },
    }


def test_bundle_manifest_accepted() -> None:
    DenseEmbeddingBundleV1.model_validate(_bundle())


def test_bundle_rejects_unknown_view() -> None:
    bad = _bundle()
    bad["bundle"]["view"] = "J9"
    with pytest.raises(ValidationError):
        DenseEmbeddingBundleV1.model_validate(bad)


def test_validation_report_accepted() -> None:
    DenseEmbeddingValidationReportV1.model_validate(
        {
            "bundle_id": "bge-m3__j1__8f31a9c42d7e",
            "gate_a_passed": True,
            "gate_b_passed": True,
            "n_rows": 10,
            "embedding_dimension": 1024,
            "summary": {"error": 0, "warning": 1, "info": 2},
            "findings": [
                {
                    "gate": "B",
                    "check": "duplicate_vectors",
                    "severity": "WARNING",
                    "message": "2 inputs distintos con vector idéntico",
                }
            ],
            "checks_run": ["dtype", "norm_l2", "ids_unique"],
            "bootstrap_seed": 12345,
        }
    )


# --- drift / determinismo ----------------------------------------------------


def test_dense_schemas_exist() -> None:
    for name in EMBEDDING_ROOT_MODELS:
        assert (SCHEMAS_DIR / f"{name}.schema.json").is_file(), f"falta schema de {name}"


def test_no_dense_schema_drift() -> None:
    drifted = [d for d in check(SCHEMAS_DIR) if d in EMBEDDING_ROOT_MODELS]
    assert drifted == [], f"hay drift en contratos densos; regenera con export_schemas: {drifted}"


def test_dense_schema_json_deterministic() -> None:
    a = schema_json(DenseEmbeddingRowV1)
    b = schema_json(DenseEmbeddingRowV1)
    assert a == b and a.endswith("\n")
