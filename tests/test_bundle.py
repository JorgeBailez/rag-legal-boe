"""Tests de Gate B, publicación atómica del bundle, recarga mmap y revalidación (offline)."""

import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from src.embeddings.bundle import (
    BundleExistsError,
    BundleValidationError,
    ExecutionMeta,
    load_validated_bundle,
    publish_bundle,
    revalidate_bundle,
)
from src.embeddings.fingerprints import source_corpus_fingerprint
from src.embeddings.input_preparation import prepare_inputs
from src.embeddings.model_registry import (
    ModelContract,
    assert_bundle_compatible,
    get_contract,
    query_profile_metadata,
)
from src.embeddings.validation import has_errors, run_gate_b
from tests.dense_fakes import FakeEncoder, FakeWordTokenizer, synthetic_corpus

TEST_CONTRACT = ModelContract(
    alias="fake-test",
    model_id="fake/test",
    declared_max_tokens=512,
    expected_embedding_dimension=8,
    model_revision="deadbeefcafe",  # fijado → Gate A no bloquea
)


def _emb(n: int, d: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, d)).astype(np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return (v / norms).astype(np.float32)


def _rows(n: int) -> list[dict]:
    return [
        {
            "row_index": i,
            "embedding_input_id": f"ein_{i:06d}",
            "parent_id": "p",
            "formatted_input_sha256": f"{i}",
        }
        for i in range(n)
    ]


# --- Gate B ------------------------------------------------------------------


def test_gate_b_clean_matrix_no_errors() -> None:
    findings = run_gate_b(_emb(4), _rows(4), expected_dim=8)
    assert not has_errors(findings)


def test_gate_b_detects_nan_and_dtype() -> None:
    emb = _emb(3)
    emb[0, 0] = np.nan
    assert any(f["check"] == "nan" for f in run_gate_b(emb, _rows(3), expected_dim=8))
    emb64 = _emb(3).astype(np.float64)
    assert any(f["check"] == "dtype" for f in run_gate_b(emb64, _rows(3), expected_dim=8))


def test_gate_b_detects_null_vector_and_dim() -> None:
    emb = _emb(3)
    emb[1, :] = 0.0
    assert any(f["check"] == "null_vectors" for f in run_gate_b(emb, _rows(3), expected_dim=8))
    assert any(f["check"] == "dimension" for f in run_gate_b(_emb(2), _rows(2), expected_dim=16))


def test_gate_b_duplicate_ids_error_and_duplicate_vectors_warning() -> None:
    rows = _rows(2)
    rows[1]["embedding_input_id"] = rows[0]["embedding_input_id"]  # id duplicado → ERROR
    findings = run_gate_b(_emb(2), rows, expected_dim=8)
    assert any(f["check"] == "ids_unique" and f["severity"] == "ERROR" for f in findings)

    emb = _emb(2)
    emb[1] = emb[0]  # mismo vector, inputs distintos → WARNING
    findings = run_gate_b(emb, _rows(2), expected_dim=8)
    assert any(f["check"] == "duplicate_vectors" and f["severity"] == "WARNING" for f in findings)


# --- publicación / recarga ---------------------------------------------------


def _prepare_and_embed():
    corpus = synthetic_corpus()
    tok = FakeWordTokenizer(model_max_length=512, special=2)
    prepared = prepare_inputs(
        "J1",
        chunks=corpus["chunks"],
        parents_by_id=corpus["parents_by_id"],
        contract=TEST_CONTRACT,
        tokenizer=tok,
    )
    emb = FakeEncoder(dimension=8).encode_documents(prepared.texts)
    scfp = source_corpus_fingerprint(corpus["chunks"], corpus["parents_by_id"])
    return prepared, emb, scfp


def test_publish_and_reload_via_mmap(tmp_path) -> None:
    prepared, emb, scfp = _prepare_and_embed()
    result = publish_bundle(
        contract=TEST_CONTRACT,
        view="J1",
        prepared=prepared,
        embeddings=emb,
        source_corpus_fingerprint=scfp,
        n_norms=2,
        execution=ExecutionMeta(encoder_backend="fake"),
        output_root=tmp_path,
    )
    bundle_dir = result["path"]
    assert bundle_dir.is_dir()
    assert result["bundle_id"].startswith("fake-test__j1__")
    # ficheros esperados
    for fname in ("manifest.json", "embeddings.npy", "rows.jsonl", "validation_report.json"):
        assert (bundle_dir / fname).is_file()
    # recarga por mmap, solo lectura, sin pickle
    manifest, rows, loaded = load_validated_bundle(bundle_dir, corpus=synthetic_corpus())
    assert loaded.dtype == np.float32
    assert loaded.shape == (len(prepared.rows), 8)
    assert len(rows) == len(prepared.rows)
    assert manifest["artifacts"]["dtype"] == "float32"
    assert manifest["bundle"]["view"] == "J1"
    # rows.jsonl: una row por línea, JSON válido
    lines = (bundle_dir / "rows.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(prepared.rows)
    json.loads(lines[0])


def test_publish_is_immutable_without_force(tmp_path) -> None:
    prepared, emb, scfp = _prepare_and_embed()
    kwargs = dict(
        contract=TEST_CONTRACT,
        view="J1",
        prepared=prepared,
        embeddings=emb,
        source_corpus_fingerprint=scfp,
        n_norms=2,
        execution=ExecutionMeta(encoder_backend="fake"),
        output_root=tmp_path,
    )
    publish_bundle(**kwargs)
    with pytest.raises(BundleExistsError):
        publish_bundle(**kwargs)


def test_publish_rejects_unpinned_contract(tmp_path) -> None:
    prepared, emb, scfp = _prepare_and_embed()
    unpinned = replace(TEST_CONTRACT, model_revision=None)
    with pytest.raises(BundleValidationError, match="revision"):
        publish_bundle(
            contract=unpinned,
            view="J1",
            prepared=prepared,
            embeddings=emb,
            source_corpus_fingerprint=scfp,
            n_norms=2,
            execution=ExecutionMeta(encoder_backend="fake"),
            output_root=tmp_path,
        )


def test_publish_rejects_gate_a_error(tmp_path) -> None:
    prepared, emb, scfp = _prepare_and_embed()
    with pytest.raises(BundleValidationError, match="Gate A"):
        publish_bundle(
            contract=TEST_CONTRACT,
            view="J1",
            prepared=prepared,
            embeddings=emb,
            source_corpus_fingerprint=scfp,
            n_norms=2,
            execution=ExecutionMeta(encoder_backend="fake"),
            output_root=tmp_path,
            gate_a_findings=[
                {
                    "gate": "A",
                    "check": "corpus_readiness",
                    "severity": "ERROR",
                    "message": "no listo",
                    "evidence": None,
                }
            ],
        )


def test_bundle_id_changes_when_corpus_fingerprint_changes(tmp_path) -> None:
    prepared, emb, scfp = _prepare_and_embed()
    first = publish_bundle(
        contract=TEST_CONTRACT,
        view="J1",
        prepared=prepared,
        embeddings=emb,
        source_corpus_fingerprint=scfp,
        n_norms=2,
        execution=ExecutionMeta(encoder_backend="fake"),
        output_root=tmp_path,
    )
    second = publish_bundle(
        contract=TEST_CONTRACT,
        view="J1",
        prepared=prepared,
        embeddings=emb,
        source_corpus_fingerprint="0" * 64,
        n_norms=2,
        execution=ExecutionMeta(encoder_backend="fake"),
        output_root=tmp_path,
    )
    assert first["bundle_id"] != second["bundle_id"]


def test_bundle_id_changes_when_prepared_inputs_change(tmp_path) -> None:
    prepared, emb, scfp = _prepare_and_embed()
    first = publish_bundle(
        contract=TEST_CONTRACT,
        view="J1",
        prepared=prepared,
        embeddings=emb,
        source_corpus_fingerprint=scfp,
        n_norms=2,
        execution=ExecutionMeta(encoder_backend="fake"),
        output_root=tmp_path,
    )
    changed = deepcopy(prepared)
    changed.rows[0]["formatted_input_sha256"] = "changed"
    second = publish_bundle(
        contract=TEST_CONTRACT,
        view="J1",
        prepared=changed,
        embeddings=emb,
        source_corpus_fingerprint=scfp,
        n_norms=2,
        execution=ExecutionMeta(encoder_backend="fake"),
        output_root=tmp_path,
    )
    assert first["bundle_id"] != second["bundle_id"]


def test_publish_aborts_and_cleans_staging_on_gate_b_error(tmp_path) -> None:
    prepared, emb, scfp = _prepare_and_embed()
    emb_bad = emb.copy()
    emb_bad[0, 0] = np.nan  # provoca ERROR en Gate B
    with pytest.raises(BundleValidationError):
        publish_bundle(
            contract=TEST_CONTRACT,
            view="J1",
            prepared=prepared,
            embeddings=emb_bad,
            source_corpus_fingerprint=scfp,
            n_norms=2,
            execution=ExecutionMeta(encoder_backend="fake"),
            output_root=tmp_path,
        )
    # No se publicó ningún bundle y el staging quedó limpio.
    assert not any(p.name != ".staging" for p in tmp_path.iterdir() if p.is_dir())
    staging = tmp_path / ".staging"
    assert not staging.exists() or not any(staging.iterdir())


def test_revalidate_detects_tampering(tmp_path) -> None:
    prepared, emb, scfp = _prepare_and_embed()
    result = publish_bundle(
        contract=TEST_CONTRACT,
        view="J1",
        prepared=prepared,
        embeddings=emb,
        source_corpus_fingerprint=scfp,
        n_norms=2,
        execution=ExecutionMeta(encoder_backend="fake"),
        output_root=tmp_path,
    )
    bundle_dir = result["path"]
    # bundle intacto → revalidación OK
    report = revalidate_bundle(bundle_dir, corpus=synthetic_corpus())
    assert report["gate_b_passed"]
    assert all(f["severity"] != "ERROR" for f in report["findings"])
    # manipular rows.jsonl → checksum ERROR
    (bundle_dir / "rows.jsonl").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(BundleValidationError, match="checksum"):
        load_validated_bundle(bundle_dir, corpus=synthetic_corpus())


def test_validated_loader_rejects_manifest_with_extra_fields(tmp_path) -> None:
    prepared, emb, scfp = _prepare_and_embed()
    result = publish_bundle(
        contract=TEST_CONTRACT,
        view="J1",
        prepared=prepared,
        embeddings=emb,
        source_corpus_fingerprint=scfp,
        n_norms=2,
        execution=ExecutionMeta(encoder_backend="fake"),
        output_root=tmp_path,
    )
    manifest_path = result["path"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["extra"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BundleValidationError, match="manifest"):
        load_validated_bundle(result["path"], corpus=synthetic_corpus())


def test_validated_loader_rejects_obsolete_corpus_fingerprint(tmp_path) -> None:
    prepared, emb, scfp = _prepare_and_embed()
    result = publish_bundle(
        contract=TEST_CONTRACT,
        view="J1",
        prepared=prepared,
        embeddings=emb,
        source_corpus_fingerprint=scfp,
        n_norms=2,
        execution=ExecutionMeta(encoder_backend="fake"),
        output_root=tmp_path,
    )
    obsolete = deepcopy(synthetic_corpus())
    obsolete["parents_by_id"]["BOE-A-0001__a1"]["text"] = "texto distinto"
    with pytest.raises(BundleValidationError, match="source_corpus_fingerprint"):
        load_validated_bundle(result["path"], corpus=obsolete)


def test_validated_loader_accepts_no_corpus(tmp_path) -> None:
    """corpus=None omite SOLO las comprobaciones bundle-corpus; el resto de la validación sigue."""
    prepared, emb, scfp = _prepare_and_embed()
    result = publish_bundle(
        contract=TEST_CONTRACT,
        view="J1",
        prepared=prepared,
        embeddings=emb,
        source_corpus_fingerprint=scfp,
        n_norms=2,
        execution=ExecutionMeta(encoder_backend="fake"),
        output_root=tmp_path,
    )
    bundle_dir = result["path"]
    # sin corpus en disco la carga funciona (caso abstención top-1)
    manifest, rows, emb_loaded = load_validated_bundle(bundle_dir, corpus=None)
    assert manifest["bundle"]["bundle_id"] == bundle_dir.name
    assert emb_loaded.shape[0] == len(rows) == manifest["artifacts"]["n_rows"]
    # pero la validación interna (checksums) sigue activa aunque no haya corpus
    (bundle_dir / "rows.jsonl").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(BundleValidationError, match="checksum"):
        load_validated_bundle(bundle_dir, corpus=None)


def test_validated_loader_rejects_document_fingerprint_tampering(tmp_path) -> None:
    prepared, emb, scfp = _prepare_and_embed()
    result = publish_bundle(
        contract=TEST_CONTRACT,
        view="J1",
        prepared=prepared,
        embeddings=emb,
        source_corpus_fingerprint=scfp,
        n_norms=2,
        execution=ExecutionMeta(encoder_backend="fake"),
        output_root=tmp_path,
    )
    manifest_path = result["path"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["document_embedding_contract"]["document_contract_fingerprint"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BundleValidationError, match="document_contract_fingerprint"):
        load_validated_bundle(result["path"], corpus=synthetic_corpus())


def test_validated_loader_rejects_directory_name_mismatch(tmp_path) -> None:
    prepared, emb, scfp = _prepare_and_embed()
    result = publish_bundle(
        contract=TEST_CONTRACT,
        view="J1",
        prepared=prepared,
        embeddings=emb,
        source_corpus_fingerprint=scfp,
        n_norms=2,
        execution=ExecutionMeta(encoder_backend="fake"),
        output_root=tmp_path,
    )
    wrong_dir = tmp_path / "wrong_bundle_name"
    result["path"].rename(wrong_dir)
    with pytest.raises(BundleValidationError, match="nombre de directorio"):
        load_validated_bundle(wrong_dir, corpus=synthetic_corpus())


def test_validated_loader_rejects_altered_artifact_path(tmp_path) -> None:
    prepared, emb, scfp = _prepare_and_embed()
    result = publish_bundle(
        contract=TEST_CONTRACT,
        view="J1",
        prepared=prepared,
        embeddings=emb,
        source_corpus_fingerprint=scfp,
        n_norms=2,
        execution=ExecutionMeta(encoder_backend="fake"),
        output_root=tmp_path,
    )
    manifest_path = result["path"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["rows"]["path"] = "nested/rows.jsonl"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BundleValidationError, match="artifact path"):
        load_validated_bundle(result["path"], corpus=synthetic_corpus())


def test_source_corpus_fingerprint_changes_when_parent_paragraph_structure_changes() -> None:
    corpus = synthetic_corpus()
    base = source_corpus_fingerprint(corpus["chunks"], corpus["parents_by_id"])

    changed_order = deepcopy(corpus)
    changed_order["parents_by_id"]["BOE-A-0001__a1"]["paragraphs"][0]["order"] = 99
    assert (
        source_corpus_fingerprint(changed_order["chunks"], changed_order["parents_by_id"]) != base
    )

    changed_text = deepcopy(corpus)
    changed_text["parents_by_id"]["BOE-A-0001__a1"]["paragraphs"][0]["text"] = "distinto"
    assert source_corpus_fingerprint(changed_text["chunks"], changed_text["parents_by_id"]) != base


def test_registry_incompatibility_with_bundle_is_rejected(tmp_path) -> None:
    prepared, emb, scfp = _prepare_and_embed()
    result = publish_bundle(
        contract=TEST_CONTRACT,
        view="J1",
        prepared=prepared,
        embeddings=emb,
        source_corpus_fingerprint=scfp,
        n_norms=2,
        execution=ExecutionMeta(encoder_backend="fake"),
        output_root=tmp_path,
    )
    incompatible = replace(TEST_CONTRACT, model_revision="other")
    with pytest.raises(ValueError, match="registry incompatible"):
        assert_bundle_compatible(incompatible, result["manifest"])


def test_query_profile_does_not_enter_bundle_identity(tmp_path) -> None:
    prepared, emb, scfp = _prepare_and_embed()
    result = publish_bundle(
        contract=TEST_CONTRACT,
        view="J1",
        prepared=prepared,
        embeddings=emb,
        source_corpus_fingerprint=scfp,
        n_norms=2,
        execution=ExecutionMeta(encoder_backend="fake"),
        output_root=tmp_path,
    )
    instruct_contract = get_contract("e5-large-instruct")
    generic = query_profile_metadata(instruct_contract, "I0_GENERIC")
    legal = query_profile_metadata(instruct_contract, "I1_LEGAL")
    assert generic["query_profile_fingerprint"] != legal["query_profile_fingerprint"]
    assert "query_profile_id" not in json.dumps(result["manifest"])
