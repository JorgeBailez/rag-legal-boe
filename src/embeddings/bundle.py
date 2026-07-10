"""Bundle denso persistente: staging → validar → checksums → manifest → rename atómico → inmutable.

Estructura publicada:

    data/indexes/dense/<bundle_id>/
    ├── manifest.json            (legible, secciones anidadas; dense_embedding_bundle_v1)
    ├── embeddings.npy           (numpy float32 [n_rows, dim])
    ├── rows.jsonl               (una dense_embedding_row_v1 por línea)
    └── validation_report.json   (dense_embedding_validation_report_v1)

`bundle_id = <model_alias>__<view_lower>__<bundle_identity_hash_12>`. No se sobrescriben bundles
existentes. Si cualquier validación falla, no se publica y se limpia el staging.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

import numpy as np

from src.contracts.embedding_models import (
    DenseEmbeddingBundleV1,
    DenseEmbeddingRowV1,
    DenseEmbeddingValidationReportV1,
)
from src.embeddings.fingerprints import (
    bundle_identity_fingerprint,
    document_contract_fingerprint,
    embedding_inputs_fingerprint,
    fingerprint,
)
from src.embeddings.fingerprints import (
    source_corpus_fingerprint as compute_source_corpus_fingerprint,
)
from src.embeddings.input_preparation import OVERLAP_TOKENS, PreparedInputs
from src.embeddings.model_registry import ModelContract
from src.embeddings.validation import (
    build_validation_report,
    has_errors,
    revision_pinned,
    run_gate_b,
)

DEFAULT_OUTPUT_ROOT = Path("data/indexes/dense")
STAGING_DIRNAME = ".staging"
_TRACKED_LIBS = ("numpy", "torch", "sentence-transformers", "transformers")


class BundleExistsError(RuntimeError):
    """El bundle de destino ya existe; los bundles publicados son inmutables."""


class BundleValidationError(RuntimeError):
    """Gate B falló: no se publica el bundle."""


@dataclass
class ExecutionMeta:
    """Metadatos de ejecución para el manifest (los aporta el orquestador)."""

    device: str = "cpu"
    threads: int = 8
    batch_size: int = 32
    duration_seconds: float = 0.0
    encoder_backend: str = "sentence-transformers"
    allow_unpinned_revision: bool = False


def library_versions() -> dict[str, str]:
    """Versiones instaladas de las librerías relevantes (para reproducibilidad)."""
    out: dict[str, str] = {}
    for name in _TRACKED_LIBS:
        try:
            out[name] = version(name)
        except PackageNotFoundError:  # pragma: no cover - depende del entorno
            continue
    return out


def compute_bundle_id(
    *,
    model_alias: str,
    view: str,
    document_contract_fp: str,
    source_corpus_fp: str,
    inputs_fp: str,
) -> str:
    identity_fp = bundle_identity_fingerprint(
        document_contract_fingerprint=document_contract_fp,
        source_corpus_fingerprint=source_corpus_fp,
        embedding_inputs_fingerprint=inputs_fp,
    )
    return f"{model_alias}__{view.lower()}__{identity_fp[:12]}"


def _sha256_file(path: Path) -> tuple[str, int]:
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data)


def _write_embeddings(path: Path, embeddings: np.ndarray) -> None:
    arr = np.ascontiguousarray(embeddings, dtype=np.float32)
    with open(path, "wb") as fh:
        np.save(fh, arr, allow_pickle=False)


def _write_rows_jsonl(path: Path, rows: list[dict]) -> None:
    lines = []
    for row in rows:
        DenseEmbeddingRowV1.model_validate(row)  # fail-fast por fila
        lines.append(json.dumps(row, ensure_ascii=False))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _build_manifest(
    *,
    bundle_id: str,
    contract: ModelContract,
    view: str,
    prepared: PreparedInputs,
    embeddings: np.ndarray,
    source_corpus_fingerprint: str,
    inputs_fp: str,
    doc_fp: str,
    n_norms: int,
    overflow_policy: str,
    overlap: int,
    execution: ExecutionMeta,
    report: dict,
    artifacts: dict,
) -> dict:
    n_rows = len(prepared.rows)
    dim = int(embeddings.shape[1]) if embeddings.ndim == 2 else 0
    duration = execution.duration_seconds
    throughput = round(n_rows / duration, 3) if duration > 0 else 0.0
    manifest = {
        "schema_version": "dense_embedding_bundle_v1",
        "bundle": {
            "bundle_id": bundle_id,
            "model_alias": contract.alias,
            "model_id": contract.model_id,
            "view": view,
            "created_at": datetime.now(UTC).isoformat(),
            "overflow_policy": overflow_policy,
        },
        "corpus": {
            "n_norms": n_norms,
            "n_source_chunks": prepared.report.get("n_source_chunks", 0),
            "n_rows": n_rows,
            "source_corpus_fingerprint": source_corpus_fingerprint,
            "embedding_inputs_fingerprint": inputs_fp,
        },
        "document_embedding_contract": {
            "model_id": contract.model_id,
            "model_revision": contract.model_revision,
            "tokenizer_id": contract.effective_tokenizer_id,
            "tokenizer_revision": contract.tokenizer_revision,
            "declared_max_tokens": contract.declared_max_tokens,
            "effective_max_tokens": prepared.effective_max_tokens,
            "expected_embedding_dimension": contract.expected_embedding_dimension,
            "embedding_dimension": dim,
            "document_template": contract.document_template,
            "pooling": contract.pooling,
            "normalize_embeddings": contract.normalize_embeddings,
            "trust_remote_code": contract.trust_remote_code,
            "remote_code_reviewed": contract.remote_code_reviewed,
            "revision_pinned": revision_pinned(contract),
            "document_contract_fingerprint": doc_fp,
            "overlap_tokens": overlap,
        },
        "execution": {
            "device": execution.device,
            "threads": execution.threads,
            "batch_size": execution.batch_size,
            "duration_seconds": round(duration, 3),
            "throughput_inputs_per_second": throughput,
            "encoder_backend": execution.encoder_backend,
            "library_versions": library_versions(),
            "allow_unpinned_revision": execution.allow_unpinned_revision,
        },
        "artifacts": {
            "embeddings": artifacts["embeddings"],
            "rows": artifacts["rows"],
            "validation_report": artifacts["validation_report"],
            "n_rows": n_rows,
            "embedding_dimension": dim,
            "dtype": "float32",
        },
        "validation": {
            "gate_a_passed": report["gate_a_passed"],
            "gate_b_passed": report["gate_b_passed"],
            "n_errors": report["summary"]["error"],
            "n_warnings": report["summary"]["warning"],
            "n_info": report["summary"]["info"],
        },
    }
    DenseEmbeddingBundleV1.model_validate(manifest)  # fail-fast
    return manifest


def publish_bundle(
    *,
    contract: ModelContract,
    view: str,
    prepared: PreparedInputs,
    embeddings: np.ndarray,
    source_corpus_fingerprint: str,
    n_norms: int,
    execution: ExecutionMeta,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    overflow_policy: str = "repair",
    gate_a_findings: list[dict] | None = None,
    bootstrap_seed: int | None = None,
) -> dict:
    """Escribe, valida (Gate B), construye el manifest y publica el bundle de forma atómica."""
    output_root = Path(output_root)
    gate_a_findings = gate_a_findings or []
    if has_errors(gate_a_findings):
        errs = [f for f in gate_a_findings if f["severity"] == "ERROR"]
        raise BundleValidationError(f"Gate A con {len(errs)} errores: {errs[:3]}")
    if not revision_pinned(contract):
        raise BundleValidationError(
            f"{contract.alias}: no se publica un bundle sin model/tokenizer revision fijadas"
        )
    if execution.allow_unpinned_revision:
        raise BundleValidationError(
            "--allow-unpinned-revision solo es válido para exploración; no publica bundles"
        )
    if contract.trust_remote_code and not contract.remote_code_reviewed:
        raise BundleValidationError(
            f"{contract.alias}: trust_remote_code=True exige remote_code_reviewed=True "
            "para publicar"
        )

    overlap = prepared.report.get("overlap_tokens", OVERLAP_TOKENS)
    doc_fp = document_contract_fingerprint(
        contract,
        view=view,
        effective_max_tokens=prepared.effective_max_tokens,
        overflow_policy=overflow_policy,
        overlap=overlap,
    )
    inputs_fp = embedding_inputs_fingerprint(prepared.rows)
    bundle_id = compute_bundle_id(
        model_alias=contract.alias,
        view=view,
        document_contract_fp=doc_fp,
        source_corpus_fp=source_corpus_fingerprint,
        inputs_fp=inputs_fp,
    )
    target_dir = output_root / bundle_id
    if target_dir.exists():
        raise BundleExistsError(
            f"el bundle {bundle_id} ya existe en {target_dir}; "
            "los bundles publicados son inmutables"
        )

    staging_dir = output_root / STAGING_DIRNAME / f"{bundle_id}__{uuid4().hex[:8]}"
    staging_dir.mkdir(parents=True, exist_ok=True)
    try:
        emb_path = staging_dir / "embeddings.npy"
        rows_path = staging_dir / "rows.jsonl"
        vr_path = staging_dir / "validation_report.json"
        man_path = staging_dir / "manifest.json"

        _write_embeddings(emb_path, embeddings)
        _write_rows_jsonl(rows_path, prepared.rows)

        emb = np.ascontiguousarray(embeddings, dtype=np.float32)
        gate_b = run_gate_b(emb, prepared.rows, expected_dim=contract.expected_embedding_dimension)
        report = build_validation_report(
            bundle_id=bundle_id,
            gate_a_findings=gate_a_findings,
            gate_b_findings=gate_b,
            n_rows=len(prepared.rows),
            embedding_dimension=int(emb.shape[1]) if emb.ndim == 2 else 0,
            bootstrap_seed=bootstrap_seed,
        )
        _write_json(vr_path, report)
        if has_errors(gate_b):
            errs = [f for f in gate_b if f["severity"] == "ERROR"]
            raise BundleValidationError(f"Gate B con {len(errs)} errores: {errs[:3]}")

        emb_sha, emb_size = _sha256_file(emb_path)
        rows_sha, rows_size = _sha256_file(rows_path)
        vr_sha, vr_size = _sha256_file(vr_path)
        artifacts = {
            "embeddings": {"path": "embeddings.npy", "sha256": emb_sha, "size_bytes": emb_size},
            "rows": {"path": "rows.jsonl", "sha256": rows_sha, "size_bytes": rows_size},
            "validation_report": {
                "path": "validation_report.json",
                "sha256": vr_sha,
                "size_bytes": vr_size,
            },
        }
        manifest = _build_manifest(
            bundle_id=bundle_id,
            contract=contract,
            view=view,
            prepared=prepared,
            embeddings=emb,
            source_corpus_fingerprint=source_corpus_fingerprint,
            inputs_fp=inputs_fp,
            doc_fp=doc_fp,
            n_norms=n_norms,
            overflow_policy=overflow_policy,
            overlap=overlap,
            execution=execution,
            report=report,
            artifacts=artifacts,
        )
        _write_json(man_path, manifest)

        os.replace(staging_dir, target_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    return {
        "bundle_id": bundle_id,
        "path": target_dir,
        "manifest": manifest,
        "validation_report": report,
    }


# --------------------------------------------------------------------------- #
# Carga / revalidación
# --------------------------------------------------------------------------- #


def _load_bundle_unchecked(bundle_dir: Path) -> tuple[dict, list[dict], np.ndarray]:
    """Carga cruda de un bundle publicado. Uso privado; no valida contratos ni checksums."""
    bundle_dir = Path(bundle_dir)
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (bundle_dir / "rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    embeddings = np.load(bundle_dir / "embeddings.npy", mmap_mode="r", allow_pickle=False)
    return manifest, rows, embeddings


def _validate_artifact_checksums(bundle_dir: Path, manifest: dict) -> None:
    for name in ("embeddings", "rows", "validation_report"):
        art = manifest["artifacts"][name]
        path = bundle_dir / art["path"]
        if not path.is_file():
            raise BundleValidationError(f"falta el artefacto {art['path']}")
        sha, size = _sha256_file(path)
        if sha != art["sha256"]:
            raise BundleValidationError(f"checksum no coincide para {art['path']}")
        if size != art["size_bytes"]:
            raise BundleValidationError(f"size_bytes no coincide para {art['path']}")


def _validate_artifact_paths(manifest: dict) -> None:
    expected = {
        "embeddings": "embeddings.npy",
        "rows": "rows.jsonl",
        "validation_report": "validation_report.json",
    }
    for name, expected_path in expected.items():
        actual = manifest["artifacts"][name]["path"]
        if actual != expected_path:
            raise BundleValidationError(f"artifact path inválido para {name}: {actual!r}")


def _document_contract_fingerprint_from_manifest(manifest: dict) -> str:
    dec = manifest["document_embedding_contract"]
    return fingerprint(
        {
            "model_id": dec["model_id"],
            "model_revision": dec["model_revision"],
            "tokenizer_id": dec["tokenizer_id"],
            "tokenizer_revision": dec["tokenizer_revision"],
            "declared_max_tokens": dec["declared_max_tokens"],
            "effective_max_tokens": dec["effective_max_tokens"],
            "expected_embedding_dimension": dec["expected_embedding_dimension"],
            "document_template": dec["document_template"],
            "pooling": dec["pooling"],
            "normalize_embeddings": dec["normalize_embeddings"],
            "trust_remote_code": dec["trust_remote_code"],
            "view": manifest["bundle"]["view"],
            "overflow_policy": manifest["bundle"]["overflow_policy"],
            "overlap_tokens": dec["overlap_tokens"],
        }
    )


def _validate_revision_policy(manifest: dict) -> None:
    dec = manifest["document_embedding_contract"]
    tokenizer_same_as_model = dec["tokenizer_id"] == dec["model_id"]
    tokenizer_pinned = tokenizer_same_as_model or dec["tokenizer_revision"] is not None
    if not dec["revision_pinned"] or dec["model_revision"] is None or not tokenizer_pinned:
        raise BundleValidationError("bundle sin revisiones fijadas para modelo/tokenizer")
    if dec["trust_remote_code"] and not dec["remote_code_reviewed"]:
        raise BundleValidationError("trust_remote_code=True exige remote_code_reviewed=True")
    if manifest["execution"]["allow_unpinned_revision"]:
        raise BundleValidationError("bundle generado con --allow-unpinned-revision")


def load_validated_bundle(
    bundle_dir: Path, *, corpus: dict | None = None
) -> tuple[dict, list[dict], np.ndarray]:
    """Carga pública validada de un bundle publicado.

    Con `corpus` (chunks + parents en disco) se valida además la consistencia bundle-corpus
    (`source_corpus_fingerprint`, `n_norms`, existencia de parents). Con `corpus=None` se omiten
    SOLO esas comprobaciones dependientes del corpus; el resto de la validación interna del bundle
    (esquema del manifest, `bundle_id`, gates A/B, política de revisión, checksums, esquema de
    rows/embeddings, `embedding_inputs_fingerprint`, Gate B) sigue siendo obligatoria. Útil para
    consumidores que solo puntúan con embeddings+rows y no resuelven texto (p. ej. abstención
    top-1) sin arrastrar `data/processed/`.
    """
    bundle_dir = Path(bundle_dir)
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    try:
        manifest = DenseEmbeddingBundleV1.model_validate(manifest).model_dump(mode="json")
    except Exception as exc:  # noqa: BLE001
        raise BundleValidationError("manifest.json no cumple dense_embedding_bundle_v1") from exc
    bundle_id = manifest["bundle"]["bundle_id"]
    if bundle_dir.name != bundle_id:
        raise BundleValidationError(
            f"nombre de directorio incompatible: {bundle_dir.name!r} != {bundle_id!r}"
        )
    _validate_artifact_paths(manifest)
    doc_fp = _document_contract_fingerprint_from_manifest(manifest)
    if doc_fp != manifest["document_embedding_contract"]["document_contract_fingerprint"]:
        raise BundleValidationError("document_contract_fingerprint no coincide con el manifest")
    expected_bundle_id = compute_bundle_id(
        model_alias=manifest["bundle"]["model_alias"],
        view=manifest["bundle"]["view"],
        document_contract_fp=doc_fp,
        source_corpus_fp=manifest["corpus"]["source_corpus_fingerprint"],
        inputs_fp=manifest["corpus"]["embedding_inputs_fingerprint"],
    )
    if bundle_id != expected_bundle_id:
        raise BundleValidationError("bundle_id no coincide con los fingerprints del manifest")
    if not manifest["validation"]["gate_a_passed"] or not manifest["validation"]["gate_b_passed"]:
        raise BundleValidationError("manifest indica Gate A/B no aprobado")
    _validate_revision_policy(manifest)
    _validate_artifact_checksums(bundle_dir, manifest)

    report = json.loads(
        (bundle_dir / manifest["artifacts"]["validation_report"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    try:
        report = DenseEmbeddingValidationReportV1.model_validate(report).model_dump(mode="json")
    except Exception as exc:  # noqa: BLE001
        raise BundleValidationError(
            "validation_report.json no cumple dense_embedding_validation_report_v1"
        ) from exc
    if report["bundle_id"] != manifest["bundle"]["bundle_id"]:
        raise BundleValidationError("validation_report.bundle_id no coincide con el manifest")
    if not report["gate_a_passed"] or not report["gate_b_passed"]:
        raise BundleValidationError("validation_report indica Gate A/B no aprobado")

    rows: list[dict] = []
    rows_path = bundle_dir / manifest["artifacts"]["rows"]["path"]
    for i, line in enumerate(rows_path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                rows.append(DenseEmbeddingRowV1.model_validate_json(line).model_dump(mode="json"))
            except Exception as exc:  # noqa: BLE001
                raise BundleValidationError(f"rows.jsonl línea {i} inválida") from exc
    embeddings = np.load(
        bundle_dir / manifest["artifacts"]["embeddings"]["path"],
        mmap_mode="r",
        allow_pickle=False,
    )
    arr = np.asarray(embeddings)
    if manifest["artifacts"]["n_rows"] != len(rows):
        raise BundleValidationError("artifacts.n_rows no coincide con rows.jsonl")
    dim = int(arr.shape[1]) if arr.ndim == 2 else 0
    if manifest["artifacts"]["embedding_dimension"] != dim:
        raise BundleValidationError("artifacts.embedding_dimension no coincide con embeddings.npy")
    if manifest["artifacts"]["dtype"] != "float32":
        raise BundleValidationError("artifacts.dtype debe ser float32")
    if manifest["corpus"]["n_rows"] != len(rows):
        raise BundleValidationError("corpus.n_rows no coincide con rows.jsonl")

    rows_fp = embedding_inputs_fingerprint(rows)
    if rows_fp != manifest["corpus"]["embedding_inputs_fingerprint"]:
        raise BundleValidationError("embedding_inputs_fingerprint no coincide con rows.jsonl")

    # Comprobaciones que exigen el corpus en disco: se omiten si `corpus is None` (ver docstring).
    if corpus is not None:
        if "n_norms" in corpus and manifest["corpus"]["n_norms"] != corpus["n_norms"]:
            raise BundleValidationError("corpus.n_norms no coincide con el corpus actual")
        current_corpus_fp = compute_source_corpus_fingerprint(
            corpus.get("chunks", []), corpus.get("parents_by_id", {})
        )
        if current_corpus_fp != manifest["corpus"]["source_corpus_fingerprint"]:
            raise BundleValidationError("source_corpus_fingerprint obsoleto para el corpus actual")
        missing_parents = sorted(
            {r["parent_id"] for r in rows} - set((corpus.get("parents_by_id") or {}).keys())
        )
        if missing_parents:
            raise BundleValidationError(f"rows con parent_id inexistente: {missing_parents[:5]}")

    missing_anchors = [r["embedding_input_id"] for r in rows if r.get("context_anchor") is None]
    if missing_anchors:
        raise BundleValidationError(f"rows sin context_anchor: {missing_anchors[:5]}")

    gate_b = run_gate_b(
        arr,
        rows,
        expected_dim=manifest["document_embedding_contract"]["expected_embedding_dimension"],
    )
    if has_errors(gate_b):
        errs = [f for f in gate_b if f["severity"] == "ERROR"]
        raise BundleValidationError(f"Gate B con {len(errs)} errores: {errs[:3]}")
    return manifest, rows, embeddings


def revalidate_bundle(bundle_dir: Path, *, corpus: dict) -> dict:
    """Revalida un bundle existente con la misma carga pública que consulta/benchmark."""
    try:
        manifest, rows, embeddings = load_validated_bundle(bundle_dir, corpus=corpus)
        arr = np.asarray(embeddings)
        gate_b = run_gate_b(
            arr,
            rows,
            expected_dim=manifest["document_embedding_contract"]["expected_embedding_dimension"],
        )
        return build_validation_report(
            bundle_id=manifest["bundle"]["bundle_id"],
            gate_a_findings=[],
            gate_b_findings=gate_b,
            n_rows=len(rows),
            embedding_dimension=int(arr.shape[1]) if arr.ndim == 2 else 0,
        )
    except Exception as exc:  # noqa: BLE001 - se reporta como ERROR de carga, no se propaga
        return build_validation_report(
            bundle_id=Path(bundle_dir).name,
            gate_a_findings=[],
            gate_b_findings=[
                {
                    "gate": "B",
                    "check": "validated_load",
                    "severity": "ERROR",
                    "message": str(exc),
                    "evidence": type(exc).__name__,
                }
            ],
            n_rows=0,
            embedding_dimension=0,
        )
