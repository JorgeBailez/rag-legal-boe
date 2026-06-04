"""Gates de validación del índice denso (A: pre-encoding; B: pre-publicación).

Severidades: ERROR bloquea la publicación; WARNING publica pero queda en el reporte; INFO es
diagnóstico. Cada hallazgo se ajusta al contrato `dense_embedding_validation_report_v1`.
"""

from __future__ import annotations

import numpy as np

from src.contracts.embedding_models import DenseEmbeddingValidationReportV1
from src.embeddings.input_preparation import PreparedInputs
from src.embeddings.model_registry import ModelContract

ERROR = "ERROR"
WARNING = "WARNING"
INFO = "INFO"


def _finding(
    gate: str, check: str, severity: str, message: str, evidence: str | None = None
) -> dict:
    return {
        "gate": gate,
        "check": check,
        "severity": severity,
        "message": message,
        "evidence": evidence,
    }


def has_errors(findings: list[dict]) -> bool:
    return any(f["severity"] == ERROR for f in findings)


def summarize_severity(findings: list[dict]) -> dict:
    return {
        "error": sum(1 for f in findings if f["severity"] == ERROR),
        "warning": sum(1 for f in findings if f["severity"] == WARNING),
        "info": sum(1 for f in findings if f["severity"] == INFO),
    }


def revision_pinned(contract: ModelContract) -> bool:
    """True si el modelo (y su tokenizer) tienen commit hash fijado."""
    tok_ok = contract.tokenizer_id is None or contract.tokenizer_revision is not None
    return contract.model_revision is not None and tok_ok


# --------------------------------------------------------------------------- #
# Gate A — antes de codificar
# --------------------------------------------------------------------------- #


def run_gate_a(
    *,
    readiness: dict | None,
    contract: ModelContract,
    allow_unpinned_revision: bool,
    prepared: PreparedInputs,
) -> list[dict]:
    """Valida corpus, contrato, revisiones e inputs preparados antes de codificar."""
    f: list[dict] = []

    if readiness is None:
        f.append(
            _finding(
                "A",
                "corpus_readiness",
                ERROR,
                "no se encontró pre_embedding_readiness (ejecuta scripts/audit_corpus.py)",
            )
        )
    elif not readiness.get("ready"):
        f.append(
            _finding(
                "A",
                "corpus_readiness",
                ERROR,
                "pre_embedding_readiness.ready=false",
                evidence=str(readiness.get("blocking_findings")),
            )
        )
    else:
        f.append(_finding("A", "corpus_readiness", INFO, "auditoría jurídica aprobada"))

    if not revision_pinned(contract):
        if allow_unpinned_revision:
            f.append(
                _finding(
                    "A",
                    "revision_pinned",
                    WARNING,
                    f"{contract.alias}: commit hash sin fijar (--allow-unpinned-revision)",
                )
            )
        else:
            f.append(
                _finding(
                    "A",
                    "revision_pinned",
                    ERROR,
                    f"{contract.alias}: commit hash sin fijar; fija las revisiones o usa "
                    "--allow-unpinned-revision",
                )
            )
    if contract.trust_remote_code and not contract.remote_code_reviewed:
        f.append(
            _finding(
                "A",
                "remote_code_reviewed",
                ERROR,
                f"{contract.alias}: trust_remote_code=True sin revisión local del código "
                "remoto (exige remote_code_reviewed=True para publicar)",
            )
        )

    rows = prepared.rows
    if not rows:
        f.append(_finding("A", "inputs_present", ERROR, "no se prepararon inputs"))
    if prepared.report.get("n_truncated", 0) != 0:
        f.append(
            _finding(
                "A",
                "no_truncation",
                ERROR,
                f"hay {prepared.report['n_truncated']} inputs truncados",
            )
        )

    ids = [r["embedding_input_id"] for r in rows]
    if len(set(ids)) != len(ids):
        f.append(_finding("A", "ids_unique", ERROR, "embedding_input_id duplicado"))
    if [r["row_index"] for r in rows] != list(range(len(rows))):
        f.append(_finding("A", "row_index_continuous", ERROR, "row_index no continuo"))

    bad_anchor = [
        r["embedding_input_id"]
        for r in rows
        if r.get("context_anchor")
        and not (
            1 <= r["context_anchor"]["paragraph_start"] <= r["context_anchor"]["paragraph_end"]
        )
    ]
    missing_anchor = [r["embedding_input_id"] for r in rows if r.get("context_anchor") is None]
    if bad_anchor:
        f.append(
            _finding(
                "A", "anchors_valid", ERROR, "context_anchor inválido", evidence=str(bad_anchor[:5])
            )
        )
    if missing_anchor:
        f.append(
            _finding(
                "A",
                "anchors_present",
                ERROR,
                "context_anchor obligatorio ausente",
                evidence=str(missing_anchor[:5]),
            )
        )

    repaired = prepared.report.get("n_overflow_repaired_inputs", 0)
    f.append(_finding("A", "overflow_repaired", INFO, f"inputs reparados por overflow: {repaired}"))
    discarded = prepared.report.get("n_auxiliary_context_windows_discarded", 0)
    if discarded:
        f.append(
            _finding(
                "A",
                "auxiliary_context_windows_discarded",
                INFO,
                f"ventanas auxiliares descartadas sin cobertura juridica: {discarded}",
                evidence=str(prepared.report.get("auxiliary_context_windows_discarded_sample", [])),
            )
        )
    return f


# --------------------------------------------------------------------------- #
# Gate B — antes de publicar
# --------------------------------------------------------------------------- #


def run_gate_b(
    embeddings: np.ndarray,
    rows: list[dict],
    *,
    expected_dim: int,
    l2_tol: float = 1e-2,
) -> list[dict]:
    """Valida la matriz de embeddings y su correspondencia con las rows antes de publicar."""
    f: list[dict] = []
    n = embeddings.shape[0] if embeddings.ndim >= 1 else 0

    if len(rows) != n:
        f.append(
            _finding("B", "rows_match", ERROR, f"n_rows={len(rows)} != filas de embeddings={n}")
        )
    if embeddings.ndim != 2:
        f.append(_finding("B", "shape", ERROR, f"embeddings no es 2D: ndim={embeddings.ndim}"))
        return f  # sin forma 2D no se puede seguir
    dim = embeddings.shape[1]
    if dim != expected_dim:
        f.append(_finding("B", "dimension", ERROR, f"dim={dim} != esperado={expected_dim}"))
    if embeddings.dtype != np.float32:
        f.append(_finding("B", "dtype", ERROR, f"dtype={embeddings.dtype} != float32"))

    nan = int(np.isnan(embeddings).sum())
    if nan:
        f.append(_finding("B", "nan", ERROR, f"hay {nan} valores NaN"))
    inf = int(np.isinf(embeddings).sum())
    if inf:
        f.append(_finding("B", "inf", ERROR, f"hay {inf} valores Inf"))

    norms = np.linalg.norm(embeddings, axis=1) if n else np.array([])
    null = int((norms == 0).sum())
    if null:
        f.append(_finding("B", "null_vectors", ERROR, f"hay {null} vectores nulos"))
    nonnull = norms[norms > 0]
    if nonnull.size:
        max_dev = float(np.abs(nonnull - 1.0).max())
        if max_dev > l2_tol:
            f.append(
                _finding(
                    "B",
                    "l2_norm",
                    ERROR,
                    f"norma L2 fuera de tolerancia (max desvío={max_dev:.4f})",
                )
            )

    if [r["row_index"] for r in rows] != list(range(len(rows))):
        f.append(_finding("B", "row_index_continuous", ERROR, "row_index no continuo"))
    ids = [r["embedding_input_id"] for r in rows]
    if len(set(ids)) != len(ids):
        f.append(_finding("B", "ids_unique", ERROR, "embedding_input_id duplicado"))
    if any(not r.get("parent_id") for r in rows):
        f.append(_finding("B", "parent_present", ERROR, "row sin parent_id"))

    # Vectores idénticos para inputs distintos → WARNING.
    groups: dict[bytes, set[str]] = {}
    for i in range(n):
        groups.setdefault(embeddings[i].tobytes(), set()).add(rows[i]["formatted_input_sha256"])
    dup_groups = sum(1 for v in groups.values() if len(v) > 1)
    if dup_groups:
        f.append(
            _finding(
                "B",
                "duplicate_vectors",
                WARNING,
                f"{dup_groups} grupos de inputs distintos con vector idéntico",
            )
        )

    f.append(_finding("B", "row_count", INFO, f"vectores={n}, dim={dim}"))
    return f


# --------------------------------------------------------------------------- #
# Reporte
# --------------------------------------------------------------------------- #


def build_validation_report(
    *,
    bundle_id: str,
    gate_a_findings: list[dict],
    gate_b_findings: list[dict],
    n_rows: int,
    embedding_dimension: int,
    bootstrap_seed: int | None = None,
) -> dict:
    """Ensambla y valida el `validation_report.json` (contrato Pydantic)."""
    findings = list(gate_a_findings) + list(gate_b_findings)
    report = {
        "bundle_id": bundle_id,
        "gate_a_passed": not has_errors(gate_a_findings),
        "gate_b_passed": not has_errors(gate_b_findings),
        "n_rows": n_rows,
        "embedding_dimension": embedding_dimension,
        "summary": summarize_severity(findings),
        "findings": findings,
        "checks_run": sorted({f["check"] for f in findings}),
        "bootstrap_seed": bootstrap_seed,
    }
    DenseEmbeddingValidationReportV1.model_validate(report)  # fail-fast
    return report
