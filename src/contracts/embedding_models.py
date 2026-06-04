"""Contratos Pydantic v2 de los artefactos densos de Fase 2 (fuente única de verdad).

Tres contratos raíz, estrictos (`extra="forbid"`):
- `dense_embedding_row_v1`        → una fila de `rows.jsonl` (un input de embedding + trazabilidad).
- `dense_embedding_bundle_v1`     → `manifest.json` (legible, anidado por secciones).
- `dense_embedding_validation_report_v1` → `validation_report.json` (gates, severidades, hallazgos).

Generan los JSON Schema de `schemas/` vía `export_schemas.py` (test anti-drift). No mezclan los
contratos jurídicos de Fase 1 (`src.contracts.models`); solo se registran como root models
exportables.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    """Base estricta: prohíbe campos no declarados (detecta contratos ajenos/derivas)."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# dense_embedding_row_v1 — una fila de rows.jsonl
# --------------------------------------------------------------------------- #


class ChunkFieldSource(_Strict):
    """El input proviene de un campo existente de un chunk de Fase 1 (referencia, no se duplica)."""

    kind: Literal["chunk_field"] = "chunk_field"
    chunk_id: str
    field: Literal["retrieval_text", "text"]


class DerivedTextSource(_Strict):
    """El input es texto NUEVO derivado en Fase 2 (se persiste aquí, con trazabilidad token-aware).

    `origin`:
    - `fixed_token_window`: ventana token-aware dentro de un parent (vista C1); `chunk_id` es None.
    - `overflow_repair`: subdivisión de un chunk que excedía el límite del modelo; `chunk_id` es el
      chunk origen.
    `segment_index`/`segment_count` posicionan el segmento dentro de su división (para overflow
    repair equivalen al overflow_index/overflow_count del encargo).
    """

    kind: Literal["derived_text"] = "derived_text"
    origin: Literal["fixed_token_window", "overflow_repair"]
    chunk_id: str | None = None
    text: str
    token_start: int
    token_end: int
    segment_index: int
    segment_count: int


RowSource = Annotated[ChunkFieldSource | DerivedTextSource, Field(discriminator="kind")]


class ContextAnchor(_Strict):
    """Rango de párrafos del parent cubierto por el input (1-based, inclusivo).

    Se resuelve para inputs basados en párrafos (J1/J2), para ventanas C1 por solape token-aware y
    para overflow heredando/refinando el chunk origen.
    """

    paragraph_start: int
    paragraph_end: int


class DenseEmbeddingRowV1(_Strict):
    """Fila de `rows.jsonl`: un input de embedding con trazabilidad (sin texto pesado de Fase 1)."""

    row_index: int
    embedding_input_id: str
    document_id: str
    block_id: str
    parent_id: str
    source: RowSource
    context_anchor: ContextAnchor | None = None
    token_count: int
    formatted_input_sha256: str


# --------------------------------------------------------------------------- #
# dense_embedding_bundle_v1 — manifest.json (anidado por secciones)
# --------------------------------------------------------------------------- #


class BundleSection(_Strict):
    bundle_id: str
    model_alias: str
    model_id: str
    view: Literal["J1", "J2", "C1"]
    created_at: str
    overflow_policy: str


class CorpusSection(_Strict):
    n_norms: int
    n_source_chunks: int
    n_rows: int
    source_corpus_fingerprint: str
    embedding_inputs_fingerprint: str


class DocumentEmbeddingContractSection(_Strict):
    model_id: str
    model_revision: str | None
    tokenizer_id: str
    tokenizer_revision: str | None
    declared_max_tokens: int
    effective_max_tokens: int
    expected_embedding_dimension: int
    embedding_dimension: int
    document_template: str
    pooling: str
    normalize_embeddings: bool
    trust_remote_code: bool
    remote_code_reviewed: bool
    revision_pinned: bool
    document_contract_fingerprint: str
    overlap_tokens: int


class ExecutionSection(_Strict):
    device: str
    threads: int
    batch_size: int
    duration_seconds: float
    throughput_inputs_per_second: float
    encoder_backend: str
    library_versions: dict[str, str]
    allow_unpinned_revision: bool


class ArtifactRef(_Strict):
    path: str
    sha256: str
    size_bytes: int


class ArtifactsSection(_Strict):
    embeddings: ArtifactRef
    rows: ArtifactRef
    validation_report: ArtifactRef
    n_rows: int
    embedding_dimension: int
    dtype: str


class ValidationSummarySection(_Strict):
    gate_a_passed: bool
    gate_b_passed: bool
    n_errors: int
    n_warnings: int
    n_info: int


class DenseEmbeddingBundleV1(_Strict):
    """`manifest.json` de un bundle denso publicado (legible, secciones anidadas)."""

    schema_version: Literal["dense_embedding_bundle_v1"] = "dense_embedding_bundle_v1"
    bundle: BundleSection
    corpus: CorpusSection
    document_embedding_contract: DocumentEmbeddingContractSection
    execution: ExecutionSection
    artifacts: ArtifactsSection
    validation: ValidationSummarySection


# --------------------------------------------------------------------------- #
# dense_embedding_validation_report_v1 — validation_report.json
# --------------------------------------------------------------------------- #


class SeveritySummary(_Strict):
    error: int = 0
    warning: int = 0
    info: int = 0


class ValidationFinding(_Strict):
    gate: Literal["A", "B"]
    check: str
    severity: Literal["ERROR", "WARNING", "INFO"]
    message: str
    evidence: str | None = None


class DenseEmbeddingValidationReportV1(_Strict):
    """`validation_report.json`: resultado de Gate A/B con severidades y hallazgos."""

    schema_version: Literal["dense_embedding_validation_report_v1"] = (
        "dense_embedding_validation_report_v1"
    )
    bundle_id: str
    gate_a_passed: bool
    gate_b_passed: bool
    n_rows: int
    embedding_dimension: int
    summary: SeveritySummary
    findings: list[ValidationFinding] = Field(default_factory=list)
    checks_run: list[str] = Field(default_factory=list)
    bootstrap_seed: int | None = None


# Registro de contratos raíz densos exportables (nombre de schema → modelo).
EMBEDDING_ROOT_MODELS: dict[str, type[BaseModel]] = {
    "dense_embedding_row_v1": DenseEmbeddingRowV1,
    "dense_embedding_bundle_v1": DenseEmbeddingBundleV1,
    "dense_embedding_validation_report_v1": DenseEmbeddingValidationReportV1,
}
