"""Modelos Pydantic v2 de los contratos de datos persistidos (fuente única de verdad).

Cada contrato raíz fija `schema_version` con `Literal[...]` y prohíbe campos extra
(`extra="forbid"`), de modo que un JSON ajeno o con derivas se rechaza en validación. Estos
modelos generan los JSON Schema de `schemas/` (ver `export_schemas.py`).

Política de propiedad (resumen; detalle en `docs/modelo_documental.md`):
- Texto vigente + párrafos → SOLO `ParentsV2`.
- Historial + resolución temporal completa + notas de modificación → SOLO `HistoryV2`.
- Materias completas → SOLO `DocumentV2.analysis.subjects`; en chunks solo `subject_codes`.
- `indexable`/`excluded_reason` → SOLO `DocumentV2.blocks[]`.
- `retrieval_text` → SOLO `ChunksV2.chunks[]` (lo construye el chunker).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------- #
# Base
# --------------------------------------------------------------------------- #


class _Strict(BaseModel):
    """Base estricta: prohíbe campos no declarados (detecta contratos ajenos/derivas)."""

    model_config = ConfigDict(extra="forbid")


# Sub-modelos compartidos -----------------------------------------------------


class Coded(_Strict):
    """Valor codificado del BOE (`{code, label}`)."""

    code: str | None = None
    label: str | None = None


class Hierarchy(_Strict):
    """Jerarquía jurídica a 6 niveles (mutuamente excluyentes cuerpo/anexo)."""

    book: str | None = None
    title: str | None = None
    chapter: str | None = None
    section: str | None = None
    subsection: str | None = None
    annex: str | None = None


class Citation(_Strict):
    """Cita humana + URL oficial con ancla de bloque."""

    label: str = Field(..., examples=["Ley 7/1985, artículo 2"])
    url: str | None = Field(
        None, examples=["https://www.boe.es/buscar/act.php?id=BOE-A-1985-5392#a2"]
    )


class GenerationMeta(_Strict):
    """Metadatos mínimos de generación (trazabilidad de quién/cuándo)."""

    generated_at: str = Field(..., description="ISO-8601 UTC del momento de generación.")
    generator: str = Field(..., description="Componente/versión que generó el artefacto.")


# --------------------------------------------------------------------------- #
# 1) boe_legal_document_v2 — descriptor
# --------------------------------------------------------------------------- #


class DocumentMetadata(_Strict):
    """Metadatos documentales de la norma (idénticos a los del parser)."""

    title: str | None = None
    short_title: str | None = None
    identifier: str | None = None
    eli_url: str | None = None
    html_url: str | None = None
    scope: Coded | None = None
    department: Coded | None = None
    rank: Coded | None = None
    official_number: str | None = None
    publication_date: str | None = None
    document_date: str | None = None
    effective_date: str | None = None
    last_update_datetime: str | None = None
    consolidation_status: Coded | None = None
    derogation_status: str | None = None
    annulment_status: str | None = None
    expired_validity: str | None = None
    legal_status_notice: str | None = None


class Subject(_Strict):
    """Materia (subject) del análisis del BOE."""

    code: str | None = None
    label: str | None = None


class AnalysisNote(_Strict):
    text: str | None = None


class Reference(_Strict):
    target_norm_id: str | None = None
    relation: Coded | None = None
    text: str | None = None


class References(_Strict):
    previous: list[Reference] = Field(default_factory=list)
    next: list[Reference] = Field(default_factory=list)


class Analysis(_Strict):
    """Análisis: materias (propietario único), notas y referencias jurídicas."""

    subjects: list[Subject] = Field(default_factory=list)
    notes: list[AnalysisNote] = Field(default_factory=list)
    references: References = Field(default_factory=References)


class DocumentSource(_Strict):
    """Procedencia: ruta RELATIVA y estable al manifest (nunca absoluta)."""

    name: str
    base_url: str | None = None
    downloaded_at: str | None = None
    manifest_ref: str = Field(
        ..., description="Ruta relativa al manifest, p. ej. data/manifests/<id>.json"
    )


class BlockDescriptor(_Strict):
    """Descriptor de bloque: identidad, flags, jerarquía, cita e indexabilidad.

    NO contiene texto vigente, párrafos, versiones ni resolución temporal completa (esos viven
    en parents/histories). Propietario de `indexable`/`excluded_reason` y de los flags de filtro.
    """

    block_id: str
    parent_id: str | None = Field(
        None, description="Ref. al registro en parents/; null si no hay texto vigente."
    )
    order: int
    block_type: str | None = None
    block_title: str | None = None
    full_title: str | None = None
    semantic_role: str | None = None
    has_retrievable_body: bool = False
    is_annex: bool = False
    contains_table: bool = False
    table_text_available: bool = False
    contains_image: bool = False
    content_status: str = "present"
    is_without_content: bool = False
    temporal_status: str = Field(
        ..., description="Estado mínimo operativo: resolved|unresolved|ambiguous|..."
    )
    hierarchy: Hierarchy = Field(default_factory=Hierarchy)
    indexable: bool = False
    excluded_reason: str | None = None
    citation: Citation


class DocumentV2(_Strict):
    """Descriptor legible de una norma (`boe_legal_document_v2`)."""

    schema_version: Literal["boe_legal_document_v2"] = "boe_legal_document_v2"
    document_id: str
    source: DocumentSource
    metadata: DocumentMetadata
    analysis: Analysis
    blocks: list[BlockDescriptor]
    generation_meta: GenerationMeta


# --------------------------------------------------------------------------- #
# 2) boe_legal_history_v2 — provenance temporal (1 registro por block_id)
# --------------------------------------------------------------------------- #


class Version(_Strict):
    """Versión de un bloque (solo metadatos)."""

    source_norm_id: str | None = None
    publication_date: str | None = None
    validity_date: str | None = None
    is_current: bool = False


class ModificationNote(_Strict):
    text: str | None = None
    target_norm_id: str | None = None


class CandidateVersion(_Strict):
    version_index: int
    publication_date: str | None = None
    source_norm_id: str | None = None


class TemporalResolution(_Strict):
    """Resolución temporal completa (diagnóstico local del bloque)."""

    status: str
    selection_method: str | None = None
    index_last_update_date: str | None = None
    selected_version_index: int | None = None
    selected_publication_date: str | None = None
    selected_source_norm_id: str | None = None
    candidate_versions: list[CandidateVersion] = Field(default_factory=list)
    max_publication_date: str | None = None
    warnings: list[str] = Field(default_factory=list)


class BlockHistory(_Strict):
    """Provenance temporal de un bloque (existe para CADA block_id, incl. monoversión)."""

    block_id: str
    versions: list[Version] = Field(default_factory=list)
    modification_notes: list[ModificationNote] = Field(default_factory=list)
    temporal_resolution: TemporalResolution
    temporal_quarantined: bool = False
    index_title: str | None = None
    index_url: str | None = None
    index_last_update_date: str | None = None
    index_last_update_date_raw: str | None = None
    # Los warnings temporales viven en `temporal_resolution.warnings` (no se duplican aquí).


class HistoryV2(_Strict):
    """Historial temporal de una norma (`boe_legal_history_v2`)."""

    schema_version: Literal["boe_legal_history_v2"] = "boe_legal_history_v2"
    document_id: str
    blocks: list[BlockHistory]
    generation_meta: GenerationMeta


# --------------------------------------------------------------------------- #
# 3) boe_legal_parents_v2 — propietario único del texto vigente
# --------------------------------------------------------------------------- #


class Paragraph(_Strict):
    order: int
    css_class: str = Field(..., alias="class")
    text: str

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class CurrentVersion(_Strict):
    """Resumen ligero de la versión vigente seleccionada (espejo de history.is_current)."""

    source_norm_id: str | None = None
    publication_date: str | None = None
    validity_date: str | None = None


class ParentRecord(_Strict):
    """Bloque jurídico padre: dueño del texto vigente y los párrafos. SIN `indexable`."""

    parent_id: str
    document_id: str
    block_id: str
    order: int
    block_type: str | None = None
    title: str | None = None
    full_title: str | None = None
    semantic_role: str | None = None
    text: str
    paragraphs: list[Paragraph] = Field(default_factory=list)
    hierarchy: Hierarchy = Field(default_factory=Hierarchy)
    citation: Citation
    current_version: CurrentVersion
    is_annex: bool = False
    contains_table: bool = False
    table_text_available: bool = False
    contains_image: bool = False
    content_status: str = "present"
    is_without_content: bool = False
    # `modification_notes` NO viven aquí: son propiedad de HistoryV2 (se hidratan por join).


class ParentsV2(_Strict):
    """Almacén de bloques padre de una norma (`boe_legal_parents_v2`)."""

    schema_version: Literal["boe_legal_parents_v2"] = "boe_legal_parents_v2"
    document_id: str
    parents: list[ParentRecord]
    generation_meta: GenerationMeta


# --------------------------------------------------------------------------- #
# 4) boe_legal_chunks_v2 — child chunks vector-ready (payload mínimo)
# --------------------------------------------------------------------------- #


class ChunkPosition(_Strict):
    index: int
    count_for_parent: int


class ChunkFilters(_Strict):
    """Proyección compacta de flags para filtrado en retrieval."""

    rank_code: str | None = None
    scope_code: str | None = None
    subject_codes: list[str] = Field(default_factory=list)
    semantic_role: str | None = None
    without_content: bool = False
    annex: bool = False
    table: bool = False
    image: bool = False


class Chunk(_Strict):
    """Child chunk mínimo destinado al índice vectorial (sin parent_text ni metadatos doc.)."""

    chunk_id: str
    parent_id: str
    document_id: str
    block_id: str
    position: ChunkPosition
    text: str
    retrieval_text: str
    citation: Citation
    filters: ChunkFilters


class ChunkingStrategy(_Strict):
    name: str
    max_chars: int
    overlap_paragraphs: int
    split_unit: str
    parent_unit: str


class SourceRefs(_Strict):
    """Referencias relativas a los artefactos fuente del flujo normal (sin history)."""

    document: str
    parents: str


class ChunksV2(_Strict):
    """Documento de chunks vector-ready de una norma (`boe_legal_chunks_v2`)."""

    schema_version: Literal["boe_legal_chunks_v2"] = "boe_legal_chunks_v2"
    document_id: str
    source_refs: SourceRefs
    chunking_strategy: ChunkingStrategy
    chunks: list[Chunk]
    generation_meta: GenerationMeta


# Registro de contratos raíz: nombre de fichero de schema → modelo.
ROOT_MODELS: dict[str, type[BaseModel]] = {
    "boe_legal_document_v2": DocumentV2,
    "boe_legal_history_v2": HistoryV2,
    "boe_legal_parents_v2": ParentsV2,
    "boe_legal_chunks_v2": ChunksV2,
}
