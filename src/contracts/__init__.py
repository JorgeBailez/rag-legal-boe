"""Contratos de datos v2 del pipeline (modelos Pydantic = fuente única).

Define los cuatro contratos persistidos por norma y sus sub-modelos:

- `boe_legal_document_v2`  → `DocumentV2`   (descriptor legible; sin texto pesado)
- `boe_legal_history_v2`   → `HistoryV2`    (provenance temporal; un registro por block_id)
- `boe_legal_parents_v2`   → `ParentsV2`    (propietario único del texto vigente)
- `boe_legal_chunks_v2`    → `ChunksV2`     (child chunks vector-ready, payload mínimo)

Los modelos son la fuente de verdad: `export_schemas.py` deriva `schemas/*.json`
(JSON Schema Draft 2020-12) de forma determinista. Toda raíz usa `extra="forbid"` y
`schema_version: Literal[...]` para detectar contratos ajenos o derivas.
"""

from src.contracts.models import (
    ROOT_MODELS,
    ChunksV2,
    DocumentV2,
    HistoryV2,
    ParentsV2,
)

__all__ = ["ChunksV2", "DocumentV2", "HistoryV2", "ParentsV2", "ROOT_MODELS"]
