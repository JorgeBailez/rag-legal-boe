"""Chunking jurídico parent-child por bloque BOE (contrato `boe_legal_chunks_v2`).

Consume el **descriptor** (`boe_legal_document_v2`) y el **parent store**
(`boe_legal_parents_v2`) y produce child chunks **vector-ready mínimos**: identidad, `text`,
`retrieval_text`, cita y filtros compactos. **No** copia el texto del padre ni metadatos
documentales ni `subjects` completos (el texto vigente vive solo en parents; las materias solo en
el document). El troceado es por párrafos (no por cortes de caracteres) y no altera el texto legal.

`retrieval_text` se compone de short_title + jerarquía + full_title/block_title + texto del child
(separador inteligente, sin `..`, con dedup de cabecera).
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from src.contracts.models import ChunksV2

CHUNKS_SCHEMA_VERSION = "boe_legal_chunks_v2"
STRATEGY_NAME = "legal_parent_child_paragraphs"
GENERATOR = "src.preprocessing.chunker"
DEFAULT_MAX_CHARS = 1800
DEFAULT_OVERLAP_PARAGRAPHS = 1


def load_json(path: Path) -> dict:
    """Carga un artefacto JSON (document v2 o parents v2)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _clean_text(value: str) -> str:
    """Normaliza espacios internos sin alterar el contenido textual."""
    return re.sub(r"\s+", " ", value or "").strip()


def _generation_meta() -> dict:
    return {"generated_at": datetime.now(UTC).isoformat(), "generator": GENERATOR}


# Signos finales tras los que NO se añade un punto adicional al unir el prefijo.
_SENTENCE_END = ".:;?!»\"'"


def _join_context(parts: list[str | None]) -> str:
    """Une partes de contexto con `. `, salvo si la previa ya acaba en un signo final."""
    out = ""
    for part in parts:
        part = (part or "").strip()
        if not part:
            continue
        if not out:
            out = part
        else:
            sep = " " if out[-1] in _SENTENCE_END else ". "
            out += sep + part
    return out


def build_chunk_retrieval_text(
    chunk_text: str,
    short_title: str | None,
    hierarchy: dict | None,
    heading: str | None,
) -> str:
    """Antepone contexto jurídico (norma + jerarquía + cabecera) sin reescribir el texto legal.

    Mantiene el comportamiento de la v1: separador inteligente (sin `..`) y dedup de cabecera si
    el texto del chunk ya empieza por ella. Insumos idénticos a la v1 ⇒ resultado byte-idéntico.
    """
    hierarchy = hierarchy or {}
    hierarchy_str = " ".join(
        part
        for part in (
            hierarchy.get("book"),
            hierarchy.get("title"),
            hierarchy.get("chapter"),
            hierarchy.get("section"),
            hierarchy.get("subsection"),
            hierarchy.get("annex"),
        )
        if part
    )
    if heading and _clean_text(chunk_text).startswith(_clean_text(heading)):
        heading = None  # la cabecera ya está al inicio del texto del chunk

    prefix = _join_context([short_title, hierarchy_str or None, heading])
    return _clean_text(_join_context([prefix, chunk_text]))


def build_chunk_filters(document: dict, descriptor: dict) -> dict:
    """Proyección compacta de flags de filtrado (códigos de materia, rango, ámbito, semántica)."""
    metadata = document.get("metadata", {})
    subjects = (document.get("analysis", {}) or {}).get("subjects", [])
    subject_codes = [s.get("code") for s in subjects if s.get("code")]
    return {
        "rank_code": (metadata.get("rank") or {}).get("code"),
        "scope_code": (metadata.get("scope") or {}).get("code"),
        "subject_codes": subject_codes,
        "semantic_role": descriptor.get("semantic_role"),
        "without_content": descriptor.get("is_without_content", False),
        "annex": descriptor.get("is_annex", False),
        "table": descriptor.get("contains_table", False),
        "image": descriptor.get("contains_image", False),
    }


def _group_paragraphs(
    paragraphs: list[str], max_chars: int, overlap_paragraphs: int
) -> list[list[str]]:
    """Agrupa párrafos en orden hasta `max_chars`, con overlap al dividir."""
    joined_len = len("\n".join(paragraphs))
    if joined_len <= max_chars or len(paragraphs) <= 1:
        return [paragraphs]

    groups: list[list[str]] = []
    current: list[str] = []
    for para in paragraphs:
        addition = len(para) + (1 if current else 0)
        current_len = len("\n".join(current))
        if current and current_len + addition > max_chars:
            groups.append(current)
            overlap = current[-overlap_paragraphs:] if overlap_paragraphs else []
            current = list(overlap)
        current.append(para)
    if current:
        groups.append(current)
    return groups


def chunk_block(
    document: dict,
    descriptor: dict,
    parent: dict,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_paragraphs: int = DEFAULT_OVERLAP_PARAGRAPHS,
) -> list[dict]:
    """Genera los chunks v2 de un bloque indexable a partir de su descriptor + parent."""
    if not descriptor.get("indexable"):
        return []

    paragraphs = [p["text"] for p in parent.get("paragraphs", []) if p.get("text")]
    if not paragraphs:
        return []

    document_id = document.get("document_id")
    block_id = descriptor.get("block_id")
    parent_id = descriptor.get("parent_id")
    short_title = (document.get("metadata") or {}).get("short_title")
    hierarchy = descriptor.get("hierarchy")
    heading = descriptor.get("full_title") or descriptor.get("block_title")
    citation = descriptor.get("citation") or {}
    filters = build_chunk_filters(document, descriptor)

    groups = _group_paragraphs(paragraphs, max_chars, overlap_paragraphs)
    chunks: list[dict] = []
    for index, group in enumerate(groups, start=1):
        text = "\n".join(group)
        chunks.append(
            {
                "chunk_id": f"{document_id}__{block_id}__c{index:03d}",
                "parent_id": parent_id,
                "document_id": document_id,
                "block_id": block_id,
                "position": {"index": index, "count_for_parent": len(groups)},
                "text": text,
                "retrieval_text": build_chunk_retrieval_text(text, short_title, hierarchy, heading),
                "citation": {"label": citation.get("label"), "url": citation.get("url")},
                "filters": filters,
            }
        )
    return chunks


def create_chunks(
    document: dict,
    parents: dict,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_paragraphs: int = DEFAULT_OVERLAP_PARAGRAPHS,
) -> dict:
    """Construye el documento `boe_legal_chunks_v2` (payload de dominio, sin diagnósticos).

    Los conteos/oversized/missing_parents son **diagnóstico** y los calcula la auditoría hacia
    `reports/` (no se persisten en el payload). Ver `chunking_diagnostics`.
    """
    document_id = document.get("document_id")
    descriptors = document.get("blocks", [])
    parents_by_id = {p["parent_id"]: p for p in parents.get("parents", [])}

    chunks: list[dict] = []
    for descriptor in descriptors:
        if not descriptor.get("indexable"):
            continue
        parent = parents_by_id.get(descriptor.get("parent_id"))
        if parent is None:
            continue
        chunks.extend(chunk_block(document, descriptor, parent, max_chars, overlap_paragraphs))

    return {
        "schema_version": CHUNKS_SCHEMA_VERSION,
        "document_id": document_id,
        "source_refs": {
            "document": f"data/processed/documents/{document_id}.json",
            "parents": f"data/processed/parents/{document_id}.json",
        },
        "chunking_strategy": {
            "name": STRATEGY_NAME,
            "max_chars": max_chars,
            "overlap_paragraphs": overlap_paragraphs,
            "split_unit": "paragraphs",
            "parent_unit": "boe_block",
        },
        "chunks": chunks,
        "generation_meta": _generation_meta(),
    }


def chunking_diagnostics(
    document: dict, parents: dict, chunks_doc: dict, max_chars: int = DEFAULT_MAX_CHARS
) -> dict:
    """Diagnóstico de chunking (para `reports/`, no para el payload de dominio)."""
    descriptors = document.get("blocks", [])
    parents_ids = {p["parent_id"] for p in parents.get("parents", [])}
    indexable = [b for b in descriptors if b.get("indexable")]
    chunks = chunks_doc.get("chunks", [])
    chunked_blocks = {c["block_id"] for c in chunks}
    missing_parents = [b["block_id"] for b in indexable if b.get("parent_id") not in parents_ids]
    blocks_without_chunks = [
        b["block_id"]
        for b in indexable
        if b.get("parent_id") in parents_ids and b["block_id"] not in chunked_blocks
    ]
    oversized = [c["chunk_id"] for c in chunks if len(c.get("text", "")) > max_chars]
    return {
        "source_blocks_count": len(descriptors),
        "indexable_blocks_count": len(indexable),
        "chunks_count": len(chunks),
        "blocks_without_chunks": blocks_without_chunks,
        "missing_parents": missing_parents,
        "oversized_chunks": oversized,
    }


def save_chunks(chunks_document: dict, output_dir: Path) -> Path:
    """Valida contra el contrato `ChunksV2` y persiste como `{document_id}.json`."""
    ChunksV2.model_validate(chunks_document)  # fail-fast antes de escribir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{chunks_document['document_id']}.json"
    out_path.write_text(
        json.dumps(chunks_document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path
