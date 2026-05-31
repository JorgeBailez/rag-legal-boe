"""Chunking jurídico parent-child por bloque BOE (v0).

Convierte un documento `boe_legal_document_v1` en chunks recuperables
(`boe_legal_chunks_v1`) preservando el bloque jurídico como documento padre
(`parent_id` + `parent_text`). El chunking es por párrafos (no por cortes arbitrarios de
caracteres) y no altera el texto legal.

No hace red, no genera embeddings ni índices: solo prepara unidades recuperables.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

CHUNKS_SCHEMA_VERSION = "boe_legal_chunks_v1"
STRATEGY_NAME = "legal_parent_child_v1"
DEFAULT_MAX_CHARS = 1800
DEFAULT_OVERLAP_PARAGRAPHS = 1


def load_processed_document(path: Path) -> dict:
    """Carga el documento procesado (`boe_legal_document_v1`) desde JSON."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _clean_text(value: str) -> str:
    """Normaliza espacios internos sin alterar el contenido textual."""
    return re.sub(r"\s+", " ", value or "").strip()


def build_chunk_metadata(document: dict, block: dict) -> dict:
    """Metadatos compartidos por todos los chunks de un bloque."""
    metadata = document.get("metadata", {})
    analysis = document.get("analysis", {})
    retrieval = block.get("retrieval", {})
    latest = block.get("latest_version") or {}

    return {
        "schema_version": document.get("schema_version"),
        "source": document.get("source", {}).get("name"),
        "legal_status_notice": metadata.get("legal_status_notice"),
        "norm_title": metadata.get("title"),
        "short_title": metadata.get("short_title"),
        "document_id": document.get("document_id"),
        "block_id": block.get("block_id"),
        "block_type": block.get("block_type"),
        "block_title": block.get("block_title"),
        "full_title": block.get("full_title"),
        "citation_label": retrieval.get("citation_label"),
        "source_url": retrieval.get("source_url"),
        "hierarchy": block.get("hierarchy"),
        "publication_date": latest.get("publication_date"),
        "validity_date": latest.get("validity_date"),
        "source_norm_id": latest.get("source_norm_id"),
        "rank": metadata.get("rank"),
        "scope": metadata.get("scope"),
        "subjects": analysis.get("subjects", []),
        "is_preamble": block.get("block_type") == "preambulo",
        "semantic_role": block.get("semantic_role"),
        "has_retrievable_body": block.get("has_retrievable_body"),
        "is_annex": block.get("is_annex", False),
        "contains_table": block.get("contains_table", False),
        "table_text_available": block.get("table_text_available", False),
        "contains_image": block.get("contains_image", False),
        "content_status": block.get("content_status", "present"),
        "is_without_content": block.get("is_without_content", False),
    }


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


def build_chunk_retrieval_text(chunk_text: str, metadata: dict) -> str:
    """Antepone contexto jurídico al texto del chunk (sin reescribir el texto legal).

    No genera `..` (separador inteligente) y deduplica la cabecera: si el texto del chunk
    ya empieza por `full_title`/`block_title`, no se repite en el prefijo (`c001`).
    """
    hierarchy = metadata.get("hierarchy") or {}
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
    heading = metadata.get("full_title") or metadata.get("block_title")
    if heading and _clean_text(chunk_text).startswith(_clean_text(heading)):
        heading = None  # la cabecera ya está al inicio del texto del chunk

    prefix = _join_context([metadata.get("short_title"), hierarchy_str or None, heading])
    # `_clean_text` colapsa el `\n` del chunk (retrieval_text es de una sola línea).
    return _clean_text(_join_context([prefix, chunk_text]))


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
        # Longitud que añadiría este párrafo (con su salto de línea si ya hay contenido).
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
    block: dict,
    document: dict,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_paragraphs: int = DEFAULT_OVERLAP_PARAGRAPHS,
) -> list[dict]:
    """Genera los chunks de un bloque indexable preservando la relación parent-child."""
    if not block.get("retrieval", {}).get("indexable"):
        return []

    latest = block.get("latest_version") or {}
    paragraphs = [p["text"] for p in latest.get("paragraphs", []) if p.get("text")]
    if not paragraphs:
        return []

    document_id = document.get("document_id")
    block_id = block.get("block_id")
    parent_id = block.get("parent_id")
    parent_text = latest.get("text", "")
    metadata = build_chunk_metadata(document, block)

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
                "chunk_index": index,
                "chunk_count_for_parent": len(groups),
                "chunking_strategy": STRATEGY_NAME,
                "text": text,
                "retrieval_text": build_chunk_retrieval_text(text, metadata),
                "parent_text": parent_text,
                "metadata": metadata,
            }
        )
    return chunks


def create_chunks(
    document: dict,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_paragraphs: int = DEFAULT_OVERLAP_PARAGRAPHS,
) -> dict:
    """Construye el documento de chunks `boe_legal_chunks_v1` a partir del documento fuente."""
    document_id = document.get("document_id")
    blocks = document.get("blocks", [])

    indexable_blocks = [b for b in blocks if b.get("retrieval", {}).get("indexable")]
    chunks: list[dict] = []
    blocks_without_chunks: list[str] = []
    oversized_chunks: list[str] = []

    for block in indexable_blocks:
        block_chunks = chunk_block(block, document, max_chars, overlap_paragraphs)
        if not block_chunks:
            blocks_without_chunks.append(block.get("block_id"))
            continue
        for chunk in block_chunks:
            if len(chunk["text"]) > max_chars:
                oversized_chunks.append(chunk["chunk_id"])
        chunks.extend(block_chunks)

    return {
        "schema_version": CHUNKS_SCHEMA_VERSION,
        "document_id": document_id,
        "source_document_path": f"data/processed/documents/{document_id}.json",
        "chunking_strategy": {
            "name": STRATEGY_NAME,
            "max_chars": max_chars,
            "overlap_paragraphs": overlap_paragraphs,
            "split_unit": "paragraphs",
            "parent_unit": "boe_block",
        },
        "chunks": chunks,
        "quality_checks": {
            "source_blocks_count": len(blocks),
            "indexable_blocks_count": len(indexable_blocks),
            "chunks_count": len(chunks),
            "blocks_without_chunks": blocks_without_chunks,
            "oversized_chunks": oversized_chunks,
            "warnings": [],
        },
    }


def save_chunks(chunks_document: dict, output_dir: Path) -> Path:
    """Persiste el documento de chunks como `{document_id}.json`."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{chunks_document['document_id']}.json"
    out_path.write_text(
        json.dumps(chunks_document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path
