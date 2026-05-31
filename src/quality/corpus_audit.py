"""Auditoría de calidad del corpus (parser + chunker), de solo lectura.

Contrasta los artefactos generados (`boe_legal_document_v1`, `boe_legal_chunks_v1`) contra
el contrato esperado y produce hallazgos clasificados + métricas. No modifica nada del
pipeline: todas las funciones reciben dicts ya cargados (o rutas de raw para trazabilidad)
y devuelven estructuras de datos.

Clasificación: Correcto · Aceptable MVP · Revisar antes de embeddings · Mejora posterior.
Severidad: ERROR (viola el contrato) · WARN (incompleto/dudoso) · INFO (observación).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import re
from pathlib import Path

from lxml import etree

from src.boe.parser import (
    EXCLUDED_TYPES,
    SCHEMA_VERSION,
    clean_text,
    heading_has_retrievable_body,
    load_xml,
    resolve_current_version,
    validate_response,
)
from src.preprocessing.chunker import CHUNKS_SCHEMA_VERSION

# Universo esperado de tipos de bloque (observado en el corpus); otros se reportan como nuevos.
EXPECTED_BLOCK_TYPES = {"nota_inicial", "preambulo", "encabezado", "precepto", "firma"}
# Bloques editoriales (no normativos): no deberían ser indexables.
EDITORIAL_TYPES = {"nota_inicial"}

# Clases de encabezado que el parser SÍ usa para construir la jerarquía (todas inequívocas).
HANDLED_HEADING_CLASSES = {
    "libro_num",
    "titulo_num",
    "capitulo_num",
    "seccion",
    "subseccion",
    "anexo_num",
}
# Clases de encabezado/rótulo estructural completas (para detectar cuerpo sustantivo).
STRUCTURAL_HEADING_CLASSES = {
    "libro_num",
    "libro_tit",
    "libro",
    "titulo_num",
    "titulo_tit",
    "titulo",
    "capitulo_num",
    "capitulo_tit",
    "capitulo",
    "seccion",
    "subseccion",
    "anexo_num",
    "anexo_tit",
    "anexo",
}
# Clases-rótulo singulares: el parser les da `full_title` pero NO las usa para jerarquía
# (limitación menor aceptada, no bloqueante).
SINGULAR_LABEL_CLASSES = {"libro", "titulo", "capitulo", "anexo"}

ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ISO_DATETIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
XML_TAG = re.compile(r"<[a-zA-Z/]")
CHUNK_ID = re.compile(r"^BOE-[A-Z]-\d{4}-\d+__.+__c\d{3}$")
# `..` artificial: exactamente dos puntos, no parte de una elipsis legal `...`.
_ARTIFICIAL_DOUBLE_DOT = re.compile(r"(?<!\.)\.\.(?!\.)")

REQUIRED_DOC_KEYS = (
    "schema_version",
    "document_id",
    "source",
    "metadata",
    "analysis",
    "blocks",
    "quality_checks",
)
REQUIRED_CHUNK_META = (
    "schema_version",
    "source",
    "legal_status_notice",
    "norm_title",
    "short_title",
    "document_id",
    "block_id",
    "block_type",
    "citation_label",
    "source_url",
    "hierarchy",
    "rank",
    "scope",
    "subjects",
    "is_preamble",
)


def finding(check, severity, classification, document_id, ref, message, evidence=None):
    """Construye un hallazgo estructurado."""
    return {
        "check": check,
        "severity": severity,
        "classification": classification,
        "document_id": document_id,
        "ref": ref,
        "message": message,
        "evidence": evidence,
    }


# --------------------------------------------------------------------------- #
# Integridad estructural — documento
# --------------------------------------------------------------------------- #


def check_document(doc: dict, processing_date: str | None = None) -> list[dict]:
    """Verifica el contrato del documento `boe_legal_document_v1`."""
    out: list[dict] = []
    did = doc.get("document_id")
    if processing_date is None:
        processing_date = datetime.date.today().isoformat()

    if doc.get("schema_version") != SCHEMA_VERSION:
        out.append(
            finding(
                "doc.schema",
                "ERROR",
                "Revisar antes de embeddings",
                did,
                None,
                f"schema_version inesperado: {doc.get('schema_version')!r}",
            )
        )
    for k in REQUIRED_DOC_KEYS:
        if k not in doc:
            out.append(
                finding(
                    "doc.keys",
                    "ERROR",
                    "Revisar antes de embeddings",
                    did,
                    None,
                    f"falta clave de nivel superior: {k}",
                )
            )

    meta = doc.get("metadata", {})
    if meta.get("identifier") and meta["identifier"] != did:
        out.append(
            finding(
                "doc.identity",
                "ERROR",
                "Revisar antes de embeddings",
                did,
                None,
                f"document_id != metadata.identifier ({meta['identifier']})",
            )
        )
    for k in ("publication_date", "document_date", "effective_date"):
        v = meta.get(k)
        if v and not ISO_DATE.match(v):
            out.append(
                finding(
                    "doc.date",
                    "WARN",
                    "Revisar antes de embeddings",
                    did,
                    k,
                    f"fecha no ISO: {v!r}",
                )
            )
    lud = meta.get("last_update_datetime")
    if lud and not ISO_DATETIME.match(lud):
        out.append(
            finding(
                "doc.date",
                "WARN",
                "Revisar antes de embeddings",
                did,
                "last_update_datetime",
                f"datetime no ISO: {lud!r}",
            )
        )

    qc = doc.get("quality_checks", {})
    ib, tb = qc.get("index_blocks_count"), qc.get("text_blocks_count")
    if ib != tb:
        out.append(
            finding(
                "doc.counts",
                "ERROR",
                "Revisar antes de embeddings",
                did,
                None,
                f"index != text blocks ({ib}/{tb})",
            )
        )
    if qc.get("unmatched_index_blocks") or qc.get("unmatched_text_blocks"):
        out.append(
            finding(
                "doc.counts",
                "ERROR",
                "Revisar antes de embeddings",
                did,
                None,
                "hay bloques sin emparejar índice/texto",
            )
        )

    out.extend(_check_blocks(doc, did, processing_date))
    return out


def _check_temporal(b: dict, did: str, processing_date: str) -> list[dict]:
    """Verifica, de forma independiente, la vigencia temporal del bloque.

    Recalcula la resolución desde `versions[]` + `index_last_update_date` (no confía en lo que
    guardó el parser): cuarentena → ERROR; `latest_version` distinto de la versión vigente por
    índice → ERROR; entrada en vigor futura → WARN informativo.
    """
    out: list[dict] = []
    versions = b.get("versions") or []
    if not versions:
        return out
    bid = b.get("block_id")
    res = resolve_current_version(versions, b.get("index_last_update_date"))
    status = res["status"]
    lv = b.get("latest_version") or {}
    indexable = (b.get("retrieval") or {}).get("indexable")
    if status != "resolved":
        out.append(
            finding(
                "block.temporal_quarantine",
                "ERROR",
                "Revisar antes de embeddings",
                did,
                bid,
                f"versión vigente no resoluble por índice ({status})",
                evidence=status,
            )
        )
        # El parser debe haber puesto el bloque en cuarentena (sin latest_version, no indexable).
        if lv or indexable:
            out.append(
                finding(
                    "block.temporal_inconsistent",
                    "ERROR",
                    "Revisar antes de embeddings",
                    did,
                    bid,
                    "estado no-resuelto pero el bloque no está en cuarentena",
                )
            )
        return out

    if (
        lv.get("publication_date") != res["selected_publication_date"]
        or lv.get("source_norm_id") != res["selected_source_norm_id"]
    ):
        out.append(
            finding(
                "block.temporal_mismatch",
                "ERROR",
                "Revisar antes de embeddings",
                did,
                bid,
                "latest_version no es la versión vigente por índice",
                evidence=f"{lv.get('publication_date')} != {res['selected_publication_date']}",
            )
        )
    vdate = lv.get("validity_date")
    if vdate and vdate > processing_date:
        out.append(
            finding(
                "block.future_effective",
                "WARN",
                "Aceptable MVP",
                did,
                bid,
                f"versión vigente con entrada en vigor futura ({vdate} > {processing_date})",
            )
        )
    return out


def _check_blocks(doc: dict, did: str, processing_date: str) -> list[dict]:
    out: list[dict] = []
    html_url = (doc.get("metadata") or {}).get("html_url")
    for b in doc.get("blocks", []):
        out.extend(_check_temporal(b, did, processing_date))
        bid = b.get("block_id")
        bt = b.get("block_type")
        if bt not in EXPECTED_BLOCK_TYPES:
            out.append(
                finding(
                    "block.type",
                    "WARN",
                    "Revisar antes de embeddings",
                    did,
                    bid,
                    f"block_type no esperado: {bt!r}",
                )
            )
        if b.get("parent_id") != f"{did}__{bid}":
            out.append(
                finding(
                    "block.parent_id",
                    "ERROR",
                    "Revisar antes de embeddings",
                    did,
                    bid,
                    "parent_id no sigue el patrón {doc}__{block_id}",
                )
            )

        versions = b.get("versions") or []
        latest_flags = [v for v in versions if v.get("is_latest")]
        if versions and len(latest_flags) != 1:
            out.append(
                finding(
                    "block.is_latest",
                    "ERROR",
                    "Revisar antes de embeddings",
                    did,
                    bid,
                    f"se esperaba exactamente un is_latest, hay {len(latest_flags)}",
                )
            )
        for v in versions:
            if "text" in v or "paragraphs" in v:
                out.append(
                    finding(
                        "block.versions_metadata",
                        "ERROR",
                        "Mejora posterior",
                        did,
                        bid,
                        "versions[] debería contener solo metadatos (lleva text/paragraphs)",
                    )
                )
                break

        retr = b.get("retrieval") or {}
        lv0 = b.get("latest_version") or {}
        paragraphs = lv0.get("paragraphs", [])
        has_body = heading_has_retrievable_body(paragraphs)
        expected_indexable = has_body and bt not in EXCLUDED_TYPES and bool(lv0.get("text"))
        actual_indexable = retr.get("indexable")
        if actual_indexable != expected_indexable:
            out.append(
                finding(
                    "block.indexable",
                    "WARN",
                    "Aceptable MVP",
                    did,
                    bid,
                    f"indexable={actual_indexable} != regla ({expected_indexable})",
                )
            )
        # Encabezado con cuerpo sustantivo que NO se indexa → contenido fuera de retrieval.
        if bt == "encabezado" and has_body and not actual_indexable:
            out.append(
                finding(
                    "block.heading_body_not_indexed",
                    "ERROR",
                    "Revisar antes de embeddings",
                    did,
                    bid,
                    "encabezado con cuerpo sustantivo no indexado",
                )
            )
        # Encabezado puramente estructural (rótulo) marcado como indexable.
        if bt == "encabezado" and not has_body and actual_indexable:
            out.append(
                finding(
                    "block.structural_indexed",
                    "ERROR",
                    "Revisar antes de embeddings",
                    did,
                    bid,
                    "rótulo estructural puro marcado como indexable",
                )
            )
        out.extend(_check_semantics(b, did, bid, paragraphs, has_body))
        if html_url and retr.get("source_url") != f"{html_url}#{bid}":
            out.append(
                finding(
                    "block.source_url",
                    "ERROR",
                    "Revisar antes de embeddings",
                    did,
                    bid,
                    "source_url != html_url#block_id",
                )
            )

        lv = b.get("latest_version") or {}
        if XML_TAG.search(lv.get("text", "") or ""):
            out.append(
                finding(
                    "block.text_tags",
                    "ERROR",
                    "Revisar antes de embeddings",
                    did,
                    bid,
                    "el texto del bloque contiene etiquetas tipo XML/HTML",
                )
            )
        for n in lv.get("modification_notes", []):
            nt = (n.get("text") or "").strip()
            if nt and nt in (lv.get("text") or ""):
                out.append(
                    finding(
                        "block.note_leak",
                        "ERROR",
                        "Revisar antes de embeddings",
                        did,
                        bid,
                        "una nota de modificación aparece en el texto normativo",
                        evidence=nt[:80],
                    )
                )
    return out


def _check_semantics(
    b: dict, did: str, bid: str, paragraphs: list[dict], has_body: bool
) -> list[dict]:
    """Coherencia de `semantic_role`, `is_annex` y `hierarchy.annex`."""
    out: list[dict] = []
    role = b.get("semantic_role")
    is_annex = b.get("is_annex")
    bt = b.get("block_type")
    hierarchy = b.get("hierarchy") or {}

    if bt == "encabezado":
        if is_annex and has_body and role != "annex":
            out.append(
                finding(
                    "block.semantic_role",
                    "WARN",
                    "Mejora posterior",
                    did,
                    bid,
                    f"is_annex pero semantic_role={role!r}",
                )
            )
        if has_body and role == "structural_heading":
            out.append(
                finding(
                    "block.semantic_role",
                    "WARN",
                    "Mejora posterior",
                    did,
                    bid,
                    "tiene cuerpo pero role=structural_heading",
                )
            )

    # annex arrastrada: annex junto a algún nivel del cuerpo (deberían ser excluyentes).
    body_levels = any(
        hierarchy.get(k) for k in ("book", "title", "chapter", "section", "subsection")
    )
    if hierarchy.get("annex") and body_levels:
        out.append(
            finding(
                "block.annex_dragged",
                "WARN",
                "Revisar antes de embeddings",
                did,
                bid,
                "hierarchy.annex coexiste con niveles del cuerpo",
            )
        )

    # anexo local (singular) sin contexto de annex asignado.
    if is_annex and role == "annex" and not hierarchy.get("annex"):
        out.append(
            finding(
                "block.annex_local_context_missing",
                "WARN",
                "Mejora posterior",
                did,
                bid,
                "is_annex sin hierarchy.annex",
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Integridad estructural — chunks
# --------------------------------------------------------------------------- #


def check_chunks(chunks_doc: dict, doc: dict) -> list[dict]:
    """Verifica el contrato de los chunks y su coherencia con el documento."""
    out: list[dict] = []
    did = chunks_doc.get("document_id")
    if chunks_doc.get("schema_version") != CHUNKS_SCHEMA_VERSION:
        out.append(
            finding(
                "chunks.schema",
                "ERROR",
                "Revisar antes de embeddings",
                did,
                None,
                f"schema_version inesperado: {chunks_doc.get('schema_version')!r}",
            )
        )

    blocks = {b["block_id"]: b for b in doc.get("blocks", [])}
    indexable_ids = {
        bid for bid, b in blocks.items() if (b.get("retrieval") or {}).get("indexable")
    }

    seen_ids: set[str] = set()
    by_parent: dict[str, list[dict]] = {}
    for ch in chunks_doc.get("chunks", []):
        cid = ch.get("chunk_id")
        bid = ch.get("block_id")
        if not CHUNK_ID.match(cid or ""):
            out.append(
                finding(
                    "chunk.id_format",
                    "ERROR",
                    "Revisar antes de embeddings",
                    did,
                    cid,
                    "chunk_id no sigue {doc}__{block_id}__cNNN",
                )
            )
        if cid in seen_ids:
            out.append(
                finding(
                    "chunk.id_unique",
                    "ERROR",
                    "Revisar antes de embeddings",
                    did,
                    cid,
                    "chunk_id duplicado",
                )
            )
        seen_ids.add(cid)

        b = blocks.get(bid)
        if b is None:
            out.append(
                finding(
                    "chunk.orphan",
                    "ERROR",
                    "Revisar antes de embeddings",
                    did,
                    cid,
                    f"chunk de un block_id inexistente: {bid}",
                )
            )
            continue
        if not (b.get("retrieval") or {}).get("indexable"):
            out.append(
                finding(
                    "chunk.non_indexable",
                    "ERROR",
                    "Revisar antes de embeddings",
                    did,
                    cid,
                    f"chunk de un bloque no indexable ({b.get('block_type')})",
                )
            )
        if ch.get("parent_id") != f"{did}__{bid}":
            out.append(
                finding(
                    "chunk.parent_id",
                    "ERROR",
                    "Revisar antes de embeddings",
                    did,
                    cid,
                    "parent_id no coincide",
                )
            )
        if ch.get("parent_text") != (b.get("latest_version") or {}).get("text", ""):
            out.append(
                finding(
                    "chunk.parent_text",
                    "ERROR",
                    "Revisar antes de embeddings",
                    did,
                    cid,
                    "parent_text != texto del bloque padre",
                )
            )
        # Vigencia: el chunk no debe proceder de una versión histórica.
        res = resolve_current_version(b.get("versions") or [], b.get("index_last_update_date"))
        if res["status"] == "resolved":
            cpub = (ch.get("metadata") or {}).get("publication_date")
            if cpub and cpub != res["selected_publication_date"]:
                out.append(
                    finding(
                        "chunk.temporal_stale",
                        "ERROR",
                        "Revisar antes de embeddings",
                        did,
                        cid,
                        "chunk construido desde una versión no vigente",
                        evidence=f"{cpub} != {res['selected_publication_date']}",
                    )
                )
        if XML_TAG.search(ch.get("text", "") or ""):
            out.append(
                finding(
                    "chunk.text_tags",
                    "ERROR",
                    "Revisar antes de embeddings",
                    did,
                    cid,
                    "el texto del chunk contiene etiquetas",
                )
            )
        meta = ch.get("metadata", {})
        for mk in REQUIRED_CHUNK_META:
            if mk not in meta:
                out.append(
                    finding(
                        "chunk.meta_keys",
                        "ERROR",
                        "Revisar antes de embeddings",
                        did,
                        cid,
                        f"falta metadata.{mk}",
                    )
                )
        if b.get("block_type") == "preambulo" and not meta.get("is_preamble"):
            out.append(
                finding(
                    "chunk.is_preamble",
                    "WARN",
                    "Aceptable MVP",
                    did,
                    cid,
                    "chunk de preámbulo sin is_preamble=true",
                )
            )
        # nota / redacción histórica dentro del texto del chunk
        for n in (b.get("latest_version") or {}).get("modification_notes", []):
            nt = (n.get("text") or "").strip()
            if nt and nt in (ch.get("text", "") or ""):
                out.append(
                    finding(
                        "chunk.note_leak",
                        "ERROR",
                        "Revisar antes de embeddings",
                        did,
                        cid,
                        "nota de modificación dentro del texto del chunk",
                    )
                )
        out.extend(check_retrieval_text(ch))
        by_parent.setdefault(bid, []).append(ch)

    # cobertura + secuencia
    for bid in indexable_ids - set(by_parent):
        out.append(
            finding(
                "chunk.coverage",
                "ERROR",
                "Revisar antes de embeddings",
                did,
                bid,
                "bloque indexable sin ningún chunk",
            )
        )
    for bid, chs in by_parent.items():
        chs_sorted = sorted(chs, key=lambda c: c.get("chunk_index", 0))
        idxs = [c.get("chunk_index") for c in chs_sorted]
        if idxs != list(range(1, len(chs_sorted) + 1)):
            out.append(
                finding(
                    "chunk.sequence",
                    "ERROR",
                    "Revisar antes de embeddings",
                    did,
                    bid,
                    f"chunk_index no secuencial: {idxs}",
                )
            )
        if any(c.get("chunk_count_for_parent") != len(chs_sorted) for c in chs_sorted):
            out.append(
                finding(
                    "chunk.count",
                    "ERROR",
                    "Revisar antes de embeddings",
                    did,
                    bid,
                    "chunk_count_for_parent incoherente",
                )
            )
    return out


def check_retrieval_text(chunk: dict) -> list[dict]:
    """Audita la calidad del `retrieval_text` de un chunk."""
    out: list[dict] = []
    did = chunk.get("document_id")
    cid = chunk.get("chunk_id")
    rt = chunk.get("retrieval_text", "") or ""
    meta = chunk.get("metadata", {})
    st = meta.get("short_title")
    ft = meta.get("full_title")

    # `..` artificial del prefijo: dos puntos (no elipsis `...`) que NO vienen del texto legal.
    body = clean_text(chunk.get("text", "") or "")
    if _ARTIFICIAL_DOUBLE_DOT.search(rt) and not _ARTIFICIAL_DOUBLE_DOT.search(body):
        out.append(
            finding(
                "rt.double_period",
                "WARN",
                "Aceptable MVP",
                did,
                cid,
                "retrieval_text contiene '..' artificial del prefijo de contexto",
            )
        )
    if st and not rt.startswith(st):
        out.append(
            finding(
                "rt.prefix",
                "WARN",
                "Aceptable MVP",
                did,
                cid,
                "retrieval_text no empieza por short_title",
            )
        )
    # Duplicación REAL: la cabecera aparece más de una vez en retrieval_text.
    if ft and rt.count(clean_text(ft)) > 1:
        out.append(
            finding(
                "rt.full_title_dup",
                "INFO",
                "Aceptable MVP",
                did,
                cid,
                "el full_title aparece más de una vez en retrieval_text",
            )
        )
    # jerarquía repetida dentro del retrieval_text
    hier = meta.get("hierarchy") or {}
    for label in (hier.get("title"), hier.get("chapter"), hier.get("section")):
        if label and rt.count(label) > 1:
            out.append(
                finding(
                    "rt.hierarchy_dup",
                    "INFO",
                    "Aceptable MVP",
                    did,
                    cid,
                    f"etiqueta de jerarquía repetida: {label!r}",
                )
            )
            break
    return out


# --------------------------------------------------------------------------- #
# Overlap
# --------------------------------------------------------------------------- #


def analyze_overlap(chunks_doc: dict, doc: dict) -> dict:
    """Métricas y verificación del overlap entre chunks de un mismo padre."""
    blocks = {b["block_id"]: b for b in doc.get("blocks", [])}
    by_parent: dict[str, list[dict]] = {}
    for ch in chunks_doc.get("chunks", []):
        by_parent.setdefault(ch["block_id"], []).append(ch)

    parents_split = 0
    overlap_ok = 0
    overlap_violations: list[str] = []
    order_preserved = 0
    order_violations: list[str] = []
    near_identical: list[str] = []
    dup_paragraphs = 0
    dup_chars = 0

    for bid, chs in by_parent.items():
        if len(chs) <= 1:
            continue
        parents_split += 1
        chs = sorted(chs, key=lambda c: c.get("chunk_index", 0))
        paras_per_chunk = [c["text"].split("\n") for c in chs]

        # overlap exacto de 1 párrafo en cada frontera
        boundary_ok = True
        for a, b in zip(paras_per_chunk, paras_per_chunk[1:], strict=False):
            if a and b and a[-1] == b[0]:
                dup_paragraphs += 1
                dup_chars += len(a[-1])
            else:
                boundary_ok = False
        if boundary_ok:
            overlap_ok += 1
        else:
            overlap_violations.append(bid)

        # reconstrucción: quitar el overlap y comparar con los párrafos del bloque
        merged: list[str] = list(paras_per_chunk[0])
        for nxt in paras_per_chunk[1:]:
            start = 1 if (merged and nxt and merged[-1] == nxt[0]) else 0
            merged.extend(nxt[start:])
        block_paras = [
            p["text"]
            for p in (blocks.get(bid, {}).get("latest_version") or {}).get("paragraphs", [])
        ]
        if merged == block_paras:
            order_preserved += 1
        else:
            order_violations.append(bid)

        # chunks casi idénticos (Jaccard de palabras > 0.8)
        for a, b in zip(chs, chs[1:], strict=False):
            wa, wb = set(a["text"].split()), set(b["text"].split())
            if wa and wb and len(wa & wb) / len(wa | wb) > 0.8:
                near_identical.append(f"{a['chunk_id']}~{b['chunk_id']}")

    return {
        "parents_split": parents_split,
        "overlap_boundary_ok": overlap_ok,
        "overlap_violations": overlap_violations,
        "order_preserved": order_preserved,
        "order_violations": order_violations,
        "near_identical_pairs": near_identical,
        "duplicated_paragraphs": dup_paragraphs,
        "duplicated_chars": dup_chars,
    }


# --------------------------------------------------------------------------- #
# Oversized
# --------------------------------------------------------------------------- #


def oversized_rows(chunks_doc: dict, doc: dict, max_chars: int) -> list[dict]:
    """Filas de la tabla de chunks sobredimensionados (una por chunk > max_chars)."""
    blocks = {b["block_id"]: b for b in doc.get("blocks", [])}
    rows: list[dict] = []
    for ch in chunks_doc.get("chunks", []):
        text = ch.get("text", "") or ""
        if len(text) <= max_chars:
            continue
        paras = text.split("\n")
        rows.append(
            {
                "document_id": ch.get("document_id"),
                "block_id": ch.get("block_id"),
                "block_type": (blocks.get(ch.get("block_id"), {}) or {}).get("block_type"),
                "chunk_id": ch.get("chunk_id"),
                "text_chars": len(text),
                "retrieval_text_chars": len(ch.get("retrieval_text", "") or ""),
                "words_count": len(text.split()),
                "paragraphs_count": len(paras),
                "single_paragraph_oversized": len(paras) == 1,
                "max_chars_excess": len(text) - max_chars,
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Tipos de bloque y jerarquía
# --------------------------------------------------------------------------- #


def block_type_stats(docs: dict[str, dict]) -> dict:
    """Distribución de block_type e indexabilidad por tipo en el corpus."""
    from collections import Counter

    counts: Counter = Counter()
    indexable: Counter = Counter()
    for doc in docs.values():
        for b in doc.get("blocks", []):
            bt = b.get("block_type")
            counts[bt] += 1
            if (b.get("retrieval") or {}).get("indexable"):
                indexable[bt] += 1
    return {"counts": dict(counts), "indexable_counts": dict(indexable)}


def hierarchy_stats(docs: dict[str, dict]) -> dict:
    """Qué niveles de jerarquía conserva el parser y cuáles pierde (solo `encabezado`)."""
    from collections import Counter

    classes: Counter = Counter()
    norms_with_unhandled: dict[str, list[str]] = {}
    norms_with_singular_labels: dict[str, list[str]] = {}
    headings_without_full_title = 0
    for nid, doc in docs.items():
        unhandled_here: set[str] = set()
        singular_here: set[str] = set()
        for b in doc.get("blocks", []):
            if b.get("block_type") != "encabezado":
                continue
            lv = b.get("latest_version") or {}
            present = {p["class"] for p in lv.get("paragraphs", [])}
            heading_present = present & STRUCTURAL_HEADING_CLASSES
            for c in heading_present:
                classes[c] += 1
            singular_here |= heading_present & SINGULAR_LABEL_CLASSES
            # Bloqueante: un encabezado con clases de rótulo pero SIN full_title (no reconocido).
            if heading_present and b.get("full_title") is None:
                headings_without_full_title += 1
                unhandled_here |= heading_present - HANDLED_HEADING_CLASSES - SINGULAR_LABEL_CLASSES
        if unhandled_here:
            norms_with_unhandled[nid] = sorted(unhandled_here)
        if singular_here:
            norms_with_singular_labels[nid] = sorted(singular_here)
    return {
        "handled_classes": sorted(HANDLED_HEADING_CLASSES),
        "heading_class_counts": dict(classes),
        "norms_with_unhandled_hierarchy": norms_with_unhandled,
        "norms_with_singular_labels": norms_with_singular_labels,
        "headings_without_full_title": headings_without_full_title,
    }


# --------------------------------------------------------------------------- #
# Eficiencia y redundancia
# --------------------------------------------------------------------------- #


def efficiency_metrics(chunks_doc: dict, overlap: dict) -> dict:
    """Tamaños y redundancia del JSON de chunks de una norma."""
    chunks = chunks_doc.get("chunks", [])
    json_bytes = len(json.dumps(chunks_doc, ensure_ascii=False).encode("utf-8"))
    by_parent: dict[str, list[dict]] = {}
    for ch in chunks:
        by_parent.setdefault(ch["block_id"], []).append(ch)
    childs = [len(v) for v in by_parent.values()]
    parent_text_total = sum(len(ch.get("parent_text", "") or "") for ch in chunks)
    parent_text_unique = sum(len(v[0].get("parent_text") or "") for v in by_parent.values())
    subjects_repeat = sum(
        len(json.dumps(ch["metadata"].get("subjects", []), ensure_ascii=False)) for ch in chunks
    )
    return {
        "json_bytes": json_bytes,
        "n_chunks": len(chunks),
        "n_parents": len(by_parent),
        "childs_mean": round(sum(childs) / len(childs), 2) if childs else 0,
        "childs_max": max(childs) if childs else 0,
        "parent_text_total_chars": parent_text_total,
        "parent_text_unique_chars": parent_text_unique,
        "parent_text_redundant_chars": parent_text_total - parent_text_unique,
        "subjects_repeated_chars": subjects_repeat,
        "overlap_duplicated_chars": overlap.get("duplicated_chars", 0),
    }


# --------------------------------------------------------------------------- #
# Citas y metadatos (clasificación estática)
# --------------------------------------------------------------------------- #


def classify_metadata() -> list[dict]:
    """Clasifica cada campo de metadata del chunk por su propósito."""
    return [
        {"field": "citation_label", "category": "cita", "note": "etiqueta humana de la cita"},
        {"field": "source_url", "category": "cita", "note": "enlace oficial con ancla #block_id"},
        {"field": "document_id", "category": "cita+filtro", "note": "identidad de la norma"},
        {"field": "block_id", "category": "cita+trazabilidad", "note": "ancla del bloque"},
        {
            "field": "parent_id",
            "category": "trazabilidad",
            "note": "join al bloque padre (parent-child)",
        },
        {
            "field": "source_norm_id",
            "category": "trazabilidad",
            "note": "norma que fijó la versión vigente",
        },
        {
            "field": "hierarchy",
            "category": "filtro",
            "note": "título/capítulo/sección (incompleta: ver H2)",
        },
        {"field": "rank", "category": "filtro", "note": "rango de la norma"},
        {"field": "scope", "category": "filtro", "note": "ámbito"},
        {
            "field": "subjects",
            "category": "filtro",
            "note": "materias; voluminoso si se repite por vector",
        },
        {"field": "norm_title", "category": "no_repetir_por_vector", "note": "constante por norma"},
        {
            "field": "legal_status_notice",
            "category": "no_repetir_por_vector",
            "note": "constante global",
        },
    ]


# --------------------------------------------------------------------------- #
# Trazabilidad XML → documento → chunk
# --------------------------------------------------------------------------- #


def trace_block(raw_dir: Path, norm_id: str, block_id: str, doc: dict, chunks_doc: dict) -> dict:
    """Reconstruye un bloque desde el XML raw hasta documento y chunks."""
    texto_path = Path(raw_dir) / norm_id / "texto.xml"
    xml_snippet = None
    n_versions_xml = 0
    if texto_path.is_file():
        data = validate_response(load_xml(texto_path), texto_path)
        node = data.find(f".//bloque[@id='{block_id}']")
        if node is not None:
            n_versions_xml = len(node.findall("version"))
            raw = etree.tostring(node, encoding="unicode")
            xml_snippet = raw[:600] + (" …" if len(raw) > 600 else "")

    block = next((b for b in doc.get("blocks", []) if b["block_id"] == block_id), None)
    chunks = [c for c in chunks_doc.get("chunks", []) if c["block_id"] == block_id]
    doc_out = None
    if block:
        lv = block.get("latest_version") or {}
        doc_out = {
            "block_type": block.get("block_type"),
            "block_title": block.get("block_title"),
            "full_title": block.get("full_title"),
            "hierarchy": block.get("hierarchy"),
            "n_versions": len(block.get("versions") or []),
            "latest_source_norm_id": lv.get("source_norm_id"),
            "n_paragraphs": len(lv.get("paragraphs", [])),
            "n_modification_notes": len(lv.get("modification_notes", [])),
            "text_chars": len(lv.get("text", "") or ""),
            "indexable": (block.get("retrieval") or {}).get("indexable"),
        }
    return {
        "norma": norm_id,
        "block_id": block_id,
        "xml": {"n_versions": n_versions_xml, "snippet": xml_snippet},
        "documento": doc_out,
        "chunks": [
            {
                "chunk_id": c["chunk_id"],
                "text_chars": len(c["text"]),
                "chunk_index": c["chunk_index"],
                "chunk_count_for_parent": c["chunk_count_for_parent"],
            }
            for c in chunks
        ],
    }


# --------------------------------------------------------------------------- #
# Agregación
# --------------------------------------------------------------------------- #


def summarize(findings: list[dict]) -> dict:
    """Agrega los hallazgos por severidad, clasificación y check."""
    from collections import Counter

    by_sev: Counter = Counter()
    by_class: Counter = Counter()
    by_check: Counter = Counter()
    for f in findings:
        by_sev[f["severity"]] += 1
        by_class[f["classification"]] += 1
        by_check[f["check"]] += 1
    return {
        "total": len(findings),
        "by_severity": dict(by_sev),
        "by_classification": dict(by_class),
        "by_check": dict(by_check),
    }


# --------------------------------------------------------------------------- #
# Integridad temporal (vigencia)
# --------------------------------------------------------------------------- #


def temporal_integrity(
    docs: dict[str, dict],
    chunks: dict[str, dict] | None = None,
    processing_date: str | None = None,
) -> dict:
    """Audita la vigencia temporal de todo el corpus (machine-readable).

    Recalcula la resolución de cada bloque desde `versions[]` + `index_last_update_date` (no
    confía en lo persistido) y agrega listas de divergencias. `ready=false` si hay cualquier
    bloque irresoluble, mismatch o chunk construido desde versión histórica. La entrada en vigor
    futura se reporta pero no bloquea (política explícita del MVP).
    """
    chunks = chunks or {}
    if processing_date is None:
        processing_date = datetime.date.today().isoformat()

    blocks_checked = versioned_blocks = latest_matches_index = 0
    non_chrono: list[str] = []
    mismatches: list[str] = []
    ambiguous_blocks: list[str] = []
    missing_index_date: list[str] = []
    missing_publication_date: list[str] = []
    invalid_dates: list[str] = []
    index_not_max: list[str] = []
    future_effective: list[str] = []
    chunks_non_current: list[str] = []
    warnings: list[str] = []

    for did, doc in docs.items():
        bmap = {b["block_id"]: b for b in doc.get("blocks", [])}
        for b in doc.get("blocks", []):
            blocks_checked += 1
            versions = b.get("versions") or []
            if not versions:
                continue
            versioned_blocks += 1
            ref = f"{did}/{b.get('block_id')}"
            pubs = [v.get("publication_date") for v in versions]
            if all(pubs) and pubs != sorted(pubs):
                non_chrono.append(ref)
            if any(p is None for p in pubs):
                missing_publication_date.append(ref)
            if (b.get("temporal_resolution") or {}).get("status") == "invalid_date":
                invalid_dates.append(ref)

            res = resolve_current_version(versions, b.get("index_last_update_date"))
            status = res["status"]
            lv = b.get("latest_version") or {}
            if status == "resolved":
                if (
                    lv.get("publication_date") != res["selected_publication_date"]
                    or lv.get("source_norm_id") != res["selected_source_norm_id"]
                ):
                    mismatches.append(ref)
                    warnings.append(f"{ref}: latest_version != selección por índice")
                else:
                    latest_matches_index += 1
                    vdate = lv.get("validity_date")
                    if vdate and vdate > processing_date:
                        future_effective.append(ref)
            elif status == "ambiguous":
                ambiguous_blocks.append(ref)
            elif status == "missing_index_date":
                if ref not in invalid_dates:
                    missing_index_date.append(ref)
            elif status == "index_not_max":
                index_not_max.append(ref)
            else:  # unresolved
                mismatches.append(ref)

        for ch in chunks.get(did, {}).get("chunks", []):
            b = bmap.get(ch.get("block_id"))
            if not b:
                continue
            res = resolve_current_version(b.get("versions") or [], b.get("index_last_update_date"))
            if res["status"] != "resolved":
                continue
            cpub = (ch.get("metadata") or {}).get("publication_date")
            if cpub and cpub != res["selected_publication_date"]:
                chunks_non_current.append(ch.get("chunk_id"))

    ready = not (
        mismatches
        or ambiguous_blocks
        or missing_index_date
        or invalid_dates
        or index_not_max
        or chunks_non_current
    )
    return {
        "ready": ready,
        "processing_date": processing_date,
        "blocks_checked": blocks_checked,
        "versioned_blocks": versioned_blocks,
        "non_chronological_xml_order_blocks": non_chrono,
        "latest_matches_index": latest_matches_index,
        "mismatches": mismatches,
        "ambiguous_blocks": ambiguous_blocks,
        "missing_index_date": missing_index_date,
        "missing_publication_date": missing_publication_date,
        "invalid_dates": invalid_dates,
        "index_not_max": index_not_max,
        "future_effective_selected_versions": future_effective,
        "chunks_built_from_non_current_version": chunks_non_current,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- #
# Integridad raw (manifests)
# --------------------------------------------------------------------------- #


def verify_manifest(norm_id: str, manifest_dir: Path) -> dict:
    """Recomputa sha256/size de cada fichero del manifest de una norma contra el raw en disco."""
    manifest_path = Path(manifest_dir) / f"{norm_id}.json"
    result = {
        "norm_id": norm_id,
        "files_checked": 0,
        "missing_files": [],
        "size_mismatches": [],
        "sha256_mismatches": [],
    }
    if not manifest_path.is_file():
        result["missing_files"].append(manifest_path.as_posix())
        return result
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for f in manifest.get("files", []):
        result["files_checked"] += 1
        path = Path(f.get("path", ""))
        if not path.is_file():
            result["missing_files"].append(f.get("path"))
            continue
        data = path.read_bytes()
        if f.get("size_bytes") is not None and len(data) != f["size_bytes"]:
            result["size_mismatches"].append(f.get("path"))
        if f.get("sha256") and hashlib.sha256(data).hexdigest() != f["sha256"]:
            result["sha256_mismatches"].append(f.get("path"))
    return result


def raw_integrity(norm_ids: list[str], manifest_dir: Path) -> dict:
    """Agrega `verify_manifest` sobre el corpus → sección `raw_integrity` (machine-readable)."""
    files_checked = 0
    missing: list[str] = []
    size: list[str] = []
    sha: list[str] = []
    for nid in norm_ids:
        r = verify_manifest(nid, manifest_dir)
        files_checked += r["files_checked"]
        missing.extend(r["missing_files"])
        size.extend(r["size_mismatches"])
        sha.extend(r["sha256_mismatches"])
    return {
        "ready": not (missing or size or sha),
        "files_checked": files_checked,
        "missing_files": missing,
        "size_mismatches": size,
        "sha256_mismatches": sha,
    }


# --------------------------------------------------------------------------- #
# Gate previo a embeddings
# --------------------------------------------------------------------------- #


def compute_readiness(
    findings: list[dict],
    unhandled_hierarchy: dict,
    editorial_indexable: list,
    duplicate_catalog: bool = False,
    temporal: dict | None = None,
    raw: dict | None = None,
) -> dict:
    """`pre_embedding_readiness`: bloqueantes (corregir ahora) vs diferidos (a indexación)."""
    errors = [f for f in findings if f["severity"] == "ERROR"]
    note_leaks = [f for f in errors if f["check"] in ("block.note_leak", "chunk.note_leak")]
    heading_body_out = [f for f in errors if f["check"] == "block.heading_body_not_indexed"]
    rt_double = [f for f in findings if f["check"] == "rt.double_period"]

    blocking: list[str] = []
    if errors:
        blocking.append(f"integrity_errors={len(errors)}")
    if editorial_indexable:
        blocking.append("H1_nota_inicial_indexable")
    if rt_double:
        blocking.append("rt_double_period")
    if unhandled_hierarchy:
        blocking.append("unhandled_hierarchy")
    if heading_body_out:
        blocking.append("substantive_heading_not_indexed")
    if note_leaks:
        blocking.append("note_leak")
    if duplicate_catalog:
        blocking.append("duplicate_corpus_catalog")

    if temporal is not None and not temporal.get("ready", True):
        if temporal.get("mismatches"):
            blocking.append("temporal_mismatches")
        if temporal.get("ambiguous_blocks"):
            blocking.append("temporal_ambiguous")
        if temporal.get("missing_index_date"):
            blocking.append("temporal_missing_index_date")
        if temporal.get("invalid_dates"):
            blocking.append("temporal_invalid_date")
        if temporal.get("index_not_max"):
            blocking.append("temporal_index_not_max")
        if temporal.get("chunks_built_from_non_current_version"):
            blocking.append("chunks_built_from_non_current_version")
        if not any(b.startswith("temporal_") or b.startswith("chunks_") for b in blocking):
            blocking.append("temporal_unclassified")
    if raw is not None and not raw.get("ready", True):
        blocking.append("raw_integrity")

    return {
        "ready": not blocking,
        "blocking_findings": blocking,
        "deferred_findings": ["H3_oversized_token_measurement"],
    }
