"""Auditoría de calidad del corpus (contratos v2), de solo lectura.

Contrasta los artefactos persistidos v2 (`boe_legal_document_v2` + `boe_legal_history_v2` +
`boe_legal_parents_v2` + `boe_legal_chunks_v2`) contra el contrato esperado y produce hallazgos
clasificados + métricas. La **representación procesada autoritativa es compuesta**
(`documents + histories + parents`); `join_norm` la reconstruye en una vista rica para reutilizar
las comprobaciones jurídicas/temporales sobre la unidad de bloque.

Separación (precisión de diseño):
- Validación **local** por artefacto → modelos Pydantic (`src.contracts`).
- Validación **relacional** (joins, cobertura, ownership) → este módulo (`check_relational`,
  `check_parents`, `check_history`).

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
    classify_version_paragraphs,
    clean_text,
    heading_has_retrievable_body,
    is_suspicious_blockquote_class,
    load_xml,
    normalize_date,
    resolve_current_version,
    validate_response,
)

SCHEMA_VERSION = "boe_legal_document_v2"
HISTORY_SCHEMA_VERSION = "boe_legal_history_v2"
PARENTS_SCHEMA_VERSION = "boe_legal_parents_v2"
CHUNKS_SCHEMA_VERSION = "boe_legal_chunks_v2"


def join_norm(document: dict, history: dict, parents: dict) -> dict:
    """Reconstruye la vista rica por bloque desde los 3 artefactos v2 (composite autoritativo).

    Devuelve una estructura equivalente a la interna del parser (con `latest_version`,
    `versions`, `temporal_resolution`, `retrieval`, `index_*`), para que las comprobaciones
    jurídicas y temporales operen sobre la unidad de bloque sin duplicar lógica.
    """
    hist_by_block = {h["block_id"]: h for h in history.get("blocks", [])}
    parent_by_block = {p["block_id"]: p for p in parents.get("parents", [])}
    blocks = []
    for desc in document.get("blocks", []):
        bid = desc["block_id"]
        h = hist_by_block.get(bid, {})
        p = parent_by_block.get(bid)
        tr = h.get("temporal_resolution") or {}
        versions = [
            {
                "source_norm_id": v.get("source_norm_id"),
                "publication_date": v.get("publication_date"),
                "validity_date": v.get("validity_date"),
                "is_latest": bool(v.get("is_current")),
            }
            for v in h.get("versions", [])
        ]
        latest_version = None
        if p is not None:
            latest_version = {
                "source_norm_id": (p.get("current_version") or {}).get("source_norm_id"),
                "publication_date": (p.get("current_version") or {}).get("publication_date"),
                "validity_date": (p.get("current_version") or {}).get("validity_date"),
                "text": p.get("text", ""),
                "paragraphs": p.get("paragraphs", []),
                # Las notas de modificación son propiedad de history: se hidratan por join.
                "modification_notes": h.get("modification_notes", []),
            }
        blocks.append(
            {
                **desc,
                "versions": versions,
                "latest_version": latest_version,
                "temporal_resolution": tr,
                "temporal_quarantined": h.get("temporal_quarantined", False),
                "index_title": h.get("index_title"),
                "index_url": h.get("index_url"),
                "index_last_update_date": h.get("index_last_update_date"),
                "retrieval": {
                    "indexable": desc.get("indexable", False),
                    "retrieval_text": None,
                    "citation_label": (desc.get("citation") or {}).get("label"),
                    "source_url": (desc.get("citation") or {}).get("url"),
                    "excluded_reason": desc.get("excluded_reason"),
                },
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "document_id": document.get("document_id"),
        "source": document.get("source", {}),
        "metadata": document.get("metadata", {}),
        "analysis": document.get("analysis", {}),
        "blocks": blocks,
        "quality_checks": {},
    }


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
# Fórmulas editoriales del BOE que NUNCA deben quedar en el texto normativo indexado (gate
# anti-regresión del aparato editorial: avisos «Téngase…» y el marcador «Redacción anterior:»).
# Se exige el ':' tras «anterior(es)» para no confundir con la prosa de disposiciones transitorias.
# Es la red SECUNDARIA (por frases) frente a la invariante estructural de dos caras
# (`verify_editorial_invariant`): la invariante cubre el aparato envuelto en <blockquote>; este
# regex cubre el borde irreducible heurístico de las notas sueltas FUERA de blockquote.
EDITORIAL_LEAK_RE = re.compile(
    r"t[eé]ngase en cuenta|redacci[oó]n(es)?\s+anterior(es)?\s*:", re.IGNORECASE
)
# Umbral de anomalía: fracción de `<p>` de la versión vigente descartados como aparato editorial.
# Por encima, el bloque es mayoritariamente cita editorial → aflora para inspección humana.
EDITORIAL_DROP_FRACTION_WARN = 0.5
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
    "generation_meta",
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


def check_document(
    document: dict,
    history: dict | None = None,
    parents: dict | None = None,
    processing_date: str | None = None,
) -> list[dict]:
    """Verifica el contrato `boe_legal_document_v2` + las comprobaciones de bloque (joined).

    Si se pasan `history` y `parents`, los bloques se comprueban sobre la vista compuesta
    (texto/versiones/temporal reales). Sin ellos, solo se valida el descriptor de alto nivel.
    """
    out: list[dict] = []
    did = document.get("document_id")
    if processing_date is None:
        processing_date = datetime.date.today().isoformat()

    if document.get("schema_version") != SCHEMA_VERSION:
        out.append(
            finding(
                "doc.schema",
                "ERROR",
                "Revisar antes de embeddings",
                did,
                None,
                f"schema_version inesperado: {document.get('schema_version')!r}",
            )
        )
    for k in REQUIRED_DOC_KEYS:
        if k not in document:
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

    meta = document.get("metadata", {})
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

    # `source.manifest_ref` debe ser ruta RELATIVA y estable (no absoluta).
    manifest_ref = (document.get("source") or {}).get("manifest_ref")
    if manifest_ref and (Path(manifest_ref).is_absolute() or ":" in manifest_ref):
        out.append(
            finding(
                "doc.manifest_ref",
                "ERROR",
                "Revisar antes de embeddings",
                did,
                None,
                f"manifest_ref no es ruta relativa estable: {manifest_ref!r}",
            )
        )

    # Comprobaciones de bloque sobre la vista compuesta (si hay history+parents).
    if history is not None and parents is not None:
        joined = join_norm(document, history, parents)
        out.extend(_check_blocks(joined, did, processing_date))
    return out


def check_parents(document: dict, parents: dict) -> list[dict]:
    """Verifica el contrato `parents_v1` y su cobertura/propiedad del texto vigente."""
    out: list[dict] = []
    did = parents.get("document_id")
    if parents.get("schema_version") != PARENTS_SCHEMA_VERSION:
        out.append(
            finding(
                "parents.schema",
                "ERROR",
                "Revisar antes de embeddings",
                did,
                None,
                f"schema_version inesperado: {parents.get('schema_version')!r}",
            )
        )
    records = parents.get("parents", [])
    seen: set[str] = set()
    for p in records:
        pid = p.get("parent_id")
        if pid in seen:
            out.append(
                finding(
                    "parents.duplicate",
                    "ERROR",
                    "Revisar antes de embeddings",
                    did,
                    pid,
                    "parent_id duplicado en el parent store",
                )
            )
        seen.add(pid)
        if not (p.get("text") or "").strip():
            out.append(
                finding(
                    "parents.empty_text",
                    "ERROR",
                    "Revisar antes de embeddings",
                    did,
                    pid,
                    "parent sin texto vigente (no debería existir)",
                )
            )
        if "indexable" in p:
            out.append(
                finding(
                    "parents.indexable_present",
                    "ERROR",
                    "Revisar antes de embeddings",
                    did,
                    pid,
                    "parent contiene 'indexable' (propietario es document.blocks[])",
                )
            )
        if "modification_notes" in p:
            out.append(
                finding(
                    "parents.modification_notes_present",
                    "ERROR",
                    "Revisar antes de embeddings",
                    did,
                    pid,
                    "parent contiene 'modification_notes' (propietario es history)",
                )
            )
    return out


def check_history(document: dict, history: dict) -> list[dict]:
    """Verifica el contrato `history_v2` y que cubre TODOS los block_id del documento."""
    out: list[dict] = []
    did = history.get("document_id")
    if history.get("schema_version") != HISTORY_SCHEMA_VERSION:
        out.append(
            finding(
                "history.schema",
                "ERROR",
                "Revisar antes de embeddings",
                did,
                None,
                f"schema_version inesperado: {history.get('schema_version')!r}",
            )
        )
    doc_ids = [b["block_id"] for b in document.get("blocks", [])]
    hist_ids = {h["block_id"] for h in history.get("blocks", [])}
    for bid in doc_ids:
        if bid not in hist_ids:
            out.append(
                finding(
                    "history.coverage",
                    "ERROR",
                    "Revisar antes de embeddings",
                    did,
                    bid,
                    "block_id sin registro en history (debe existir para todos)",
                )
            )
    return out


def check_relational(document: dict, history: dict, parents: dict, chunks_doc: dict) -> list[dict]:
    """Comprobaciones relacionales entre los 4 artefactos (joins, ownership, cobertura)."""
    out: list[dict] = []
    did = document.get("document_id")
    desc_ids = {b["block_id"] for b in document.get("blocks", [])}
    parent_ids = {p["parent_id"] for p in parents.get("parents", [])}
    parent_block_ids = {p["block_id"] for p in parents.get("parents", [])}
    subject_codes = {
        s.get("code") for s in (document.get("analysis") or {}).get("subjects", []) if s.get("code")
    }

    # Cada parent pertenece a un descriptor y su block_id existe.
    for p in parents.get("parents", []):
        if p.get("block_id") not in desc_ids:
            out.append(
                finding(
                    "rel.parent_block_missing",
                    "ERROR",
                    "Revisar antes de embeddings",
                    did,
                    p.get("parent_id"),
                    "parent.block_id no existe en document",
                )
            )

    # Cada history.block_id existe en document.
    for h in history.get("blocks", []):
        if h.get("block_id") not in desc_ids:
            out.append(
                finding(
                    "rel.history_block_missing",
                    "ERROR",
                    "Revisar antes de embeddings",
                    did,
                    h.get("block_id"),
                    "history.block_id no existe en document",
                )
            )

    # Cada descriptor con parent_id apunta a un parent existente; y todo bloque resuelto con
    # texto debe tener parent (no se pierde texto vigente).
    for b in document.get("blocks", []):
        pid = b.get("parent_id")
        if pid is not None and pid not in parent_ids:
            out.append(
                finding(
                    "rel.parent_id_dangling",
                    "ERROR",
                    "Revisar antes de embeddings",
                    did,
                    b["block_id"],
                    f"parent_id {pid} sin registro en parents",
                )
            )

    # Cada chunk referencia un parent existente y un subject_code resoluble.
    for ch in chunks_doc.get("chunks", []):
        if ch.get("parent_id") not in parent_ids:
            out.append(
                finding(
                    "rel.chunk_parent_missing",
                    "ERROR",
                    "Revisar antes de embeddings",
                    did,
                    ch.get("chunk_id"),
                    "chunk.parent_id sin registro en parents",
                )
            )
        for code in (ch.get("filters") or {}).get("subject_codes", []):
            if code not in subject_codes:
                out.append(
                    finding(
                        "rel.subject_code_unknown",
                        "ERROR",
                        "Revisar antes de embeddings",
                        did,
                        ch.get("chunk_id"),
                        f"subject_code {code!r} no resuelve en document",
                    )
                )
                break

    # current_version del parent coincide con la versión vigente del history.
    hist_by_id = {h["block_id"]: h for h in history.get("blocks", [])}
    for p in parents.get("parents", []):
        h = hist_by_id.get(p.get("block_id"))
        if not h:
            continue
        current = next((v for v in h.get("versions", []) if v.get("is_current")), None)
        cv = p.get("current_version") or {}
        if current and (
            current.get("publication_date") != cv.get("publication_date")
            or current.get("source_norm_id") != cv.get("source_norm_id")
        ):
            out.append(
                finding(
                    "rel.current_version_mismatch",
                    "ERROR",
                    "Revisar antes de embeddings",
                    did,
                    p.get("parent_id"),
                    "parent.current_version != versión is_current de history",
                )
            )

    # parent_id null ⇒ no debe existir parent para ese bloque.
    for b in document.get("blocks", []):
        if b.get("parent_id") is None and b["block_id"] in parent_block_ids:
            out.append(
                finding(
                    "rel.null_parent_but_present",
                    "ERROR",
                    "Revisar antes de embeddings",
                    did,
                    b["block_id"],
                    "parent_id null pero existe parent para el bloque",
                )
            )

    # Ningún texto vigente completo fuera de parents: el chunk.text debe ser substring del
    # texto del parent correspondiente (no copia del bloque completo salvo que sea su único chunk).
    parents_text = {p["parent_id"]: p.get("text", "") for p in parents.get("parents", [])}
    for ch in chunks_doc.get("chunks", []):
        ptext = parents_text.get(ch.get("parent_id"))
        if ptext is not None and ch.get("text", "") not in ptext:
            # los chunks se forman de párrafos del parent; su texto siempre está contenido.
            out.append(
                finding(
                    "rel.chunk_text_not_in_parent",
                    "ERROR",
                    "Revisar antes de embeddings",
                    did,
                    ch.get("chunk_id"),
                    "chunk.text no está contenido en el texto del parent",
                )
            )
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
        leak = EDITORIAL_LEAK_RE.search(lv.get("text") or "")
        if leak:
            out.append(
                finding(
                    "block.editorial_leak",
                    "ERROR",
                    "Revisar antes de embeddings",
                    did,
                    bid,
                    "fórmula editorial (Téngase/Redacción anterior:) en el texto normativo",
                    evidence=leak.group(0),
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


def check_chunks(chunks_doc: dict, joined_doc: dict) -> list[dict]:
    """Verifica el contrato `boe_legal_chunks_v2` y su coherencia con la vista compuesta.

    `joined_doc` es la vista de `join_norm` (document+history+parents): aporta `latest_version`,
    `versions`, `index_*` y `retrieval.indexable` por bloque. Los chunks v2 no llevan
    `parent_text` ni metadatos documentales; el texto del padre se resuelve por join.
    """
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

    blocks = {b["block_id"]: b for b in joined_doc.get("blocks", [])}
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

        # Eficiencia: ningún chunk debe arrastrar parent_text ni metadatos documentales.
        if "parent_text" in ch:
            out.append(
                finding(
                    "chunk.parent_text_present",
                    "ERROR",
                    "Revisar antes de embeddings",
                    did,
                    cid,
                    "el chunk contiene parent_text (debe resolverse por join a parents)",
                )
            )

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
        # Vigencia: el chunk no debe proceder de una versión histórica.
        res = resolve_current_version(b.get("versions") or [], b.get("index_last_update_date"))
        if res["status"] == "resolved":
            cpub = (b.get("latest_version") or {}).get("publication_date")
            sel = res["selected_publication_date"]
            if cpub and sel and cpub != sel:
                out.append(
                    finding(
                        "chunk.temporal_stale",
                        "ERROR",
                        "Revisar antes de embeddings",
                        did,
                        cid,
                        "chunk construido desde una versión no vigente",
                        evidence=f"{cpub} != {sel}",
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
        # Cita preservada en el payload (label + url oficial con ancla).
        cit = ch.get("citation") or {}
        if not cit.get("label"):
            out.append(
                finding(
                    "chunk.citation",
                    "ERROR",
                    "Revisar antes de embeddings",
                    did,
                    cid,
                    "falta citation.label en el chunk",
                )
            )
        # nota / redacción histórica dentro del texto del chunk (join a parents).
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
        if EDITORIAL_LEAK_RE.search(ch.get("text", "") or "") or EDITORIAL_LEAK_RE.search(
            ch.get("retrieval_text", "") or ""
        ):
            out.append(
                finding(
                    "chunk.editorial_leak",
                    "ERROR",
                    "Revisar antes de embeddings",
                    did,
                    cid,
                    "fórmula editorial dentro del texto del chunk",
                )
            )
        out.extend(check_retrieval_text(ch, b))
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
        chs_sorted = sorted(chs, key=lambda c: (c.get("position") or {}).get("index", 0))
        idxs = [(c.get("position") or {}).get("index") for c in chs_sorted]
        if idxs != list(range(1, len(chs_sorted) + 1)):
            out.append(
                finding(
                    "chunk.sequence",
                    "ERROR",
                    "Revisar antes de embeddings",
                    did,
                    bid,
                    f"position.index no secuencial: {idxs}",
                )
            )
        if any(
            (c.get("position") or {}).get("count_for_parent") != len(chs_sorted) for c in chs_sorted
        ):
            out.append(
                finding(
                    "chunk.count",
                    "ERROR",
                    "Revisar antes de embeddings",
                    did,
                    bid,
                    "position.count_for_parent incoherente",
                )
            )
    return out


def check_retrieval_text(chunk: dict, block: dict) -> list[dict]:
    """Audita la calidad del `retrieval_text` de un chunk v2 (contexto desde el bloque joined)."""
    out: list[dict] = []
    did = chunk.get("document_id")
    cid = chunk.get("chunk_id")
    rt = chunk.get("retrieval_text", "") or ""
    ft = block.get("full_title")
    hier = block.get("hierarchy") or {}

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


def efficiency_metrics(chunks_doc: dict, overlap: dict, parents_doc: dict | None = None) -> dict:
    """Tamaños y redundancia del payload v2 (chunks vector-ready + parent store)."""
    chunks = chunks_doc.get("chunks", [])
    json_bytes = len(json.dumps(chunks_doc, ensure_ascii=False).encode("utf-8"))
    by_parent: dict[str, list[dict]] = {}
    for ch in chunks:
        by_parent.setdefault(ch["block_id"], []).append(ch)
    childs = [len(v) for v in by_parent.values()]
    # v2: el chunk NO lleva parent_text ni subjects -> redundancia 0 por diseño.
    parent_text_in_chunks = sum(len(ch.get("parent_text", "") or "") for ch in chunks)
    parents_unique_chars = 0
    if parents_doc is not None:
        parents_unique_chars = sum(
            len(p.get("text", "") or "") for p in parents_doc.get("parents", [])
        )
    return {
        "json_bytes": json_bytes,
        "n_chunks": len(chunks),
        "n_parents": len(by_parent),
        "childs_mean": round(sum(childs) / len(childs), 2) if childs else 0,
        "childs_max": max(childs) if childs else 0,
        "parent_text_in_chunks_chars": parent_text_in_chunks,
        "parents_store_unique_text_chars": parents_unique_chars,
        "subjects_repeated_chars": 0,
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
                "chunk_index": (c.get("position") or {}).get("index"),
                "chunk_count_for_parent": (c.get("position") or {}).get("count_for_parent"),
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
            # v2: el chunk no lleva fecha; su vigencia se hereda del parent (joined latest_version).
            cpub = (b.get("latest_version") or {}).get("publication_date")
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
# Aparato editorial — invariante estructural de dos caras + observabilidad del drop
# --------------------------------------------------------------------------- #


def verify_editorial_invariant(
    version_el: etree._Element,
    persisted_paragraphs: list[dict],
    did: str | None,
    bid: str | None,
) -> tuple[list[dict], dict]:
    """Reaplica la regla canónica al raw vigente y la contrasta con el cuerpo persistido.

    Núcleo PURO de la invariante estructural (sin disco). Reclasifica los `<p>` de la `<version>`
    vigente raw con `classify_version_paragraphs` (la MISMA regla usada al persistir, recomputada
    aparte — igual que `temporal_integrity` recomputa `resolve_current_version`) y compara el cuerpo
    esperado contra el persistido por **pertenencia/orden** (igualdad de lista de textos), nunca por
    containment de texto: el casi-duplicado redacción-derogada≈vigente daría falso positivo, como ya
    ocurrió con los 89 FP de `note_leak`.

    Doble cara: un `<p>` de más en `paragraphs` (fuga editorial) rompe la igualdad; uno de menos
    (sobre-borrado de texto vigente) también. Devuelve `(findings, drop_record)`:
    - `block.editorial_invariant` (ERROR) si difieren, con el primer `<p>` que diverge.
    - `block.editorial_drop_structural` (WARN) si se descartó clase estructural/tabla/artículo.
    - `block.editorial_drop_fraction` (WARN) si se descartó > `EDITORIAL_DROP_FRACTION_WARN`.
    """
    from collections import Counter

    kept, notes, dropped = classify_version_paragraphs(version_el)
    expected = [p["text"] for p in kept]
    actual = [p.get("text") for p in persisted_paragraphs]
    findings: list[dict] = []

    if expected != actual:
        i = next(
            (
                k
                for k in range(max(len(expected), len(actual)))
                if _at(expected, k) != _at(actual, k)
            ),
            0,
        )
        kind = "fuga" if len(actual) > len(expected) else "sobre-borrado"
        findings.append(
            finding(
                "block.editorial_invariant",
                "ERROR",
                "Revisar antes de embeddings",
                did,
                bid,
                f"cuerpo persistido != recomputado del raw ({kind}); "
                f"primer <p> que difiere en posición {i}",
                evidence=f"esperado={_at(expected, i)!r} | persistido={_at(actual, i)!r}",
            )
        )

    dropped_classes = Counter(d["class"] for d in dropped)
    suspicious = {c for c in dropped_classes if is_suspicious_blockquote_class(c)}
    n_total = len(kept) + len(notes) + len(dropped)
    fraction = len(dropped) / n_total if n_total else 0.0
    sample = dropped[0]["text"][:80] if dropped else None

    if suspicious:
        findings.append(
            finding(
                "block.editorial_drop_structural",
                "WARN",
                "Revisar antes de embeddings",
                did,
                bid,
                f"se descartó clase estructural/tabla en <blockquote>: {sorted(suspicious)} "
                "(posible estructura vigente mal envuelta)",
                evidence=sample,
            )
        )
    if fraction > EDITORIAL_DROP_FRACTION_WARN:
        findings.append(
            finding(
                "block.editorial_drop_fraction",
                "WARN",
                "Revisar antes de embeddings",
                did,
                bid,
                f"se descartó {fraction:.0%} de los <p> de la versión vigente como editorial "
                f"({len(dropped)}/{n_total})",
                evidence=sample,
            )
        )

    drop_record = {
        "document_id": did,
        "block_id": bid,
        "n_kept": len(kept),
        "n_notes": len(notes),
        "n_dropped_blockquote": len(dropped),
        "dropped_fraction": round(fraction, 3),
        "dropped_classes": dict(dropped_classes),
        "sample": sample,
        "anomaly": sorted(
            (["structural_class"] if suspicious else [])
            + (["high_fraction"] if fraction > EDITORIAL_DROP_FRACTION_WARN else [])
        ),
    }
    return findings, drop_record


def _at(seq: list, i: int):
    """Elemento `i` de la lista o `None` si fuera de rango (para diffs de longitud distinta)."""
    return seq[i] if 0 <= i < len(seq) else None


def _vigente_version_el(bloque: etree._Element, index_date, lv: dict) -> etree._Element | None:
    """Devuelve la `<version>` raw que ambos lados reconocen como vigente, o None.

    Recomputa `resolve_current_version` desde el raw (igual que `temporal_integrity`) y exige que
    coincida con la versión persistida (`source_norm_id` + `publication_date`). None si hay
    cuarentena o disputa de versión — lo cubre `temporal_integrity`, no estas comprobaciones.
    """
    version_elements = bloque.findall("version")
    raw_versions = [
        {
            "publication_date": normalize_date(v.get("fecha_publicacion")),
            "source_norm_id": v.get("id_norma"),
        }
        for v in version_elements
    ]
    res = resolve_current_version(raw_versions, index_date)
    sel = res["selected_version_index"]
    if res["status"] != "resolved" or sel is None:
        return None
    if res["selected_source_norm_id"] != lv.get("source_norm_id") or res[
        "selected_publication_date"
    ] != lv.get("publication_date"):
        return None
    return version_elements[sel]


def check_editorial_drop(
    joined_doc: dict, raw_dir: Path, did: str
) -> tuple[list[dict], list[dict]]:
    """Invariante estructural + observabilidad del drop editorial sobre una norma (lee `texto.xml`).

    Para CADA bloque con versión vigente RESUELTA, localiza la `<version>` vigente en el raw
    (recomputando `resolve_current_version` de forma independiente) y delega en
    `verify_editorial_invariant`. Los bloques en cuarentena (sin `latest_version`) y los de versión
    en disputa (lo cubre `temporal_integrity`) se omiten para no doblar hallazgos. Devuelve
    `(findings, drop_records)`; solo se reportan `drop_records` de bloques con algún descarte.
    """
    texto_path = Path(raw_dir) / did / "texto.xml"
    if not texto_path.is_file():
        return [], []
    data = validate_response(load_xml(texto_path), texto_path)
    texto = data.find("texto")
    if texto is None:
        return [], []
    bloque_by_id = {b.get("id"): b for b in texto.findall("bloque")}

    findings: list[dict] = []
    drop_records: list[dict] = []
    for b in joined_doc.get("blocks", []):
        lv = b.get("latest_version")
        if not lv:  # cuarentena: el parser no produjo cuerpo, nada que verificar aquí.
            continue
        bid = b.get("block_id")
        bloque = bloque_by_id.get(bid)
        if bloque is None:
            findings.append(
                finding(
                    "block.editorial_invariant",
                    "ERROR",
                    "Revisar antes de embeddings",
                    did,
                    bid,
                    "bloque con versión vigente pero ausente en texto.xml raw",
                )
            )
            continue
        version_el = _vigente_version_el(bloque, b.get("index_last_update_date"), lv)
        if version_el is None:
            continue  # cuarentena/disputa de versión: lo cubre temporal_integrity.
        block_findings, drop_record = verify_editorial_invariant(
            version_el, lv.get("paragraphs", []), did, bid
        )
        findings.extend(block_findings)
        if drop_record["n_dropped_blockquote"]:
            drop_records.append(drop_record)
    return findings, drop_records


def _missing_raw_cells(version_el: etree._Element, body: str) -> list[str]:
    """Celdas `<td>`/`<th>` de tabla forma B, vigentes (fuera de blockquote), ausentes del cuerpo.

    Recorre las celdas con texto CRUDO (sin `<p>` dentro = forma B; la forma A vive en `<p>`, ya
    en el cuerpo) fuera de `<blockquote>` y devuelve las distintivas (con letra, longitud ≥ 4) cuyo
    texto no aparece en `body`. La longitud/letra evita falsos positivos de celdas numéricas o
    vacías que colisionan por azar; una tabla entera omitida deja fuera sus celdas-concepto.
    """
    missing: list[str] = []
    for cell in (*version_el.iter("td"), *version_el.iter("th")):
        if next(cell.iterancestors("blockquote"), None) is not None:
            continue  # editorial (derogada): NO debe estar en el cuerpo
        if cell.find(".//p") is not None:
            continue  # forma A: el texto vive en <p>, ya capturado en el cuerpo
        text = clean_text("".join(cell.itertext()))
        if len(text) >= 4 and any(ch.isalpha() for ch in text) and text not in body:
            missing.append(text)
    return missing


def check_table_coverage(joined_doc: dict, raw_dir: Path, did: str) -> list[dict]:
    """Cierra el punto ciego `<p>`-céntrico: tablas forma B vigentes ausentes del cuerpo persistido.

    La invariante editorial compara listas de `<p>` y NO ve las tablas con texto crudo en `<td>`
    (forma B). Aquí, re-leyendo el raw, se marca ERROR si una celda distintiva de tabla forma B,
    FUERA de blockquote y en la versión vigente, no aparece en el cuerpo del bloque (ley vigente
    perdida en silencio). Omite cuarentenas/disputas (las cubre `temporal_integrity`).
    """
    texto_path = Path(raw_dir) / did / "texto.xml"
    if not texto_path.is_file():
        return []
    data = validate_response(load_xml(texto_path), texto_path)
    texto = data.find("texto")
    if texto is None:
        return []
    bloque_by_id = {b.get("id"): b for b in texto.findall("bloque")}

    findings: list[dict] = []
    for b in joined_doc.get("blocks", []):
        lv = b.get("latest_version")
        if not lv:
            continue
        bloque = bloque_by_id.get(b.get("block_id"))
        if bloque is None:
            continue
        version_el = _vigente_version_el(bloque, b.get("index_last_update_date"), lv)
        if version_el is None:
            continue
        missing = _missing_raw_cells(version_el, lv.get("text") or "")
        if missing:
            findings.append(
                finding(
                    "block.table_cell_dropped",
                    "ERROR",
                    "Revisar antes de embeddings",
                    did,
                    b.get("block_id"),
                    f"{len(missing)} celda(s) de tabla forma B vigente ausentes del cuerpo "
                    "(texto <td> crudo no capturado)",
                    evidence=" | ".join(missing[:3])[:160],
                )
            )
    return findings


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
    note_leaks = [
        f
        for f in errors
        if f["check"]
        in ("block.note_leak", "chunk.note_leak", "block.editorial_leak", "chunk.editorial_leak")
    ]
    heading_body_out = [f for f in errors if f["check"] == "block.heading_body_not_indexed"]
    rt_double = [f for f in findings if f["check"] == "rt.double_period"]
    invariant = [f for f in errors if f["check"] == "block.editorial_invariant"]
    table_drop = [f for f in errors if f["check"] == "block.table_cell_dropped"]

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
    if invariant:
        blocking.append("editorial_invariant")
    if table_drop:
        blocking.append("table_cell_dropped")
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
