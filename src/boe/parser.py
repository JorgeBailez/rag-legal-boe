"""Parser BOE: XML raw del BOE -> representación intermedia neutral -> contratos v2.

Convierte los XML raw ya descargados de una norma (metadatos, analisis, indice, texto) en una
**representación intermedia privada** (no persistida) y deriva de ella los tres contratos
persistidos: `boe_legal_document_v2` (descriptor), `boe_legal_history_v2` y
`boe_legal_parents_v2`. Ver `docs/modelo_documental.md`.

Capa de independencia: aísla al resto del sistema de la forma exacta del XML del BOE.
No usa red, no toca el raw, no usa `full.xml` como fuente y no parsea `metadata_eli.xml`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from lxml import etree

from src.contracts.models import DocumentV2, HistoryV2, ParentsV2
from src.core.exceptions import ParsingError

DOCUMENT_SCHEMA_VERSION = "boe_legal_document_v2"
HISTORY_SCHEMA_VERSION = "boe_legal_history_v2"
PARENTS_SCHEMA_VERSION = "boe_legal_parents_v2"
GENERATOR = "src.boe.parser"
SOURCE_NAME = "BOE legislación consolidada"
LEGAL_STATUS_NOTICE = (
    "Texto consolidado de carácter informativo, sin valor jurídico oficial. "
    "Remitir a la publicación oficial en el BOE."
)

# Ficheros raw que consume el parser v0 (excluye full.xml y metadata_eli.xml).
# Ficheros raw que conoce el parser. `analisis.xml` es opcional (puede no existir para
# algunas normas); el resto son obligatorios para producir el documento.
REQUIRED_RAW_FILES = ("metadatos.xml", "analisis.xml", "indice.xml", "texto.xml")
MANDATORY_RAW_FILES = ("metadatos.xml", "indice.xml", "texto.xml")

# Clases de párrafo que son notas editoriales (no texto normativo).
NOTE_CLASSES = {"nota_pie", "nota_pie_2"}

# Aviso editorial del BOE («Téngase en cuenta que…») que a veces aparece como párrafo de cuerpo
# FUERA de <blockquote>; red de seguridad por texto (lo demás va en blockquote).
_EDITORIAL_NOTE_PREFIX = "téngase en cuenta"

# Tipos que NUNCA son indexables aunque tengan cuerpo (cierre / nota editorial inicial).
EXCLUDED_TYPES = {"firma", "nota_inicial"}

# Clases de encabezado/rótulo estructural (no cuentan como cuerpo recuperable).
STRUCTURAL_LABEL_CLASSES = {
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

_NORM_ID_RE = re.compile(r"BOE-[A-Z]-\d{4}-\d+")

# Cuerpo normativo cuya única redacción vigente es «(Sin contenido)» (artículo vaciado).
_WITHOUT_CONTENT_RE = re.compile(r"^sin\s+contenido$", re.IGNORECASE)

# Estados posibles de la resolución temporal de la versión vigente de un bloque.
# Solo `resolved` es indexable; el resto van a cuarentena (sin `latest_version`, sin chunks).
TEMPORAL_STATUSES = (
    "resolved",
    "unresolved",
    "ambiguous",
    "missing_index_date",
    "invalid_date",
    "index_not_max",
)


def is_table_class(css: str) -> bool:
    """True si la clase de párrafo representa una celda/cabecera de tabla."""
    return css == "cabeza_tabla" or css.startswith("cuerpo_tabla_")


def heading_has_retrievable_body(paragraphs: list[dict]) -> bool:
    """True si el bloque tiene cuerpo recuperable (algún párrafo no estructural).

    Las celdas de tabla cuentan como cuerpo. No depende de `block_id`, de la palabra
    ANEXO ni de listas manuales de normas. Las imágenes no cuentan (sin texto asociado).
    """
    return any(p.get("class") not in STRUCTURAL_LABEL_CLASSES for p in paragraphs)


def is_without_content(paragraphs: list[dict]) -> bool:
    """True si la redacción vigente del bloque es únicamente «(Sin contenido)».

    Detección conservadora: el cuerpo (párrafos no estructurales y distintos del rótulo
    `articulo`) se reduce exactamente a «sin contenido». No infiere causa ni norma; eso
    queda para `analysis`/`modification_notes` si hay evidencia trazable.
    """
    body = [
        p.get("text") or ""
        for p in paragraphs
        if p.get("class") not in STRUCTURAL_LABEL_CLASSES and p.get("class") != "articulo"
    ]
    if not body:
        return False
    stripped = re.sub(r"[^0-9a-záéíóúñü ]", "", clean_text(" ".join(body)).lower()).strip()
    return bool(_WITHOUT_CONTENT_RE.match(stripped))


def resolve_current_version(versions: list[dict], index_date: str | None) -> dict:
    """Resuelve la versión vigente de un bloque a partir de la fecha del índice.

    Política estricta (sin fallback): la única selección válida para retrieval es la versión
    cuya `publication_date` coincide de forma **exacta y única** con `index_date` y que además
    es la **máxima** `publication_date` normalizable. Cualquier otro caso devuelve un estado de
    cuarentena (`unresolved`/`ambiguous`/`missing_index_date`/`index_not_max`); el orden XML
    nunca decide la vigencia. `max(publication_date)` se calcula solo como diagnóstico.
    """
    pubs = [v.get("publication_date") for v in versions]
    norm_pubs = [p for p in pubs if p]
    max_pub = max(norm_pubs) if norm_pubs else None
    candidate_versions = [
        {
            "version_index": i,
            "publication_date": pubs[i],
            "source_norm_id": versions[i].get("source_norm_id"),
        }
        for i in range(len(versions))
    ]
    result = {
        "status": None,
        "selection_method": None,
        "index_last_update_date": index_date,
        "selected_version_index": None,
        "selected_publication_date": None,
        "selected_source_norm_id": None,
        "candidate_versions": candidate_versions,
        "max_publication_date": max_pub,
        "warnings": [],
    }
    if not versions:
        result["status"] = "unresolved"
        result["warnings"] = ["no_versions"]
        return result
    if not index_date:
        result["status"] = "missing_index_date"
        result["warnings"] = ["missing_index_date"]
        return result

    matches = [i for i, p in enumerate(pubs) if p and p == index_date]
    if len(matches) == 0:
        result["status"] = "unresolved"
        result["warnings"] = ["index_no_match"]
        return result
    if len(matches) > 1:
        result["status"] = "ambiguous"
        result["warnings"] = ["index_multiple_match"]
        return result

    sel = matches[0]
    if max_pub is None or pubs[sel] != max_pub:
        result["status"] = "index_not_max"
        result["warnings"] = ["index_not_max"]
        return result

    result.update(
        status="resolved",
        selection_method="index_date_exact_unique_match",
        selected_version_index=sel,
        selected_publication_date=pubs[sel],
        selected_source_norm_id=versions[sel].get("source_norm_id"),
    )
    return result


# --------------------------------------------------------------------------- #
# Infraestructura / utilidades
# --------------------------------------------------------------------------- #


def load_xml(path: Path) -> etree._Element:
    """Carga un XML desde disco y devuelve su elemento raíz."""
    try:
        data = Path(path).read_bytes()
    except FileNotFoundError as exc:
        raise ParsingError(f"No existe el fichero XML: {path}") from exc
    try:
        return etree.fromstring(data)
    except etree.XMLSyntaxError as exc:
        raise ParsingError(f"XML inválido en {path}: {exc}") from exc


def validate_response(root: etree._Element, source_path: Path) -> etree._Element:
    """Valida el envoltorio `response` y devuelve el elemento `data`.

    Exige raíz `response`, `status/code == "200"` y la presencia de `data`.
    """
    if root.tag != "response":
        raise ParsingError(f"Raíz inesperada {root.tag!r} (esperado 'response') en {source_path}")

    code = root.findtext("status/code")
    if code is None:
        raise ParsingError(f"Falta status/code en {source_path}")
    if code.strip() != "200":
        raise ParsingError(f"status/code={code!r} (esperado '200') en {source_path}")

    data = root.find("data")
    if data is None:
        raise ParsingError(f"Falta el nodo 'data' en {source_path}")
    return data


def normalize_date(value: str | None) -> str | None:
    """`YYYYMMDD` -> `YYYY-MM-DD`. Devuelve None si está vacío o no encaja."""
    if not value:
        return None
    value = value.strip()
    match = re.fullmatch(r"(\d{4})(\d{2})(\d{2})", value)
    if not match:
        return None
    return "-".join(match.groups())


def normalize_datetime(value: str | None) -> str | None:
    """`YYYYMMDDThhmmssZ` -> `YYYY-MM-DDTHH:MM:SSZ`. None si vacío o no encaja."""
    if not value:
        return None
    value = value.strip()
    match = re.fullmatch(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z", value)
    if not match:
        return None
    y, mo, d, h, mi, s = match.groups()
    return f"{y}-{mo}-{d}T{h}:{mi}:{s}Z"


def clean_text(value: str | None) -> str:
    """Normaliza espacios internos sin alterar el contenido textual."""
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _element_text(element: etree._Element) -> str:
    """Texto limpio de un elemento, conservando el texto de etiquetas inline."""
    return clean_text("".join(element.itertext()))


def _coded(element: etree._Element | None) -> dict | None:
    """Convierte un elemento con atributo `codigo` en `{code, label}`."""
    if element is None:
        return None
    return {"code": element.get("codigo"), "label": clean_text(element.text)}


def _extract_norm_id(text: str) -> str | None:
    """Primer identificador BOE-A-... encontrado en un texto, o None."""
    match = _NORM_ID_RE.search(text or "")
    return match.group(0) if match else None


# --------------------------------------------------------------------------- #
# Metadatos
# --------------------------------------------------------------------------- #


def parse_metadata(data: etree._Element) -> dict:
    """Mapea `data/metadatos` al bloque `metadata` del contrato."""
    meta = data.find("metadatos")
    if meta is None:
        raise ParsingError("Falta el nodo 'metadatos' en el XML de metadatos")

    def text(tag: str) -> str | None:
        value = meta.findtext(tag)
        return clean_text(value) if value else None

    rank = _coded(meta.find("rango"))
    official_number = text("numero_oficial")
    short_title = None
    if rank and rank.get("label") and official_number:
        short_title = f"{rank['label']} {official_number}"

    return {
        "title": text("titulo"),
        "short_title": short_title,
        "identifier": text("identificador"),
        "eli_url": text("url_eli"),
        "html_url": text("url_html_consolidada"),
        "scope": _coded(meta.find("ambito")),
        "department": _coded(meta.find("departamento")),
        "rank": rank,
        "official_number": official_number,
        "publication_date": normalize_date(meta.findtext("fecha_publicacion")),
        "document_date": normalize_date(meta.findtext("fecha_disposicion")),
        "effective_date": normalize_date(meta.findtext("fecha_vigencia")),
        "last_update_datetime": normalize_datetime(meta.findtext("fecha_actualizacion")),
        "consolidation_status": _coded(meta.find("estado_consolidacion")),
        "derogation_status": text("estatus_derogacion"),
        "annulment_status": text("estatus_anulacion"),
        "expired_validity": text("vigencia_agotada"),
        "legal_status_notice": LEGAL_STATUS_NOTICE,
    }


# --------------------------------------------------------------------------- #
# Análisis
# --------------------------------------------------------------------------- #


def _parse_references(parent: etree._Element | None, child_tag: str) -> list[dict]:
    if parent is None:
        return []
    references = []
    for ref in parent.findall(child_tag):
        references.append(
            {
                "target_norm_id": clean_text(ref.findtext("id_norma")) or None,
                "relation": _coded(ref.find("relacion")),
                "text": clean_text(ref.findtext("texto")),
            }
        )
    return references


def parse_analysis(data: etree._Element) -> dict:
    """Mapea `data/analisis` al bloque `analysis` del contrato."""
    analysis = data.find("analisis")
    if analysis is None:
        return {"subjects": [], "notes": [], "references": {"previous": [], "next": []}}

    subjects = [
        {"code": m.get("codigo"), "label": clean_text(m.text)}
        for m in analysis.findall("materias/materia")
    ]
    notes = [{"text": clean_text(n.text)} for n in analysis.findall("notas/nota")]

    references = analysis.find("referencias")
    previous = next_refs = []
    if references is not None:
        previous = _parse_references(references.find("anteriores"), "anterior")
        next_refs = _parse_references(references.find("posteriores"), "posterior")

    return {
        "subjects": subjects,
        "notes": notes,
        "references": {"previous": previous, "next": next_refs},
    }


# --------------------------------------------------------------------------- #
# Índice
# --------------------------------------------------------------------------- #


def parse_index(data: etree._Element) -> list[dict]:
    """Mapea los `bloque` de `indice.xml` a una lista ordenada."""
    blocks = []
    for order, bloque in enumerate(data.findall("bloque")):
        raw_update = clean_text(bloque.findtext("fecha_actualizacion")) or None
        blocks.append(
            {
                "block_id": clean_text(bloque.findtext("id")),
                "index_title": clean_text(bloque.findtext("titulo")) or None,
                "index_last_update_date": normalize_date(bloque.findtext("fecha_actualizacion")),
                "index_last_update_date_raw": raw_update,
                "index_url": clean_text(bloque.findtext("url")) or None,
                "order": order,
            }
        )
    return blocks


# --------------------------------------------------------------------------- #
# Texto / bloques
# --------------------------------------------------------------------------- #


def is_suspicious_blockquote_class(css: str) -> bool:
    """True si una clase, descartada por estar DENTRO de `<blockquote>`, podría ser texto vigente.

    Estructura, tabla o artículo citados en un blockquote son aparato editorial legítimo (un modelo
    o redacción reproducidos), pero también podrían ser estructura/tabla VIGENTE mal envuelta: su
    descarte debe **aflorar** (warning en el parser, anomalía en la auditoría), no en silencio.
    """
    return css == "articulo" or css in STRUCTURAL_LABEL_CLASSES or is_table_class(css)


def _linearize_table_rows(table: etree._Element) -> list[tuple[str, str]]:
    """Linealiza una tabla POR FILA (forma A o B): une las celdas no vacías de cada `<tr>`.

    Por fila (no celda-a-celda suelta) para que el cuerpo quede legible para retrieval («concepto:
    valor», p. ej. `Inferior a 1 año. | 0,14`). Devuelve `(clase, texto)` por fila con contenido;
    clase `cabeza_tabla` si la fila trae `<th>` (cabecera de columnas), `cuerpo_tabla_fila` si no
    (ambas reconocibles por `is_table_class`).
    """
    rows: list[tuple[str, str]] = []
    for tr in table.iter("tr"):
        cells = [c for c in tr if c.tag in ("td", "th")]
        texts = [t for t in (_element_text(c) for c in cells) if t]
        if not texts:
            continue
        css = "cabeza_tabla" if any(c.tag == "th" for c in cells) else "cuerpo_tabla_fila"
        rows.append((css, " | ".join(texts)))
    return rows


def classify_version_paragraphs(
    version: etree._Element,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Regla CANÓNICA del cuerpo: clasifica el contenido de una versión en tres cubos.

    El BOE envuelve TODO el aparato editorial en `<blockquote>`: notas de modificación, avisos
    «Téngase en cuenta…», el marcador «Redacción anterior:» y la **redacción derogada citada**. El
    texto VIGENTE vive FUERA del blockquote. Como `version.iter("p")` es recursivo, esos `<p>` se
    colaban en el cuerpo (incl. ley no vigente). La clasificación es por **contenedor** (ancestro
    `<blockquote>`, recursivo — anidan), no por clase: dentro de un blockquote hay `<p
    class="parrafo">` idénticos al vigente (p. ej. `siempreSeVe`). Se mantiene `NOTE_CLASSES` y una
    red para la nota «Téngase» suelta fuera de blockquote.

    Recorre la versión en **orden de documento** capturando los `<p>` y linealizando **todas las
    tablas POR FILA** (forma A `<td><p class="cuerpo_tabla">` y forma B `<td>` crudo): cada `<tr>`
    queda en una línea "concepto | valor", emparejada. Los `<p>` **dentro de un `<table>`** se
    saltan (su texto ya entra por la fila) para no duplicar. La regla de blockquote se respeta
    también para las tablas (tabla dentro de blockquote = editorial → `dropped`; fuera = `kept`).

    Devuelve `(kept, notes, dropped)`:
    - `kept`: cuerpo normativo indexable (`<p>` + filas de tabla vigente).
    - `notes`: provenance que NO solapa el cuerpo vigente (notas `nota_pie` + avisos «Téngase»).
    - `dropped`: aparato editorial citado en blockquote (marcador + redacción/tabla DEROGADA) —
      fuera del cuerpo y **fuera de notes** (solapa el vigente y dispararía falsos `note_leak`; el
      cambio ya consta en las notas con su Ref. al BOE).

    Función pura y sin efectos: el parser y la auditoría independiente la comparten (igual que
    `resolve_current_version`), de modo que la invariante estructural de la auditoría reaplica
    exactamente la misma regla sobre el raw que la usada al persistir.
    """
    kept: list[dict] = []
    notes: list[dict] = []
    dropped: list[dict] = []
    for el in version.iter():
        if el.tag == "p":
            # Los <p> dentro de un <table> son celdas (forma A): su texto entra por la fila
            # linealizada (rama `table`); se saltan aquí para no duplicarlos.
            if next(el.iterancestors("table"), None) is not None:
                continue
            css = el.get("class") or ""
            text = _element_text(el)
            if not text:
                continue
            in_blockquote = next(el.iterancestors("blockquote"), None) is not None
            # El aviso «Téngase…» puede venir como nota a pie con marcador inicial ("(*) ", "* "),
            # así que se saltan esos prefijos antes de comparar (sigue siendo señal editorial).
            stripped = text.lstrip("(*)[]·•—-. \t").lower()
            if css in NOTE_CLASSES or stripped.startswith(_EDITORIAL_NOTE_PREFIX):
                notes.append({"text": text, "target_norm_id": _extract_norm_id(text)})
            elif in_blockquote:
                dropped.append({"class": css, "text": text})
            else:
                kept.append({"order": len(kept) + 1, "class": css, "text": text})
        elif el.tag == "table":
            # Toda tabla (forma A o B) se linealiza por fila en su posición de documento: va a
            # `dropped` si está en blockquote (editorial), si no a `kept`. Sus <p>/<td>/<th>
            # internos se ignoran después (no duplican).
            in_blockquote = next(el.iterancestors("blockquote"), None) is not None
            for css, text in _linearize_table_rows(el):
                if in_blockquote:
                    dropped.append({"class": css, "text": text})
                else:
                    kept.append({"order": len(kept) + 1, "class": css, "text": text})
    return kept, notes, dropped


def _parse_version_paragraphs(version: etree._Element) -> tuple[list[dict], list[dict]]:
    """Cuerpo normativo + notas de una versión (proyección de `classify_version_paragraphs`).

    Descarta el aparato editorial citado en blockquote y **emite WARNING** si lo descartado incluye
    clase estructural/tabla/artículo: posible texto vigente mal envuelto que, a escala, sería una
    pérdida silenciosa difícil de detectar. La auditoría lo recoge además como anomalía durable.
    """
    kept, notes, dropped = classify_version_paragraphs(version)
    for d in dropped:
        if is_suspicious_blockquote_class(d["class"]):
            logger.warning(
                "<p> estructural/tabla {!r} dentro de <blockquote> (¿modelo citado?): {!r}",
                d["class"],
                d["text"][:60],
            )
    return kept, notes


def _full_title(block_type: str, paragraphs: list[dict]) -> str | None:
    """Cabecera legible, acotada al prefijo estructural inicial del bloque.

    Si el primer párrafo es `articulo`, se devuelve (preceptos y disposiciones). Si no,
    solo para `encabezado` se compone desde el run inicial de párrafos estructurales
    (nunca desde el cuerpo del bloque).
    """
    if not paragraphs:
        return None
    if paragraphs[0].get("class") == "articulo":
        return paragraphs[0]["text"]
    if block_type != "encabezado":
        return None

    prefix: dict[str, str] = {}
    for p in paragraphs:
        if p.get("class") in STRUCTURAL_LABEL_CLASSES:
            prefix.setdefault(p["class"], p["text"])
        else:
            break
    for num, tit in (
        ("libro_num", "libro_tit"),
        ("titulo_num", "titulo_tit"),
        ("capitulo_num", "capitulo_tit"),
        ("anexo_num", "anexo_tit"),
    ):
        if num in prefix:
            parts = [prefix[num], prefix.get(tit, "")]
            return clean_text(". ".join(x for x in parts if x))
    for single in ("seccion", "subseccion", "libro", "titulo", "capitulo", "anexo"):
        if single in prefix:
            return prefix[single]
    return None


def _block_semantics(
    block_type: str,
    paragraphs: list[dict],
    contains_image: bool,
    raw_has_table: bool,
) -> dict:
    """Calcula rol semántico, indexabilidad de cuerpo y flags de contenido del bloque."""
    table_classes = any(is_table_class(p.get("class") or "") for p in paragraphs)
    contains_table = raw_has_table or table_classes
    table_text_available = table_classes  # celdas linealizadas (forma A <p> o forma B por fila)

    has_anexo_num = any(p.get("class") == "anexo_num" for p in paragraphs)
    singular_annex = any(
        p.get("class") == "anexo" and (p.get("text") or "").strip().upper().startswith("ANEXO")
        for p in paragraphs
    )
    is_annex = has_anexo_num or singular_annex
    annex_is_local = singular_annex and not has_anexo_num
    has_body = heading_has_retrievable_body(paragraphs)

    if block_type == "precepto":
        role = "precept"
    elif block_type == "preambulo":
        role = "preamble"
    elif block_type == "firma":
        role = "signature"
    elif block_type == "nota_inicial":
        role = "initial_note"
    elif block_type == "encabezado":
        if has_body and is_annex:
            role = "annex"
        elif has_body:
            role = "content_heading"
        else:
            role = "structural_heading"
    else:
        role = block_type

    return {
        "semantic_role": role,
        "has_retrievable_body": has_body,
        "is_annex": is_annex,
        "contains_table": contains_table,
        "table_text_available": table_text_available,
        "contains_image": contains_image,
        "_annex_is_local": annex_is_local,
    }


def parse_text_blocks(
    data: etree._Element,
    index_dates: dict[str, str | None] | None = None,
    index_dates_invalid: set[str] | None = None,
) -> tuple[dict[str, dict], list[str], list[str]]:
    """Mapea `data/texto/bloque[]`. Devuelve (bloques_por_id, orden_ids, warnings).

    `index_dates` mapea `block_id -> fecha_actualizacion` (ISO) de `indice.xml`; es el único
    criterio de vigencia. Los bloques cuya versión vigente no se resuelve de forma exacta y
    única pasan a **cuarentena** (`latest_version=null`, sin párrafos), conservando `versions[]`
    para diagnóstico. `index_dates_invalid` lista bloques cuya fecha de índice venía pero no era
    normalizable (estado `invalid_date`).
    """
    texto = data.find("texto")
    if texto is None:
        raise ParsingError("Falta el nodo 'texto' en el XML de texto")

    index_dates = index_dates or {}
    index_dates_invalid = index_dates_invalid or set()

    blocks: dict[str, dict] = {}
    order_ids: list[str] = []
    warnings: list[str] = []

    for bloque in texto.findall("bloque"):
        block_id = bloque.get("id")
        block_type = bloque.get("tipo")
        title_attr = bloque.get("titulo")
        block_title = clean_text(title_attr) if title_attr else None
        version_elements = bloque.findall("version")

        versions = []
        for v in version_elements:
            versions.append(
                {
                    "source_norm_id": v.get("id_norma"),
                    "publication_date": normalize_date(v.get("fecha_publicacion")),
                    "validity_date": normalize_date(v.get("fecha_vigencia")),
                    "is_latest": False,
                }
            )

        # Resolución temporal estricta: la vigencia la decide el índice, no el orden XML.
        temporal = resolve_current_version(versions, index_dates.get(block_id))
        status = temporal["status"]
        if status == "missing_index_date" and block_id in index_dates_invalid:
            temporal["status"] = status = "invalid_date"
            temporal["warnings"] = ["invalid_index_date"]
        raw_pubs = [v.get("fecha_publicacion") for v in version_elements]
        if status != "resolved" and any(rp and normalize_date(rp) is None for rp in raw_pubs):
            temporal["warnings"] = [*temporal["warnings"], "invalid_publication_date"]
        if version_elements and any(not rp for rp in raw_pubs):
            temporal["warnings"] = [*temporal["warnings"], "missing_publication_date"]

        # Aviso informativo de orden XML no cronológico (no es criterio de vigencia).
        pub_dates = [v["publication_date"] for v in versions]
        if version_elements and all(pub_dates) and pub_dates != sorted(pub_dates):
            warnings.append(f"{block_id}: orden XML de versiones no cronológico")
        if status != "resolved":
            warnings.append(f"{block_id}: cuarentena temporal ({status})")

        sel = temporal["selected_version_index"]
        paragraphs: list[dict] = []
        notes: list[dict] = []
        latest_version = None
        if sel is not None:
            versions[sel]["is_latest"] = True
            paragraphs, notes = _parse_version_paragraphs(version_elements[sel])
            chosen = versions[sel]
            latest_version = {
                "source_norm_id": chosen["source_norm_id"],
                "publication_date": chosen["publication_date"],
                "validity_date": chosen["validity_date"],
                "text": "\n".join(p["text"] for p in paragraphs),
                "paragraphs": paragraphs,
                "modification_notes": notes,
            }

        semantics = _block_semantics(
            block_type,
            paragraphs,
            contains_image=bloque.find(".//img") is not None,
            raw_has_table=bloque.find(".//table") is not None,
        )
        without_content = is_without_content(paragraphs)
        blocks[block_id] = {
            "block_id": block_id,
            "block_type": block_type,
            "block_title": block_title,
            "full_title": _full_title(block_type, paragraphs),
            "versions": versions,
            "latest_version": latest_version,
            "temporal_resolution": temporal,
            "temporal_quarantined": sel is None,
            "content_status": "without_content" if without_content else "present",
            "is_without_content": without_content,
            **semantics,
        }
        order_ids.append(block_id)

    return blocks, order_ids, warnings


# --------------------------------------------------------------------------- #
# Jerarquía y retrieval
# --------------------------------------------------------------------------- #


def _update_hierarchy(state: dict, text_block: dict) -> None:
    """Actualiza el estado jerárquico (6 niveles) con un bloque `encabezado`.

    Solo usa clases inequívocas (`*_num`, `seccion`, `subseccion`). Cada nivel reinicia los
    inferiores y `annex`; un `anexo_num` reinicia toda la parte articulada (mutuamente
    excluyentes).
    """
    if not text_block or text_block.get("block_type") != "encabezado":
        return
    latest = text_block.get("latest_version")
    if not latest:  # bloque en cuarentena temporal: no aporta jerarquía
        return
    classes = {p["class"]: p["text"] for p in latest["paragraphs"]}
    if "libro_num" in classes:
        state.update(
            book=classes["libro_num"],
            title=None,
            chapter=None,
            section=None,
            subsection=None,
            annex=None,
        )
    elif "titulo_num" in classes:
        state.update(
            title=classes["titulo_num"], chapter=None, section=None, subsection=None, annex=None
        )
    elif "capitulo_num" in classes:
        state.update(chapter=classes["capitulo_num"], section=None, subsection=None, annex=None)
    elif "seccion" in classes:
        state.update(section=classes["seccion"], subsection=None, annex=None)
    elif "subseccion" in classes:
        state.update(subsection=classes["subseccion"], annex=None)
    elif "anexo_num" in classes:
        state.update(
            annex=classes["anexo_num"],
            book=None,
            title=None,
            chapter=None,
            section=None,
            subsection=None,
        )


def _lower_first(value: str) -> str:
    """Minúscula la inicial solo si la palabra es title-case (evita TÍTULO/CAPÍTULO)."""
    if len(value) >= 2 and value[1].islower():
        return value[0].lower() + value[1:]
    return value


def build_block_descriptor_fields(
    document_id: str,
    html_url: str | None,
    short_title: str | None,
    block: dict,
) -> dict:
    """Proyección de descriptor de un bloque: indexabilidad y cita.

    Responsabilidad exacta: decide `indexable`/`excluded_reason` (por contenido + tipo +
    cuarentena) y compone `citation_label`/`source_url`. **No** construye `retrieval_text`: ese
    campo lo genera y persiste exclusivamente el chunker (`src/preprocessing/chunker.py`).
    """
    block_id = block["block_id"]
    block_title = block.get("block_title")
    latest = block.get("latest_version") or {}
    text = latest.get("text") or ""
    label_base = short_title or document_id

    if block_title:
        citation_label = f"{label_base}, {_lower_first(block_title)}"
    else:
        citation_label = label_base

    source_url = f"{html_url}#{block_id}" if html_url else None
    quarantined = block.get("temporal_quarantined", False)
    indexable = (
        block.get("has_retrievable_body", False)
        and block.get("block_type") not in EXCLUDED_TYPES
        and bool(text)
        and not quarantined
    )

    excluded_reason = None
    if quarantined:
        status = (block.get("temporal_resolution") or {}).get("status")
        excluded_reason = f"temporal_quarantine:{status}"
    elif not indexable:
        if block.get("block_type") in EXCLUDED_TYPES:
            excluded_reason = f"excluded_type:{block.get('block_type')}"
        elif not block.get("has_retrievable_body", False):
            excluded_reason = "no_retrievable_body"

    return {
        "indexable": indexable,
        "citation_label": citation_label,
        "source_url": source_url,
        "excluded_reason": excluded_reason,
    }


# --------------------------------------------------------------------------- #
# Ensamblado
# --------------------------------------------------------------------------- #


def _build_normalized_intermediate(norm_id: str, raw_dir: Path, manifest_path: Path) -> dict:
    """Representación intermedia **privada y neutral** de una norma (no se persiste).

    Estructura rica por bloque (texto vigente, versiones, resolución temporal, descriptor) de la
    que `build_processed_bundle` deriva los contratos persistidos v2 (document + history +
    parents). No es un contrato: es el modelo de trabajo del parser.
    """
    raw_dir = Path(raw_dir)
    manifest_path = Path(manifest_path)
    norm_dir = raw_dir / norm_id

    paths = {name: norm_dir / name for name in REQUIRED_RAW_FILES}

    data_metadatos = validate_response(load_xml(paths["metadatos.xml"]), paths["metadatos.xml"])
    data_indice = validate_response(load_xml(paths["indice.xml"]), paths["indice.xml"])
    data_texto = validate_response(load_xml(paths["texto.xml"]), paths["texto.xml"])

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    metadata = parse_metadata(data_metadatos)
    # `analisis.xml` es opcional: si no está, el análisis queda vacío.
    if paths["analisis.xml"].is_file():
        data_analisis = validate_response(load_xml(paths["analisis.xml"]), paths["analisis.xml"])
        analysis = parse_analysis(data_analisis)
    else:
        analysis = {"subjects": [], "notes": [], "references": {"previous": [], "next": []}}
    index_blocks = parse_index(data_indice)
    index_dates = {b["block_id"]: b["index_last_update_date"] for b in index_blocks}
    index_dates_invalid = {
        b["block_id"]
        for b in index_blocks
        if b["index_last_update_date"] is None and b.get("index_last_update_date_raw")
    }
    text_blocks, text_order, warnings = parse_text_blocks(
        data_texto, index_dates, index_dates_invalid
    )

    document_id = metadata.get("identifier") or norm_id
    html_url = metadata.get("html_url")
    short_title = metadata.get("short_title")

    index_ids = [b["block_id"] for b in index_blocks]
    index_id_set = set(index_ids)
    unmatched_text_blocks = [bid for bid in text_order if bid not in index_id_set]

    # Orden documental: primero el índice; al final, bloques de texto sin entrada en índice.
    ordered_index = list(index_blocks)
    next_order = len(ordered_index)
    for bid in unmatched_text_blocks:
        ordered_index.append({"block_id": bid, "order": next_order})
        next_order += 1

    hierarchy_state = {
        "book": None,
        "title": None,
        "chapter": None,
        "section": None,
        "subsection": None,
        "annex": None,
    }
    blocks: list[dict] = []
    for index_entry in ordered_index:
        block_id = index_entry["block_id"]
        text_block = text_blocks.get(block_id)
        _update_hierarchy(hierarchy_state, text_block)

        hierarchy = dict(hierarchy_state)
        # Anexo singular (clase `anexo`, no `anexo_num`): contexto local no propagado.
        if text_block and text_block.get("_annex_is_local"):
            hierarchy["annex"] = text_block.get("full_title") or text_block.get("block_title")

        block = {
            "block_id": block_id,
            "parent_id": f"{document_id}__{block_id}",
            "order": index_entry["order"],
            "block_type": text_block["block_type"] if text_block else None,
            "block_title": text_block["block_title"] if text_block else None,
            "full_title": text_block["full_title"] if text_block else None,
            "semantic_role": text_block["semantic_role"] if text_block else None,
            "has_retrievable_body": text_block["has_retrievable_body"] if text_block else False,
            "is_annex": text_block["is_annex"] if text_block else False,
            "contains_table": text_block["contains_table"] if text_block else False,
            "table_text_available": text_block["table_text_available"] if text_block else False,
            "contains_image": text_block["contains_image"] if text_block else False,
            "content_status": text_block["content_status"] if text_block else "present",
            "is_without_content": text_block["is_without_content"] if text_block else False,
            # Bloque presente en índice pero ausente en texto: sin versiones → unresolved.
            "temporal_resolution": (
                text_block["temporal_resolution"]
                if text_block
                else {"status": "unresolved", "warnings": ["block_in_index_only"]}
            ),
            "temporal_quarantined": text_block["temporal_quarantined"] if text_block else True,
            "index_title": index_entry.get("index_title"),
            "index_url": index_entry.get("index_url"),
            "index_last_update_date": index_entry.get("index_last_update_date"),
            "hierarchy": hierarchy,
            "versions": text_block["versions"] if text_block else [],
            "latest_version": text_block["latest_version"] if text_block else None,
        }
        block["descriptor_fields"] = build_block_descriptor_fields(
            document_id, html_url, short_title, block
        )
        blocks.append(block)

    return {
        "document_id": document_id,
        "source": {
            "name": SOURCE_NAME,
            "base_url": manifest.get("base_url"),
            "downloaded_at": manifest.get("downloaded_at"),
            "manifest_ref": _relative_manifest_ref(manifest_path),
        },
        "metadata": metadata,
        "analysis": analysis,
        "blocks": blocks,
    }


def _relative_manifest_ref(manifest_path: Path) -> str:
    """Ruta relativa y estable al manifest (independiente del equipo)."""
    manifest_path = Path(manifest_path)
    try:
        return manifest_path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return f"data/manifests/{manifest_path.name}"


def _generation_meta() -> dict:
    """Metadatos mínimos de generación (sin conteos derivados ni readiness)."""
    return {"generated_at": datetime.now(UTC).isoformat(), "generator": GENERATOR}


# --------------------------------------------------------------------------- #
# Contratos v2: descriptor + history + parents (derivados del ensamblado)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProcessedNormBundle:
    """Bundle tipado en memoria de los tres artefactos persistidos de una norma."""

    document: dict  # boe_legal_document_v2 (descriptor)
    history: dict  # boe_legal_history_v2
    parents: dict  # boe_legal_parents_v2


def _has_current_text(block: dict) -> bool:
    """True si el bloque está resuelto y tiene texto vigente no vacío (→ tiene parent)."""
    lv = block.get("latest_version") or {}
    return not block.get("temporal_quarantined", False) and bool((lv.get("text") or "").strip())


def _build_document_v2(intermediate: dict) -> dict:
    """Descriptor legible: bloques sin texto pesado; propietario de indexable/flags/cita."""
    blocks = []
    for b in intermediate["blocks"]:
        desc = b.get("descriptor_fields") or {}
        temporal = b.get("temporal_resolution") or {}
        blocks.append(
            {
                "block_id": b["block_id"],
                "parent_id": b["parent_id"] if _has_current_text(b) else None,
                "order": b["order"],
                "block_type": b.get("block_type"),
                "block_title": b.get("block_title"),
                "full_title": b.get("full_title"),
                "semantic_role": b.get("semantic_role"),
                "has_retrievable_body": b.get("has_retrievable_body", False),
                "is_annex": b.get("is_annex", False),
                "contains_table": b.get("contains_table", False),
                "table_text_available": b.get("table_text_available", False),
                "contains_image": b.get("contains_image", False),
                "content_status": b.get("content_status", "present"),
                "is_without_content": b.get("is_without_content", False),
                "temporal_status": (temporal.get("status") if temporal else None) or "unknown",
                "hierarchy": b.get("hierarchy") or {},
                "indexable": desc.get("indexable", False),
                "excluded_reason": desc.get("excluded_reason"),
                "citation": {
                    "label": desc.get("citation_label"),
                    "url": desc.get("source_url"),
                },
            }
        )
    return {
        "schema_version": DOCUMENT_SCHEMA_VERSION,
        "document_id": intermediate["document_id"],
        "source": {
            "name": intermediate["source"]["name"],
            "base_url": intermediate["source"].get("base_url"),
            "downloaded_at": intermediate["source"].get("downloaded_at"),
            "manifest_ref": intermediate["source"]["manifest_ref"],
        },
        "metadata": intermediate["metadata"],
        "analysis": intermediate["analysis"],
        "blocks": blocks,
        "generation_meta": _generation_meta(),
    }


def _build_history(intermediate: dict) -> dict:
    """Historial temporal: un registro por CADA block_id (incl. monoversión y cuarentena).

    Propietario único de versiones, notas de modificación y resolución temporal. Los warnings
    temporales viven en `temporal_resolution.warnings` (no se duplican a nivel de bloque).
    """
    records = []
    for b in intermediate["blocks"]:
        temporal = b.get("temporal_resolution") or {}
        lv = b.get("latest_version") or {}
        sel_pub = temporal.get("selected_publication_date")
        versions = [
            {
                "source_norm_id": v.get("source_norm_id"),
                "publication_date": v.get("publication_date"),
                "validity_date": v.get("validity_date"),
                "is_current": bool(v.get("is_latest")),
            }
            for v in (b.get("versions") or [])
        ]
        records.append(
            {
                "block_id": b["block_id"],
                "versions": versions,
                "modification_notes": lv.get("modification_notes", []),
                "temporal_resolution": {
                    "status": temporal.get("status"),
                    "selection_method": temporal.get("selection_method"),
                    "index_last_update_date": temporal.get("index_last_update_date"),
                    "selected_version_index": temporal.get("selected_version_index"),
                    "selected_publication_date": sel_pub,
                    "selected_source_norm_id": temporal.get("selected_source_norm_id"),
                    "candidate_versions": temporal.get("candidate_versions", []),
                    "max_publication_date": temporal.get("max_publication_date"),
                    "warnings": temporal.get("warnings", []),
                },
                "temporal_quarantined": b.get("temporal_quarantined", False),
                "index_title": b.get("index_title"),
                "index_url": b.get("index_url"),
                "index_last_update_date": b.get("index_last_update_date"),
                "index_last_update_date_raw": b.get("index_last_update_date_raw"),
            }
        )
    return {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "document_id": intermediate["document_id"],
        "blocks": records,
        "generation_meta": _generation_meta(),
    }


def _build_parents(intermediate: dict) -> dict:
    """Propietario único del texto vigente: un registro por bloque resuelto con texto no vacío."""
    records = []
    for b in intermediate["blocks"]:
        if not _has_current_text(b):
            continue
        lv = b.get("latest_version") or {}
        desc = b.get("descriptor_fields") or {}
        records.append(
            {
                "parent_id": b["parent_id"],
                "document_id": intermediate["document_id"],
                "block_id": b["block_id"],
                "order": b["order"],
                "block_type": b.get("block_type"),
                "title": b.get("block_title"),
                "full_title": b.get("full_title"),
                "semantic_role": b.get("semantic_role"),
                "text": lv.get("text", ""),
                "paragraphs": lv.get("paragraphs", []),
                "hierarchy": b.get("hierarchy") or {},
                "citation": {
                    "label": desc.get("citation_label"),
                    "url": desc.get("source_url"),
                },
                "current_version": {
                    "source_norm_id": lv.get("source_norm_id"),
                    "publication_date": lv.get("publication_date"),
                    "validity_date": lv.get("validity_date"),
                },
                "is_annex": b.get("is_annex", False),
                "contains_table": b.get("contains_table", False),
                "table_text_available": b.get("table_text_available", False),
                "contains_image": b.get("contains_image", False),
                "content_status": b.get("content_status", "present"),
                "is_without_content": b.get("is_without_content", False),
                # `modification_notes` NO se duplican aquí: viven en history (se hidratan por join).
            }
        )
    return {
        "schema_version": PARENTS_SCHEMA_VERSION,
        "document_id": intermediate["document_id"],
        "parents": records,
        "generation_meta": _generation_meta(),
    }


def build_processed_bundle(norm_id: str, raw_dir: Path, manifest_path: Path) -> ProcessedNormBundle:
    """Ensambla la norma y deriva los tres contratos persistidos v2 (en memoria, sin escribir).

    Valida cada artefacto contra su modelo Pydantic antes de devolverlo (fail-fast).
    """
    intermediate = _build_normalized_intermediate(norm_id, raw_dir, manifest_path)
    document = _build_document_v2(intermediate)
    history = _build_history(intermediate)
    parents = _build_parents(intermediate)
    # Validación local por artefacto (contratos = fuente única de verdad).
    DocumentV2.model_validate(document)
    HistoryV2.model_validate(history)
    ParentsV2.model_validate(parents)
    return ProcessedNormBundle(document=document, history=history, parents=parents)


def _dump_json(obj: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_processed_bundle(
    bundle: ProcessedNormBundle,
    documents_dir: Path,
    histories_dir: Path,
    parents_dir: Path,
) -> dict[str, Path]:
    """Persiste el bundle: document_v2, history_v2 y parents_v2 (capa de persistencia separada)."""
    did = bundle.document["document_id"]
    return {
        "document": _dump_json(bundle.document, Path(documents_dir) / f"{did}.json"),
        "history": _dump_json(bundle.history, Path(histories_dir) / f"{did}.json"),
        "parents": _dump_json(bundle.parents, Path(parents_dir) / f"{did}.json"),
    }
