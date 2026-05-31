"""Tests unitarios del parser BOE v0 (sin red, con XML mínimos en tmp_path).

No se usa la norma completa como fixture: cada test escribe XML mínimos que ejercitan
una característica concreta del contrato `boe_legal_document_v1`.
"""

import json
from pathlib import Path

import pytest
from lxml import etree

from src.boe.parser import (
    SCHEMA_VERSION,
    _block_semantics,
    _full_title,
    _update_hierarchy,
    build_retrieval,
    is_without_content,
    load_xml,
    normalize_date,
    normalize_datetime,
    parse_boe_document,
    parse_index,
    parse_metadata,
    parse_text_blocks,
    resolve_current_version,
    save_processed_document,
    validate_response,
)
from src.core.exceptions import ParsingError

NORM_ID = "BOE-A-2015-10565"

# --- XML mínimos -------------------------------------------------------------

METADATOS_XML = """<?xml version="1.0" encoding="utf-8"?>
<response>
  <status><code>200</code><text>ok</text></status>
  <data>
    <metadatos>
      <fecha_actualizacion>20260520T070602Z</fecha_actualizacion>
      <identificador>BOE-A-2015-10565</identificador>
      <ambito codigo="1">Estatal</ambito>
      <departamento codigo="7723">Jefatura del Estado</departamento>
      <rango codigo="1300">Ley</rango>
      <fecha_disposicion>20151001</fecha_disposicion>
      <numero_oficial>39/2015</numero_oficial>
      <titulo>Ley 39/2015, de 1 de octubre, del Procedimiento Administrativo Común.</titulo>
      <fecha_publicacion>20151002</fecha_publicacion>
      <fecha_vigencia>20161002</fecha_vigencia>
      <estatus_derogacion>N</estatus_derogacion>
      <estatus_anulacion>N</estatus_anulacion>
      <vigencia_agotada>N</vigencia_agotada>
      <estado_consolidacion codigo="3">Finalizado</estado_consolidacion>
      <url_eli>https://www.boe.es/eli/es/l/2015/10/01/39</url_eli>
      <url_html_consolidada>https://www.boe.es/buscar/act.php?id=BOE-A-2015-10565</url_html_consolidada>
    </metadatos>
  </data>
</response>
"""

ANALISIS_XML = """<?xml version="1.0" encoding="utf-8"?>
<response>
  <status><code>200</code><text>ok</text></status>
  <data>
    <analisis>
      <materias>
        <materia codigo="5703">Procedimiento administrativo</materia>
      </materias>
      <notas>
        <nota>Entrada en vigor el 2 de octubre de 2016.</nota>
      </notas>
      <referencias>
        <anteriores>
          <anterior>
            <id_norma>BOE-A-1992-26318</id_norma>
            <relacion codigo="210">DEROGA</relacion>
            <texto>la Ley 30/1992, de 26 de noviembre</texto>
          </anterior>
        </anteriores>
        <posteriores>
          <posterior>
            <id_norma>BOE-A-2022-11589</id_norma>
            <relacion codigo="270">SE MODIFICA</relacion>
            <texto>art. 77, por Ley 15/2022</texto>
          </posterior>
        </posteriores>
      </referencias>
    </analisis>
  </data>
</response>
"""

INDICE_XML = """<?xml version="1.0" encoding="utf-8"?>
<response>
  <status><code>200</code><text>ok</text></status>
  <data>
    <bloque>
      <id>ti</id>
      <titulo>TÍTULO I</titulo>
      <fecha_actualizacion>20151002</fecha_actualizacion>
      <url>https://www.boe.es/datosabiertos/api/legislacion-consolidada/id/BOE-A-2015-10565/texto/bloque/ti</url>
    </bloque>
    <bloque>
      <id>a9</id>
      <titulo>Artículo 9</titulo>
      <fecha_actualizacion>20220629</fecha_actualizacion>
      <url>https://www.boe.es/datosabiertos/api/legislacion-consolidada/id/BOE-A-2015-10565/texto/bloque/a9</url>
    </bloque>
  </data>
</response>
"""

TEXTO_XML = """<?xml version="1.0" encoding="utf-8"?>
<response>
  <status><code>200</code><text>ok</text></status>
  <data>
    <texto>
      <bloque id="ti" tipo="encabezado" titulo="TÍTULO I">
        <version id_norma="BOE-A-2015-10565" fecha_publicacion="20151002" fecha_vigencia="20161002">
          <p class="titulo_num">TÍTULO I</p>
          <p class="titulo_tit">De los interesados en el procedimiento</p>
        </version>
      </bloque>
      <bloque id="a9" tipo="precepto" titulo="Artículo 9">
        <version id_norma="BOE-A-2015-10565" fecha_publicacion="20151002" fecha_vigencia="20161002">
          <p class="articulo">Artículo 9. Sistemas de identificación.</p>
          <p class="parrafo">1. Texto original del artículo.</p>
        </version>
        <version id_norma="BOE-A-2022-10757" fecha_publicacion="20220629" fecha_vigencia="20220630">
          <p class="articulo">Artículo 9. Sistemas de identificación.</p>
          <p class="parrafo">1. Texto vigente con <strong>énfasis</strong> y referencia.</p>
          <blockquote>
            <p class="nota_pie">Se modifica por la Ley 11/2022.
              <a class="refPost">Ref. BOE-A-2022-10757#df</a></p>
          </blockquote>
        </version>
      </bloque>
    </texto>
  </data>
</response>
"""

MANIFEST = {
    "norm_id": NORM_ID,
    "source": "BOE legislación consolidada",
    "base_url": "https://www.boe.es/datosabiertos/api",
    "downloaded_at": "2026-05-28T22:38:10.408125+00:00",
    "files": [],
}


def write_raw(
    tmp_path: Path, *, texto: str = TEXTO_XML, indice: str = INDICE_XML
) -> tuple[Path, Path]:
    """Escribe el raw mínimo de la norma y el manifest. Devuelve (raw_dir, manifest_path)."""
    raw_dir = tmp_path / "raw"
    norm_dir = raw_dir / NORM_ID
    norm_dir.mkdir(parents=True)
    (norm_dir / "metadatos.xml").write_text(METADATOS_XML, encoding="utf-8")
    (norm_dir / "analisis.xml").write_text(ANALISIS_XML, encoding="utf-8")
    (norm_dir / "indice.xml").write_text(indice, encoding="utf-8")
    (norm_dir / "texto.xml").write_text(texto, encoding="utf-8")
    manifest_path = tmp_path / "manifests" / f"{NORM_ID}.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(MANIFEST), encoding="utf-8")
    return raw_dir, manifest_path


# --- 1-2. validate_response --------------------------------------------------


def test_validate_response_returns_data_on_200() -> None:
    root = etree.fromstring(
        b"<response><status><code>200</code></status><data><x/></data></response>"
    )
    data = validate_response(root, Path("dummy.xml"))
    assert data.tag == "data"


def test_validate_response_raises_on_non_200() -> None:
    root = etree.fromstring(b"<response><status><code>404</code></status><data/></response>")
    with pytest.raises(ParsingError):
        validate_response(root, Path("dummy.xml"))


def test_load_xml_missing_file_raises_parsing_error(tmp_path: Path) -> None:
    with pytest.raises(ParsingError):
        load_xml(tmp_path / "no-existe.xml")


# --- 3-4. normalización de fechas -------------------------------------------


def test_normalize_date() -> None:
    assert normalize_date("20151002") == "2015-10-02"
    assert normalize_date("") is None
    assert normalize_date("2015-10-02") is None


def test_normalize_datetime() -> None:
    assert normalize_datetime("20260520T070602Z") == "2026-05-20T07:06:02Z"
    assert normalize_datetime("bad") is None


# --- 5. metadatos ------------------------------------------------------------


def test_parse_metadata_codes_and_short_title() -> None:
    data = validate_response(etree.fromstring(METADATOS_XML.encode()), Path("metadatos.xml"))
    meta = parse_metadata(data)
    assert meta["rank"] == {"code": "1300", "label": "Ley"}
    assert meta["scope"] == {"code": "1", "label": "Estatal"}
    assert meta["consolidation_status"]["code"] == "3"
    assert meta["short_title"] == "Ley 39/2015"
    assert meta["publication_date"] == "2015-10-02"
    assert meta["last_update_datetime"] == "2026-05-20T07:06:02Z"


# --- 6. índice ---------------------------------------------------------------


def test_parse_index_order() -> None:
    data = validate_response(etree.fromstring(INDICE_XML.encode()), Path("indice.xml"))
    blocks = parse_index(data)
    assert [b["block_id"] for b in blocks] == ["ti", "a9"]
    assert [b["order"] for b in blocks] == [0, 1]
    assert blocks[1]["index_last_update_date"] == "2022-06-29"


# --- 7. texto: versiones, latest, notas -------------------------------------


INDEX_DATES = {"ti": "2015-10-02", "a9": "2022-06-29"}


def test_parse_text_blocks_versions_and_notes() -> None:
    data = validate_response(etree.fromstring(TEXTO_XML.encode()), Path("texto.xml"))
    blocks, order_ids, warnings = parse_text_blocks(data, INDEX_DATES)

    assert order_ids == ["ti", "a9"]
    a9 = blocks["a9"]
    assert len(a9["versions"]) == 2
    assert [v["is_latest"] for v in a9["versions"]] == [False, True]
    assert a9["latest_version"]["source_norm_id"] == "BOE-A-2022-10757"

    # nota_pie capturada como modification_note (con target_norm_id) y fuera del texto.
    notes = a9["latest_version"]["modification_notes"]
    assert len(notes) == 1
    assert notes[0]["target_norm_id"] == "BOE-A-2022-10757"
    assert "Se modifica" not in a9["latest_version"]["text"]
    # El texto vigente conserva el contenido inline de <strong> sin la etiqueta.
    assert "énfasis" in a9["latest_version"]["text"]
    assert "<strong>" not in a9["latest_version"]["text"]
    assert warnings == []


def test_build_retrieval_excludes_notes_and_sets_indexable() -> None:
    data = validate_response(etree.fromstring(TEXTO_XML.encode()), Path("texto.xml"))
    blocks, _, _ = parse_text_blocks(data, INDEX_DATES)

    a9 = blocks["a9"]
    a9["block_id"] = "a9"
    html_url = "https://www.boe.es/buscar/act.php?id=BOE-A-2015-10565"
    retrieval = build_retrieval(NORM_ID, html_url, "Ley 39/2015", a9)
    assert retrieval["indexable"] is True
    assert retrieval["source_url"].endswith("#a9")
    assert retrieval["citation_label"] == "Ley 39/2015, artículo 9"
    assert "Se modifica" not in retrieval["retrieval_text"]

    ti = blocks["ti"]
    ti["block_id"] = "ti"
    retrieval_ti = build_retrieval(NORM_ID, "https://x", "Ley 39/2015", ti)
    assert retrieval_ti["indexable"] is False  # encabezado no indexable


# --- 8. quality_checks: conteos y unmatched ---------------------------------


def test_quality_checks_counts_and_unmatched(tmp_path: Path) -> None:
    # Índice con un id extra ('zzz') que no existe en texto.
    indice_extra = INDICE_XML.replace(
        "</data>",
        "<bloque><id>zzz</id><titulo>Extra</titulo>"
        "<fecha_actualizacion>20151002</fecha_actualizacion><url>http://x</url></bloque></data>",
    )
    raw_dir, manifest_path = write_raw(tmp_path, indice=indice_extra)
    document = parse_boe_document(NORM_ID, raw_dir, manifest_path)

    checks = document["quality_checks"]
    assert checks["index_blocks_count"] == 3
    assert checks["text_blocks_count"] == 2
    assert checks["unmatched_index_blocks"] == ["zzz"]
    assert checks["unmatched_text_blocks"] == []


# --- 9. persistencia ---------------------------------------------------------


def test_save_processed_document_writes_valid_json(tmp_path: Path) -> None:
    document = {"document_id": NORM_ID, "schema_version": SCHEMA_VERSION, "blocks": []}
    out_path = save_processed_document(document, tmp_path / "processed")
    assert out_path.name == f"{NORM_ID}.json"
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["document_id"] == NORM_ID


# --- 10. integración local ---------------------------------------------------


def test_parse_boe_document_integration(tmp_path: Path) -> None:
    raw_dir, manifest_path = write_raw(tmp_path)
    document = parse_boe_document(NORM_ID, raw_dir, manifest_path)

    assert document["schema_version"] == SCHEMA_VERSION
    assert document["document_id"] == NORM_ID
    assert document["source"]["raw_manifest_path"].endswith(f"{NORM_ID}.json")
    assert document["quality_checks"]["raw_files_present"] is True
    assert document["quality_checks"]["metadata_ok"] is True

    blocks_by_id = {b["block_id"]: b for b in document["blocks"]}
    # Jerarquía: el artículo hereda el TÍTULO I del encabezado previo.
    assert blocks_by_id["a9"]["hierarchy"]["title"] == "TÍTULO I"
    assert blocks_by_id["a9"]["parent_id"] == f"{NORM_ID}__a9"
    # Hay un bloque con múltiples versiones.
    assert any(len(b["versions"]) > 1 for b in document["blocks"])
    # analysis parseado.
    assert document["analysis"]["references"]["previous"][0]["relation"]["label"] == "DEROGA"


def test_parse_boe_document_without_analisis(tmp_path: Path) -> None:
    # `analisis.xml` es opcional: si falta, el documento se parsea con analysis vacío.
    raw_dir, manifest_path = write_raw(tmp_path)
    (raw_dir / NORM_ID / "analisis.xml").unlink()

    document = parse_boe_document(NORM_ID, raw_dir, manifest_path)

    assert document["analysis"] == {
        "subjects": [],
        "notes": [],
        "references": {"previous": [], "next": []},
    }
    # Sigue habiendo bloques y los obligatorios presentes.
    assert document["quality_checks"]["raw_files_present"] is True
    assert len(document["blocks"]) > 0


# --- correcciones pre-embeddings: indexabilidad por contenido y jerarquía -----


def _p(css: str, text: str = "x") -> dict:
    return {"class": css, "text": text}


def _block(block_type: str, paragraphs: list[dict]) -> dict:
    sem = _block_semantics(block_type, paragraphs, contains_image=False, raw_has_table=False)
    return {
        "block_id": "x",
        "block_type": block_type,
        "block_title": None,
        "full_title": _full_title(block_type, paragraphs),
        "latest_version": {
            "text": "\n".join(p["text"] for p in paragraphs),
            "paragraphs": paragraphs,
        },
        **sem,
    }


def _indexable(block_type: str, paragraphs: list[dict]) -> bool:
    block = _block(block_type, paragraphs)
    return build_retrieval("BOE-A-2015-10565", "http://x", "Ley X", block)["indexable"]


def _enc(*pairs: tuple[str, str]) -> dict:
    return {
        "block_type": "encabezado",
        "latest_version": {"paragraphs": [{"class": c, "text": t} for c, t in pairs]},
    }


def _state() -> dict:
    return {
        "book": None,
        "title": None,
        "chapter": None,
        "section": None,
        "subsection": None,
        "annex": None,
    }


def test_indexable_title_heading_is_false() -> None:
    assert _indexable("encabezado", [_p("titulo_num", "TÍTULO I"), _p("titulo_tit", "De")]) is False


def test_indexable_subseccion_heading_is_false() -> None:
    assert _indexable("encabezado", [_p("subseccion", "Subsección 1")]) is False


def test_indexable_annex_with_paragraphs() -> None:
    paras = [_p("anexo_num", "ANEXO I"), _p("anexo_tit", "Defs"), _p("parrafo", "cuerpo")]
    assert _indexable("encabezado", paras) is True


def test_indexable_annex_with_table() -> None:
    assert (
        _indexable("encabezado", [_p("anexo_num", "ANEXO"), _p("cuerpo_tabla_izq", "celda")])
        is True
    )


def test_indexable_derogation_in_encabezado() -> None:
    paras = [_p("articulo", "Disposición derogatoria."), _p("parrafo", "Quedan derogadas.")]
    assert _indexable("encabezado", paras) is True


def test_indexable_grouped_articles_in_encabezado() -> None:
    assert (
        _indexable("encabezado", [_p("articulo", "Artículos 35 a 41"), _p("parrafo", "...")])
        is True
    )


def test_annex_without_body_not_indexable() -> None:
    assert _indexable("encabezado", [_p("anexo_num", "ANEXO I"), _p("anexo_tit", "X")]) is False


def test_nota_inicial_not_indexable() -> None:
    assert (
        _indexable("nota_inicial", [_p("textoCompleto", "Incluye la corrección de errores.")])
        is False
    )


def test_semantics_flags_coherent() -> None:
    sem = _block_semantics(
        "encabezado",
        [_p("anexo_num", "ANEXO I"), _p("cabeza_tabla", "h"), _p("cuerpo_tabla_izq", "c")],
        contains_image=False,
        raw_has_table=True,
    )
    assert sem["semantic_role"] == "annex"
    assert sem["is_annex"] is True
    assert sem["contains_table"] is True
    assert sem["table_text_available"] is True
    assert sem["_annex_is_local"] is False


def test_singular_annex_is_local_context() -> None:
    sem = _block_semantics(
        "encabezado", [_p("anexo", "ANEXO. Definiciones"), _p("parrafo", "y")], False, False
    )
    assert sem["is_annex"] is True
    assert sem["_annex_is_local"] is True  # singular -> contexto local, no propagado


def test_singular_anexo_label_is_not_annex() -> None:
    sem = _block_semantics(
        "encabezado", [_p("anexo", "TEXTO REFUNDIDO DE LA LEY"), _p("parrafo", "y")], False, False
    )
    assert sem["is_annex"] is False
    assert sem["semantic_role"] == "content_heading"


def test_full_title_for_recognized_headings() -> None:
    assert (
        _full_title("encabezado", [_p("libro_num", "LIBRO I"), _p("libro_tit", "De")])
        == "LIBRO I. De"
    )
    assert (
        _full_title(
            "encabezado", [_p("anexo_num", "ANEXO I"), _p("anexo_tit", "T"), _p("parrafo", "b")]
        )
        == "ANEXO I. T"
    )
    assert _full_title("preambulo", [_p("subseccion", "I"), _p("parrafo", "x")]) is None


def test_hierarchy_new_book_resets_all_lower() -> None:
    st = {
        "book": "X",
        "title": "T",
        "chapter": "C",
        "section": "S",
        "subsection": "SS",
        "annex": "A",
    }
    _update_hierarchy(st, _enc(("libro_num", "LIBRO II")))
    assert st == {
        "book": "LIBRO II",
        "title": None,
        "chapter": None,
        "section": None,
        "subsection": None,
        "annex": None,
    }


def test_hierarchy_new_title_resets_annex_and_lower() -> None:
    st = {
        "book": "L",
        "title": "T",
        "chapter": "C",
        "section": "S",
        "subsection": "SS",
        "annex": "A",
    }
    _update_hierarchy(st, _enc(("titulo_num", "TÍTULO II")))
    assert st["book"] == "L" and st["title"] == "TÍTULO II"
    assert st["chapter"] is None and st["section"] is None and st["subsection"] is None
    assert st["annex"] is None


def test_hierarchy_chapter_section_subsection_resets() -> None:
    st = _state()
    _update_hierarchy(st, _enc(("capitulo_num", "CAP I")))
    st["section"], st["subsection"] = "S", "SS"
    _update_hierarchy(st, _enc(("seccion", "Sección 2")))
    assert st["section"] == "Sección 2" and st["subsection"] is None
    st["subsection"] = "SS"
    _update_hierarchy(st, _enc(("subseccion", "Subsección 1")))
    assert st["subsection"] == "Subsección 1"


def test_hierarchy_new_annex_resets_articulated_levels() -> None:
    st = {
        "book": "L",
        "title": "T",
        "chapter": "C",
        "section": "S",
        "subsection": "SS",
        "annex": None,
    }
    _update_hierarchy(st, _enc(("anexo_num", "ANEXO I")))
    assert st["annex"] == "ANEXO I"
    assert all(st[k] is None for k in ("book", "title", "chapter", "section", "subsection"))


# --- integridad temporal: selección de versión vigente por índice -----------


def _versions(*specs: tuple[str, str | None]) -> list[dict]:
    """Construye `versions[]` con (source_norm_id, publication_date_iso)."""
    return [{"source_norm_id": s, "publication_date": p, "validity_date": None} for s, p in specs]


def _texto_block(block_id: str, block_type: str, versions: list[tuple]) -> etree._Element:
    """Construye `data` con un único bloque y versiones (id_norma, pub, vig, [(clase,texto)])."""
    vs = ""
    for idn, pub, vig, paras in versions:
        attrs = f'id_norma="{idn}"'
        if pub:
            attrs += f' fecha_publicacion="{pub}"'
        if vig:
            attrs += f' fecha_vigencia="{vig}"'
        ptxt = "".join(f'<p class="{c}">{t}</p>' for c, t in paras)
        vs += f"<version {attrs}>{ptxt}</version>"
    xml = (
        "<response><status><code>200</code></status><data><texto>"
        f'<bloque id="{block_id}" tipo="{block_type}" titulo="t">{vs}</bloque>'
        "</texto></data></response>"
    )
    return validate_response(etree.fromstring(xml.encode()), Path("texto.xml"))


def test_resolve_unique_match_and_max() -> None:
    res = resolve_current_version(
        _versions(("N1", "2015-01-01"), ("N2", "2020-01-01")), "2020-01-01"
    )
    assert res["status"] == "resolved"
    assert res["selected_version_index"] == 1
    assert res["selected_source_norm_id"] == "N2"
    assert res["selection_method"] == "index_date_exact_unique_match"


def test_resolve_non_chronological_selects_index_not_last() -> None:
    # Última versión del XML (1990) es histórica; el índice apunta a la de 2013.
    versions = _versions(("N1985", "1985-04-03"), ("N2013", "2013-12-30"), ("N1990", "1990-01-11"))
    res = resolve_current_version(versions, "2013-12-30")
    assert res["status"] == "resolved"
    assert res["selected_version_index"] == 1  # no es la última (índice 2)
    assert res["selected_publication_date"] == "2013-12-30"


def test_resolve_historical_at_end_never_selected() -> None:
    res = resolve_current_version(
        _versions(("N2020", "2020-01-01"), ("N1990", "1990-01-01")), "2020-01-01"
    )
    assert res["selected_publication_date"] == "2020-01-01"
    assert res["selected_version_index"] == 0


def test_resolve_zero_match_is_unresolved() -> None:
    res = resolve_current_version(_versions(("N1", "2015-01-01")), "2099-01-01")
    assert res["status"] == "unresolved"
    assert res["selected_version_index"] is None


def test_resolve_two_matches_is_ambiguous() -> None:
    res = resolve_current_version(
        _versions(("N1", "2020-01-01"), ("N2", "2020-01-01")), "2020-01-01"
    )
    assert res["status"] == "ambiguous"
    assert res["selected_version_index"] is None


def test_resolve_missing_index_date() -> None:
    res = resolve_current_version(_versions(("N1", "2015-01-01")), None)
    assert res["status"] == "missing_index_date"


def test_resolve_index_not_max_is_quarantine() -> None:
    # El índice coincide con la de 2013 pero existe una posterior (2020): no es la máxima.
    res = resolve_current_version(
        _versions(("N2013", "2013-12-30"), ("N2020", "2020-01-01")), "2013-12-30"
    )
    assert res["status"] == "index_not_max"
    assert res["selected_version_index"] is None


def test_quarantine_preserves_versions_and_nulls_latest() -> None:
    data = _texto_block(
        "aX",
        "precepto",
        [
            ("N1", "20150101", None, [("articulo", "Artículo X.")]),
            ("N2", "20200101", None, [("articulo", "Artículo X.")]),
        ],
    )
    blocks, _, warnings = parse_text_blocks(data, {"aX": "2099-01-01"})  # sin coincidencia
    b = blocks["aX"]
    assert b["temporal_quarantined"] is True
    assert b["temporal_resolution"]["status"] == "unresolved"
    assert b["latest_version"] is None  # cuarentena: sin latest_version
    assert len(b["versions"]) == 2  # conserva versions[] para diagnóstico
    assert any("cuarentena" in w for w in warnings)


def test_quarantine_block_not_indexable_with_reason() -> None:
    data = _texto_block("aX", "precepto", [("N1", "20150101", None, [("articulo", "Artículo X.")])])
    blocks, _, _ = parse_text_blocks(data, {"aX": None})
    b = blocks["aX"]
    b["block_id"] = "aX"
    retrieval = build_retrieval(NORM_ID, "http://x", "Ley X", b)
    assert retrieval["indexable"] is False
    assert retrieval["excluded_reason"] == "temporal_quarantine:missing_index_date"


def test_invalid_index_date_is_quarantine() -> None:
    data = _texto_block("aX", "precepto", [("N1", "20150101", None, [("articulo", "Artículo X.")])])
    blocks, _, _ = parse_text_blocks(data, {"aX": None}, index_dates_invalid={"aX"})
    assert blocks["aX"]["temporal_resolution"]["status"] == "invalid_date"
    assert blocks["aX"]["temporal_quarantined"] is True


def test_resolved_block_keeps_correct_source_and_text() -> None:
    data = _texto_block(
        "a2",
        "precepto",
        [
            (
                "N1985",
                "19850403",
                "19850423",
                [("articulo", "Artículo 2."), ("parrafo", "Texto viejo.")],
            ),
            (
                "N2013",
                "20131230",
                "20131231",
                [("articulo", "Artículo 2."), ("parrafo", "Texto vigente.")],
            ),
            (
                "N1990",
                "19900111",
                "19900111",
                [("articulo", "Artículo 2."), ("parrafo", "Texto intermedio.")],
            ),
        ],
    )
    blocks, _, _ = parse_text_blocks(data, {"a2": "2013-12-30"})
    b = blocks["a2"]
    assert b["temporal_resolution"]["status"] == "resolved"
    assert b["latest_version"]["source_norm_id"] == "N2013"
    assert "Texto vigente." in b["latest_version"]["text"]
    assert "Texto intermedio." not in b["latest_version"]["text"]


# --- bloques vigentes «(Sin contenido)» --------------------------------------


def test_is_without_content_detection() -> None:
    assert is_without_content([_p("articulo", "Artículo 45."), _p("parrafo", "(Sin contenido)")])
    assert not is_without_content([_p("parrafo", "1. Contenido normativo real.")])
    assert not is_without_content([_p("anexo_num", "ANEXO I")])  # solo rótulo, sin cuerpo


def test_without_content_block_is_indexable_with_neutral_flag() -> None:
    data = _texto_block(
        "a45",
        "precepto",
        [
            (
                "N2013",
                "20131230",
                "20131231",
                [("articulo", "Artículo 45."), ("parrafo", "(Sin contenido)")],
            )
        ],
    )
    blocks, _, _ = parse_text_blocks(data, {"a45": "2013-12-30"})
    b = blocks["a45"]
    assert b["content_status"] == "without_content"
    assert b["is_without_content"] is True
    b["block_id"] = "a45"
    assert build_retrieval(NORM_ID, "http://x", "Ley X", b)["indexable"] is True


def test_nota_inicial_stays_in_document_blocks() -> None:
    texto = (
        '<?xml version="1.0" encoding="utf-8"?><response>'
        "<status><code>200</code></status><data><texto>"
        '<bloque id="ni" tipo="nota_inicial"><version id_norma="BOE-A-2015-10565">'
        '<p class="textoCompleto">Incluye la corrección de errores.</p></version></bloque>'
        "</texto></data></response>"
    )
    data = validate_response(etree.fromstring(texto.encode()), Path("texto.xml"))
    blocks, order_ids, _ = parse_text_blocks(data)
    assert "ni" in blocks  # se conserva para trazabilidad
    assert blocks["ni"]["semantic_role"] == "initial_note"
