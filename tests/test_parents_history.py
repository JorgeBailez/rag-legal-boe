"""Tests de cobertura/propiedad de parents e histories (sobre el parser real, sin red)."""

import json
from pathlib import Path

import pytest

from src.boe.parser import build_processed_bundle
from src.contracts.models import DocumentV2, HistoryV2, ParentsV2

NORM_ID = "BOE-A-2015-10565"

METADATOS_XML = """<?xml version="1.0" encoding="utf-8"?>
<response><status><code>200</code></status><data><metadatos>
<identificador>BOE-A-2015-10565</identificador>
<titulo>Ley 39/2015</titulo>
<rango codigo="1300">Ley</rango>
<numero_oficial>39/2015</numero_oficial>
<url_html_consolidada>https://www.boe.es/buscar/act.php?id=BOE-A-2015-10565</url_html_consolidada>
</metadatos></data></response>"""

INDICE_XML = """<?xml version="1.0" encoding="utf-8"?>
<response><status><code>200</code></status><data>
<bloque><id>pre</id><titulo>Preámbulo</titulo>
  <fecha_actualizacion>20151002</fecha_actualizacion><url>u</url></bloque>
<bloque><id>a1</id><titulo>Artículo 1</titulo>
  <fecha_actualizacion>20151002</fecha_actualizacion><url>u</url></bloque>
<bloque><id>fir</id><titulo>Firma</titulo>
  <fecha_actualizacion>20151002</fecha_actualizacion><url>u</url></bloque>
</data></response>"""

TEXTO_XML = """<?xml version="1.0" encoding="utf-8"?>
<response><status><code>200</code></status><data><texto>
<bloque id="pre" tipo="preambulo" titulo="Preámbulo">
  <version id_norma="BOE-A-2015-10565" fecha_publicacion="20151002" fecha_vigencia="20161002">
    <p class="parrafo">Exposición de motivos del preámbulo.</p></version></bloque>
<bloque id="a1" tipo="precepto" titulo="Artículo 1">
  <version id_norma="BOE-A-2015-10565" fecha_publicacion="20151002" fecha_vigencia="20161002">
    <p class="articulo">Artículo 1.</p>
    <p class="parrafo">Texto del artículo uno.</p></version></bloque>
<bloque id="fir" tipo="firma" titulo="Firma">
  <version id_norma="BOE-A-2015-10565" fecha_publicacion="20151002" fecha_vigencia="20161002">
    <p class="firma">Juan Carlos R.</p></version></bloque>
</texto></data></response>"""

MANIFEST = {
    "norm_id": NORM_ID,
    "base_url": "https://www.boe.es/datosabiertos/api",
    "downloaded_at": "2026-05-28T22:38:10Z",
    "files": [],
}


@pytest.fixture
def bundle(tmp_path: Path):
    raw = tmp_path / "raw" / NORM_ID
    raw.mkdir(parents=True)
    (raw / "metadatos.xml").write_text(METADATOS_XML, encoding="utf-8")
    (raw / "indice.xml").write_text(INDICE_XML, encoding="utf-8")
    (raw / "texto.xml").write_text(TEXTO_XML, encoding="utf-8")
    manifest = tmp_path / "manifests" / f"{NORM_ID}.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(MANIFEST), encoding="utf-8")
    return build_processed_bundle(NORM_ID, tmp_path / "raw", manifest)


def test_three_artifacts_validate(bundle) -> None:
    DocumentV2.model_validate(bundle.document)
    HistoryV2.model_validate(bundle.history)
    ParentsV2.model_validate(bundle.parents)


def test_modification_notes_owned_by_history_not_parents(bundle) -> None:
    # Ningún parent contiene modification_notes; history es el propietario.
    assert all("modification_notes" not in p for p in bundle.parents["parents"])
    assert all("modification_notes" in h for h in bundle.history["blocks"])


def test_history_covers_every_block(bundle) -> None:
    doc_ids = {b["block_id"] for b in bundle.document["blocks"]}
    hist_ids = {h["block_id"] for h in bundle.history["blocks"]}
    assert hist_ids == doc_ids  # incluso bloques monoversión


def test_parents_cover_all_resolved_with_text_incl_excluded(bundle) -> None:
    # firma (excluida) y preámbulo (indexable) y a1 (indexable): los 3 tienen texto -> 3 parents.
    parent_blocks = {p["block_id"] for p in bundle.parents["parents"]}
    assert parent_blocks == {"pre", "a1", "fir"}
    # La firma NO es indexable pero conserva parent.
    fir_desc = next(b for b in bundle.document["blocks"] if b["block_id"] == "fir")
    assert fir_desc["indexable"] is False
    assert fir_desc["parent_id"] is not None


def test_parent_has_current_version_and_no_indexable(bundle) -> None:
    p = next(p for p in bundle.parents["parents"] if p["block_id"] == "a1")
    assert p["current_version"]["publication_date"] == "2015-10-02"
    assert "indexable" not in p


def test_descriptor_owns_indexable_and_excluded_reason(bundle) -> None:
    fir = next(b for b in bundle.document["blocks"] if b["block_id"] == "fir")
    assert fir["indexable"] is False
    assert fir["excluded_reason"] is not None


def test_manifest_ref_is_relative(bundle) -> None:
    ref = bundle.document["source"]["manifest_ref"]
    assert not Path(ref).is_absolute()
    assert ref.startswith("data/manifests") or ref.endswith(".json")
