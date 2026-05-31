"""Tests del módulo de corpus (sin red, raw mínimo en tmp_path)."""

import json
from pathlib import Path

from src.boe.corpus import evaluate_criteria, load_seed_corpus, verify_norm

NORM_ID = "BOE-A-2015-10565"

METADATOS_OK = """<?xml version="1.0" encoding="utf-8"?>
<response>
  <status><code>200</code><text>ok</text></status>
  <data>
    <metadatos>
      <identificador>BOE-A-2015-10565</identificador>
      <rango codigo="1300">Ley</rango>
      <ambito codigo="1">Estatal</ambito>
      <numero_oficial>39/2015</numero_oficial>
      <titulo>Ley 39/2015, de 1 de octubre...</titulo>
      <fecha_vigencia>20161002</fecha_vigencia>
      <estatus_derogacion>N</estatus_derogacion>
      <estatus_anulacion>N</estatus_anulacion>
      <vigencia_agotada>N</vigencia_agotada>
      <estado_consolidacion codigo="3">Finalizado</estado_consolidacion>
      <url_html_consolidada>https://www.boe.es/buscar/act.php?id=BOE-A-2015-10565</url_html_consolidada>
    </metadatos>
  </data>
</response>
"""


def _meta(**overrides) -> dict:
    base = {
        "derogation_status": "N",
        "expired_validity": "N",
        "consolidation_status": {"code": "3", "label": "Finalizado"},
    }
    base.update(overrides)
    return base


def _availability(**overrides) -> dict:
    base = {e: True for e in ("metadatos", "texto", "indice", "analisis", "metadata_eli", "full")}
    base.update(overrides)
    return base


# --- evaluate_criteria -------------------------------------------------------


def test_evaluate_criteria_pass() -> None:
    result = evaluate_criteria(_meta(), _availability())
    assert result["meets_criteria"] is True
    assert result["reasons"] == []


def test_evaluate_criteria_fails_when_derogated() -> None:
    result = evaluate_criteria(_meta(derogation_status="S"), _availability())
    assert result["meets_criteria"] is False
    assert any("vigente" in r for r in result["reasons"])


def test_evaluate_criteria_fails_when_not_finalizado() -> None:
    meta = _meta(consolidation_status={"code": "2", "label": "En tramitación"})
    result = evaluate_criteria(meta, _availability())
    assert result["meets_criteria"] is False
    assert any("estado_consolidacion" in r for r in result["reasons"])


def test_evaluate_criteria_fails_when_missing_mandatory_endpoint() -> None:
    result = evaluate_criteria(_meta(), _availability(texto=False))
    assert result["meets_criteria"] is False
    assert any("obligatorios" in r for r in result["reasons"])


# --- verify_norm -------------------------------------------------------------


def _write_metadatos(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw"
    (raw_dir / NORM_ID).mkdir(parents=True)
    (raw_dir / NORM_ID / "metadatos.xml").write_text(METADATOS_OK, encoding="utf-8")
    return raw_dir


def test_verify_norm_pass_with_all_endpoints(tmp_path: Path) -> None:
    raw_dir = _write_metadatos(tmp_path)
    downloaded = {
        e: raw_dir / NORM_ID / f"{e}.xml"
        for e in ("metadatos", "texto", "indice", "analisis", "metadata_eli", "full")
    }
    row = verify_norm(NORM_ID, raw_dir, downloaded)
    assert row["exists"] is True
    assert row["meets_criteria"] is True
    assert row["rank"] == {"code": "1300", "label": "Ley"}
    assert row["availability"]["analisis"] is True


def test_verify_norm_optional_endpoint_absent_still_passes(tmp_path: Path) -> None:
    raw_dir = _write_metadatos(tmp_path)
    # Sin analisis ni metadata_eli (opcionales): sigue cumpliendo criterios.
    downloaded = {e: raw_dir / NORM_ID / f"{e}.xml" for e in ("metadatos", "texto", "indice")}
    row = verify_norm(NORM_ID, raw_dir, downloaded)
    assert row["meets_criteria"] is True
    assert row["availability"]["analisis"] is False
    assert row["availability"]["metadata_eli"] is False


def test_verify_norm_missing_mandatory_endpoint_fails(tmp_path: Path) -> None:
    raw_dir = _write_metadatos(tmp_path)
    downloaded = {e: raw_dir / NORM_ID / f"{e}.xml" for e in ("metadatos", "indice")}  # falta texto
    row = verify_norm(NORM_ID, raw_dir, downloaded)
    assert row["meets_criteria"] is False
    assert any("obligatorios" in r for r in row["reasons"])


# --- load_seed_corpus --------------------------------------------------------


def test_load_seed_corpus(tmp_path: Path) -> None:
    path = tmp_path / "seed.json"
    path.write_text(
        json.dumps({"norms": [{"norm_id": NORM_ID, "label": "Ley 39/2015"}]}),
        encoding="utf-8",
    )
    norms = load_seed_corpus(path)
    assert norms[0]["norm_id"] == NORM_ID


def test_real_seed_corpus_has_ten_norms() -> None:
    norms = load_seed_corpus()
    assert len(norms) == 10
    assert all(n["norm_id"].startswith("BOE-A-") for n in norms)
