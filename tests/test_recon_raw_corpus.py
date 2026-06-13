"""Test del reconocimiento read-only del XML raw (offline; usa lxml vía el parser)."""

from pathlib import Path

from scripts.recon_raw_corpus import recon_norm

_RESPONSE = "<response><status><code>200</code></status><data>{body}</data></response>"


def _write(base: Path, name: str, body: str) -> None:
    base.mkdir(parents=True, exist_ok=True)
    (base / name).write_text(_RESPONSE.format(body=body), encoding="utf-8")


def test_recon_detecta_clases_fechas_y_jerarquia(tmp_path: Path) -> None:
    norm = "BOE-A-2099-1"
    texto = (
        '<bloque tipo="precepto"><version fecha_publicacion="20231201">'
        '<p class="articulo">Artículo 1.</p>'
        '<p class="parrafo">Cuerpo del artículo.</p>'
        '<p class="disposicion_adicional_num">Disposición adicional primera.</p>'
        '<p class="nota_pie">Nota editorial.</p>'
        "</version></bloque>"
    )
    indice = (
        "<bloque><id>a1</id><fecha_actualizacion>20231201</fecha_actualizacion></bloque>"
        "<bloque><id>a2</id><fecha_actualizacion>1989-04-02</fecha_actualizacion></bloque>"
    )
    _write(tmp_path / norm, "texto.xml", texto)
    _write(tmp_path / norm, "indice.xml", indice)

    r = recon_norm(tmp_path, norm)
    assert "error" not in r
    assert r["block_types"] == {"precepto": 1}
    # clase con pinta estructural pero NO reconocida por el parser → candidata a endurecer.
    assert "disposicion_adicional_num" in r["unknown_structural_candidates"]
    assert r["classes_by_kind"].get("estructural", 0) == 0
    assert r["classes_by_kind"].get("nota", 0) == 1
    # fecha con formato no normalizable → riesgo de cuarentena temporal.
    assert r["invalid_date_count"] == 1
    assert "1989-04-02" in r["invalid_date_samples"]


def test_recon_degrada_sin_lanzar_con_xml_invalido(tmp_path: Path) -> None:
    norm = "BOE-A-2099-2"
    base = tmp_path / norm
    base.mkdir(parents=True)
    (base / "texto.xml").write_text("esto no es XML válido", encoding="utf-8")
    r = recon_norm(tmp_path, norm)
    assert "error" in r  # degrada con diagnóstico, no rompe
