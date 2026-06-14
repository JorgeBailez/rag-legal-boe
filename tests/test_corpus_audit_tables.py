"""Tests del check de cobertura de tablas forma B en la auditoría (offline, núcleo puro).

`_missing_raw_cells` cierra el punto ciego <p>-céntrico: detecta celdas <td> de tabla forma B
vigentes (fuera de blockquote) ausentes del cuerpo persistido, sin falsos positivos numéricos.
"""

from lxml import etree

from src.quality.corpus_audit import _missing_raw_cells

# Versión vigente con tabla forma B (texto crudo en <td>) fuera de blockquote.
_VERSION = """<version>
<p class="parrafo">Intro.</p>
<table class="tabla">
<thead><tr><th>Periodo de generación</th><th>Coeficiente</th></tr></thead>
<tbody><tr><td>Inferior a 1 año.</td><td>0,14</td></tr></tbody>
</table>
</version>"""


def _v(xml: str) -> etree._Element:
    return etree.fromstring(xml)


def test_detecta_celdas_de_tabla_forma_b_ausentes_del_cuerpo() -> None:
    # Cuerpo persistido SIN la tabla (el bug): las celdas-concepto distintivas salen como ausentes.
    missing = _missing_raw_cells(_v(_VERSION), body="Intro.")
    assert "Inferior a 1 año." in missing
    assert "Periodo de generación" in missing


def test_no_falso_positivo_si_la_tabla_esta_en_el_cuerpo() -> None:
    # Cuerpo CON la tabla linealizada por fila: no falta nada.
    body = "Intro.\nPeriodo de generación | Coeficiente\nInferior a 1 año. | 0,14"
    assert _missing_raw_cells(_v(_VERSION), body) == []


def test_no_marca_celdas_numericas_ni_cortas() -> None:
    # "0,14" (numérico, sin letra) no se exige aunque no esté: evita colisiones por azar.
    missing = _missing_raw_cells(_v(_VERSION), body="Intro.")
    assert "0,14" not in missing


def test_tabla_forma_b_en_blockquote_no_se_exige() -> None:
    # Tabla forma B citada en blockquote (derogada): NO debe estar en el cuerpo -> no se marca.
    xml = (
        '<version><p class="parrafo">Cuerpo.</p>'
        '<blockquote class="soloTexto"><table class="tabla">'
        "<tr><td>Concepto derogado distintivo.</td><td>9</td></tr></table></blockquote></version>"
    )
    assert _missing_raw_cells(_v(xml), body="Cuerpo.") == []


def test_forma_a_no_se_marca() -> None:
    # Forma A (<td><p>): su texto vive en <p> (ya en el cuerpo); el check no la considera.
    xml = (
        '<version><table class="tabla"><tr>'
        '<td><p class="cuerpo_tabla_izq">Celda con letras</p></td></tr></table></version>'
    )
    assert _missing_raw_cells(_v(xml), body="") == []
