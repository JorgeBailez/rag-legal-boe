"""Tests de captura de tablas (forma A con <p> vs forma B con <td> crudo), sin red.

Forma A: <td><p class="cuerpo_tabla_*">…</p></td>  -> la capta version.iter("p") (intacta).
Forma B: <td>…</td> sin <p>                        -> se linealiza por fila (antes se perdía).
"""

from lxml import etree

from src.boe.parser import classify_version_paragraphs, is_form_b_table, is_table_class

# Forma B real (Art. 107 Haciendas Locales): cabecera <thead><th> + filas <td> con texto crudo.
_PLUSVALIA = """<version>
<p class="parrafo">El coeficiente se determina conforme a la tabla siguiente:</p>
<table class="tabla">
<thead><tr><th>Periodo de generación</th><th>Coeficiente</th></tr></thead>
<tbody>
<tr><td>Inferior a 1 año.</td><td>0,14</td></tr>
<tr><td>1 año.</td><td>0,13</td></tr>
<tr><td>2 años.</td><td>0,15</td></tr>
</tbody>
</table>
</version>"""


def _kept(xml: str):
    kept, _notes, _dropped = classify_version_paragraphs(etree.fromstring(xml))
    return kept


def test_forma_b_captura_cabecera_y_filas_en_orden() -> None:
    kept = _kept(_PLUSVALIA)
    classes = [p["class"] for p in kept]
    texts = [p["text"] for p in kept]
    # Intro (no tabla) primero, luego cabecera y filas en su posición de documento.
    assert texts[0].startswith("El coeficiente")
    assert classes[1] == "cabeza_tabla"  # <th> -> cabecera de columnas
    assert all(c == "cuerpo_tabla_fila" for c in classes[2:])  # filas de cuerpo
    assert all(is_table_class(c) for c in classes[1:])  # reconocibles por is_table_class


def test_forma_b_linealiza_por_fila_legible() -> None:
    # Calidad de retrieval: concepto y valor en la MISMA línea, no celda-a-celda suelta.
    kept = _kept(_PLUSVALIA)
    fila = next(p for p in kept if "Inferior a 1 año." in p["text"])
    assert "0,14" in fila["text"]  # concepto + valor juntos
    cabecera = next(p for p in kept if p["class"] == "cabeza_tabla")
    assert "Periodo de generación" in cabecera["text"] and "Coeficiente" in cabecera["text"]


def test_forma_b_grande_no_pierde_filas() -> None:
    # Tabla tipo Anexo II de Tráfico (21 infracciones): ninguna fila se pierde en silencio.
    filas = "".join(f"<tr><td>Infracción {i}.</td><td>{i}</td></tr>" for i in range(1, 22))
    xml = f'<version><table class="tabla_ancha"><tbody>{filas}</tbody></table></version>'
    kept = _kept(xml)
    assert len([p for p in kept if p["class"] == "cuerpo_tabla_fila"]) == 21


def test_forma_b_dentro_de_blockquote_es_editorial() -> None:
    # Tabla forma B citada en blockquote (redacción derogada) -> fuera del cuerpo.
    xml = (
        '<version><p class="parrafo">Cuerpo vigente.</p>'
        '<blockquote class="soloTexto"><table class="tabla">'
        "<tr><td>Concepto derogado.</td><td>9,99</td></tr></table></blockquote></version>"
    )
    kept, _notes, dropped = classify_version_paragraphs(etree.fromstring(xml))
    assert [p["text"] for p in kept] == ["Cuerpo vigente."]
    assert any("Concepto derogado." in d["text"] for d in dropped)
    assert all("Concepto derogado." not in p["text"] for p in kept)


def test_forma_a_no_regresa_ni_se_duplica() -> None:
    # Forma A (<td><p class="cuerpo_tabla_izq">): se capta por la vía de <p>, sin filas extra.
    xml = (
        '<version><p class="articulo">Art. 1.</p>'
        '<table class="tabla"><tr><td><p class="cuerpo_tabla_izq">Celda A</p></td>'
        '<td><p class="cuerpo_tabla_centro">Celda B</p></td></tr></table></version>'
    )
    kept = _kept(xml)
    texts = [p["text"] for p in kept]
    assert texts == ["Art. 1.", "Celda A", "Celda B"]  # una entrada por celda, sin duplicar
    assert all(p["class"] != "cuerpo_tabla_fila" for p in kept)  # no se linealiza por fila


def test_is_form_b_table_discrimina_las_dos_formas() -> None:
    forma_b = etree.fromstring("<table><tr><td>texto crudo</td></tr></table>")
    forma_a = etree.fromstring('<table><tr><td><p class="cuerpo_tabla_izq">x</p></td></tr></table>')
    vacia = etree.fromstring("<table><tr><td>   </td></tr></table>")
    assert is_form_b_table(forma_b)
    assert not is_form_b_table(forma_a)
    assert not is_form_b_table(vacia)
