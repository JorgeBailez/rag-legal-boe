"""Tests de captura de tablas (forma A con <p> y forma B con <td> crudo), sin red.

El BOE escribe tablas de dos formas. Forma A: <td><p class="cuerpo_tabla_*">…</p></td>.
Forma B: <td>…</td> sin <p>. AMBAS se linealizan POR FILA (concepto | valor), uniendo las
celdas de cada <tr>; los <p> internos de las celdas no entran sueltos (no se duplican).
"""

from lxml import etree

from src.boe.parser import classify_version_paragraphs, is_table_class

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


def test_forma_a_se_linealiza_por_fila() -> None:
    # Forma A (<td><p class="cuerpo_tabla_*">): se linealiza por fila como la B, emparejando las
    # celdas de cada <tr>, y SIN duplicar (los <p> internos no entran además sueltos).
    xml = (
        '<version><p class="articulo">Art. 1.</p>'
        '<table class="tabla"><tr><td><p class="cuerpo_tabla_izq">Celda A</p></td>'
        '<td><p class="cuerpo_tabla_centro">Celda B</p></td></tr></table></version>'
    )
    kept = _kept(xml)
    texts = [p["text"] for p in kept]
    classes = [p["class"] for p in kept]
    assert texts == ["Art. 1.", "Celda A | Celda B"]  # una fila, concepto+valor emparejados
    assert classes == ["articulo", "cuerpo_tabla_fila"]
    assert sum(t.count("Celda A") for t in texts) == 1  # no duplicada


def test_forma_a_cabecera_th_es_cabeza_tabla() -> None:
    # Cabeceras forma A (<th><p class="cabeza_tabla">) -> fila clase cabeza_tabla.
    xml = (
        '<version><table class="tabla">'
        '<tr><th><p class="cabeza_tabla">Concepto</p></th>'
        '<th><p class="cabeza_tabla">Importe</p></th></tr>'
        '<tr><td><p class="cuerpo_tabla_izq">Cuota fija</p></td>'
        '<td><p class="cuerpo_tabla_centro">12,62</p></td></tr></table></version>'
    )
    kept = _kept(xml)
    assert kept[0]["class"] == "cabeza_tabla"
    assert kept[0]["text"] == "Concepto | Importe"
    assert kept[1]["class"] == "cuerpo_tabla_fila"
    assert kept[1]["text"] == "Cuota fija | 12,62"


def test_forma_a_multicolumna_empareja_concepto_y_valores() -> None:
    # Tres columnas (modelo Art. 72 Haciendas): concepto + dos valores en una línea, ordenados.
    xml = (
        '<version><table class="tabla">'
        '<tr><th><p class="cabeza_tabla">Puntos</p></th>'
        '<th><p class="cabeza_tabla">Urbanos</p></th>'
        '<th><p class="cabeza_tabla">Rústicos</p></th></tr>'
        '<tr><td><p class="cuerpo_tabla_izq">A) Municipios capital</p></td>'
        '<td><p class="cuerpo_tabla_centro">0,07</p></td>'
        '<td><p class="cuerpo_tabla_centro">0,06</p></td></tr></table></version>'
    )
    kept = _kept(xml)
    fila = next(p for p in kept if p["class"] == "cuerpo_tabla_fila")
    assert fila["text"] == "A) Municipios capital | 0,07 | 0,06"


def test_forma_a_dentro_de_blockquote_es_editorial() -> None:
    # Tabla forma A citada en blockquote (redacción derogada) -> fuera del cuerpo, sin duplicar.
    xml = (
        '<version><p class="parrafo">Cuerpo vigente.</p>'
        '<blockquote class="soloTexto"><table class="tabla">'
        '<tr><td><p class="cuerpo_tabla_izq">Concepto derogado</p></td>'
        '<td><p class="cuerpo_tabla_centro">9,99</p></td></tr></table></blockquote></version>'
    )
    kept, _notes, dropped = classify_version_paragraphs(etree.fromstring(xml))
    assert [p["text"] for p in kept] == ["Cuerpo vigente."]
    assert any("Concepto derogado" in d["text"] for d in dropped)
    assert all("Concepto derogado" not in p["text"] for p in kept)


def test_forma_a_subrubrica_celda_vacia() -> None:
    # Fila "subrúbrica" (1er <td> con texto, resto vacíos): las celdas vacías se filtran ->
    # la fila queda con una sola celda (sin separador " | " colgando).
    xml = (
        '<version><table class="tabla">'
        '<tr><td><p class="cuerpo_tabla_izq">A) Turismos:</p></td>'
        '<td><p class="cuerpo_tabla_centro"> </p></td></tr>'
        '<tr><td><p class="cuerpo_tabla_izq">De menos de ocho caballos</p></td>'
        '<td><p class="cuerpo_tabla_centro">12,62</p></td></tr></table></version>'
    )
    kept = _kept(xml)
    assert kept[0]["text"] == "A) Turismos:"
    assert kept[1]["text"] == "De menos de ocho caballos | 12,62"
