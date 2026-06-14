"""Tests de la invariante estructural de dos caras + observabilidad del drop editorial (offline).

Verifican el núcleo PURO `verify_editorial_invariant` (sin disco): reaplica la regla canónica al
raw vigente y la contrasta con el cuerpo persistido por pertenencia/orden, nunca por containment.
"""

from lxml import etree

from src.quality.corpus_audit import verify_editorial_invariant

# Versión vigente con cuerpo + aparato editorial citado en blockquote + nota_pie de modificación.
_VERSION = """<version>
<p class="articulo">Artículo 1. Objeto.</p>
<p class="parrafo">1. Texto vigente uno.</p>
<blockquote caduca="20250101" class="soloTexto">
<p class="cita_con_pleca">Redacción anterior:</p>
<p class="parrafo">1. Texto derogado uno.</p>
</blockquote>
<p class="parrafo">2. Texto vigente dos.</p>
<p class="nota_pie">Se modifica por la Ley 1/2025. Ref. BOE-A-2025-1</p>
</version>"""

_KEPT = ["Artículo 1. Objeto.", "1. Texto vigente uno.", "2. Texto vigente dos."]


def _version(xml: str) -> etree._Element:
    return etree.fromstring(xml)


def _persisted(texts: list[str]) -> list[dict]:
    return [{"order": i + 1, "class": "parrafo", "text": t} for i, t in enumerate(texts)]


def test_invariante_ok_no_marca_y_registra_el_drop() -> None:
    findings, drop = verify_editorial_invariant(_version(_VERSION), _persisted(_KEPT), "DOC", "a1")
    assert [f["check"] for f in findings] == []  # cuerpo persistido == recomputado
    assert drop["n_kept"] == 3
    assert drop["n_notes"] == 1  # la nota_pie es provenance, no cuerpo
    assert drop["n_dropped_blockquote"] == 2  # marcador + redacción derogada
    assert drop["dropped_classes"] == {"cita_con_pleca": 1, "parrafo": 1}
    assert drop["anomaly"] == []  # ni estructural/tabla ni fracción alta


def test_invariante_detecta_fuga_editorial() -> None:
    # Una redacción derogada del blockquote se cuela en el cuerpo persistido → fuga.
    persisted = _persisted([*_KEPT, "1. Texto derogado uno."])
    findings, _drop = verify_editorial_invariant(_version(_VERSION), persisted, "DOC", "a1")
    inv = [f for f in findings if f["check"] == "block.editorial_invariant"]
    assert len(inv) == 1 and inv[0]["severity"] == "ERROR"
    assert "fuga" in inv[0]["message"]


def test_invariante_detecta_sobre_borrado() -> None:
    # Falta un párrafo vigente en el cuerpo persistido → sobre-borrado.
    persisted = _persisted(_KEPT[:-1])
    findings, _drop = verify_editorial_invariant(_version(_VERSION), persisted, "DOC", "a1")
    inv = [f for f in findings if f["check"] == "block.editorial_invariant"]
    assert len(inv) == 1 and inv[0]["severity"] == "ERROR"
    assert "sobre-borrado" in inv[0]["message"]


def test_invariante_no_da_falso_positivo_con_derogado_igual_a_vigente() -> None:
    # La redacción derogada es LITERALMENTE idéntica a un apartado vigente (caso de los 89 FP de
    # note_leak): por pertenencia/orden NO hay falso positivo; por containment sí lo habría.
    xml = (
        '<version><p class="parrafo">1. Apartado idéntico.</p>'
        '<blockquote class="soloTexto"><p class="parrafo">1. Apartado idéntico.</p></blockquote>'
        '<p class="parrafo">2. Otro apartado.</p></version>'
    )
    persisted = _persisted(["1. Apartado idéntico.", "2. Otro apartado."])
    findings, drop = verify_editorial_invariant(_version(xml), persisted, "DOC", "a1")
    assert [f for f in findings if f["check"] == "block.editorial_invariant"] == []
    assert drop["n_dropped_blockquote"] == 1


def test_observabilidad_marca_anomalia_de_clase_tabla() -> None:
    # Una celda de tabla descartada desde dentro de un blockquote: posible tabla vigente mal
    # envuelta → debe AFLORAR como anomalía (no perderse en silencio).
    xml = (
        '<version><p class="parrafo">Cuerpo vigente.</p>'
        '<blockquote class="soloTexto"><p class="cuerpo_tabla_centro">Celda derogada</p>'
        "</blockquote></version>"
    )
    persisted = _persisted(["Cuerpo vigente."])
    findings, drop = verify_editorial_invariant(_version(xml), persisted, "DOC", "a1")
    assert "structural_class" in drop["anomaly"]
    assert any(f["check"] == "block.editorial_drop_structural" for f in findings)
    # El cuerpo sigue intacto: la anomalía no implica fuga ni sobre-borrado.
    assert [f for f in findings if f["check"] == "block.editorial_invariant"] == []


def test_observabilidad_marca_anomalia_de_fraccion_alta() -> None:
    # > 50 % de los <p> de la versión son aparato editorial citado → aflora para inspección.
    xml = (
        '<version><p class="parrafo">Único vigente.</p>'
        '<blockquote class="soloTexto"><p class="cita_con_pleca">Redacción anterior:</p>'
        '<p class="parrafo">Derogado uno.</p><p class="parrafo">Derogado dos.</p>'
        "</blockquote></version>"
    )
    persisted = _persisted(["Único vigente."])
    findings, drop = verify_editorial_invariant(_version(xml), persisted, "DOC", "a1")
    assert "high_fraction" in drop["anomaly"]
    assert drop["dropped_fraction"] > 0.5
    assert any(f["check"] == "block.editorial_drop_fraction" for f in findings)
