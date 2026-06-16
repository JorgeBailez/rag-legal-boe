"""Tests del manejo del aparato editorial del BOE (blockquote → notas), sin perder texto vigente."""

from loguru import logger
from lxml import etree

from src.boe.parser import (
    _parse_version_paragraphs,
    classify_version_paragraphs,
    is_suspicious_blockquote_class,
)
from src.quality.corpus_audit import EDITORIAL_LEAK_RE

# Versión vigente modelada sobre casos reales (a19 Consumidores + art.20 Sanidad/siempreSeVe):
# texto vigente intercalado con blockquotes editoriales (siempreSeVe, soloTexto/triplete, nota_pie).
_VERSION = """<version>
<p class="articulo">Artículo 20. Principio.</p>
<p class="parrafo">1. Apartado uno vigente.</p>
<blockquote class="siempreSeVe">
<p class="parrafo">Se advierte que el texto definitivo difiere del aprobado.</p>
</blockquote>
<p class="parrafo">2. Apartado dos vigente.</p>
<blockquote caduca="20250403" class="soloTexto">
<p class="parrafo">Téngase en cuenta que esto entra en vigor el 3 de abril de 2025.</p>
<p class="cita_con_pleca">Redacción anterior:</p>
<p class="parrafo">"1. Redacción derogada del apartado uno."</p>
</blockquote>
<p class="parrafo">3. Apartado tres vigente.</p>
<blockquote>
<p class="nota_pie">Se modifica por el art. 1 de la Ley 1/2025. Ref. BOE-A-2025-1</p>
</blockquote>
</version>"""


def _split(xml: str):
    paragraphs, notes = _parse_version_paragraphs(etree.fromstring(xml))
    return paragraphs, notes, " || ".join(p["text"] for p in paragraphs)


def test_blockquote_editorial_fuera_del_cuerpo_y_vigente_intacto() -> None:
    paragraphs, notes, body = _split(_VERSION)
    # Cuerpo = solo los 4 párrafos vigentes (artículo + apartados 1, 2, 3), en orden.
    assert [p["text"] for p in paragraphs] == [
        "Artículo 20. Principio.",
        "1. Apartado uno vigente.",
        "2. Apartado dos vigente.",
        "3. Apartado tres vigente.",
    ]
    note_text = " || ".join(n["text"] for n in notes)
    # KILLER: un <p parrafo> dentro de <blockquote siempreSeVe> NO es cuerpo (la clase engaña).
    assert "Se advierte" not in body and "Se advierte" not in note_text  # editorial: descartado
    # Aviso «Téngase»: fuera del cuerpo, conservado como nota (provenance, no solapa el cuerpo).
    assert "Téngase en cuenta" not in body and "Téngase en cuenta" in note_text
    # Marcador + redacción DEROGADA: fuera del cuerpo y NO en notes (solaparían note_leak).
    assert "Redacción anterior:" not in body
    assert "Redacción derogada" not in body and "Redacción derogada" not in note_text
    # nota_pie de modificación sigue siendo nota (provenance con Ref. al BOE).
    assert "Se modifica por el art. 1" in note_text


def test_redaccion_anterior_en_prosa_de_transitoria_NO_se_toca() -> None:
    xml = (
        '<version><p class="articulo">Disposición transitoria 14.ª</p>'
        '<p class="parrafo">La redacción anterior de este artículo se aplicará a los hechos '
        "anteriores a la entrada en vigor.</p></version>"
    )
    paragraphs, notes, _body = _split(xml)
    assert len(paragraphs) == 2  # texto vigente conservado
    assert notes == []
    assert any("redacción anterior" in p["text"].lower() for p in paragraphs)


def test_comilla_legitima_fuera_de_blockquote_NO_se_toca() -> None:
    xml = (
        '<version><p class="articulo">Art. 1.</p>'
        '<p class="parrafo">"Cita entrecomillada legítima dentro del artículo."</p></version>'
    )
    paragraphs, notes, _body = _split(xml)
    assert any(p["text"].startswith('"') for p in paragraphs)
    assert notes == []


def test_clase_estructural_dentro_de_blockquote_se_saca_del_cuerpo() -> None:
    xml = (
        '<version><p class="articulo">Anexo.</p>'
        '<blockquote><p class="anexo_num">ANEXO II citado (derogado)</p></blockquote></version>'
    )
    paragraphs, notes, body = _split(xml)
    assert all(p["class"] != "anexo_num" for p in paragraphs)  # editorial citado: fuera del cuerpo
    assert "ANEXO II citado" not in body
    assert all("ANEXO II citado" not in n["text"] for n in notes)  # derogado citado: no a notes


def test_nota_pie_suelta_sigue_siendo_nota() -> None:
    xml = (
        '<version><p class="articulo">Art. 1.</p><p class="parrafo">Cuerpo.</p>'
        '<p class="nota_pie">Se modifica por la Ley 2/2024. Ref. BOE-A-2024-1</p></version>'
    )
    paragraphs, notes, _body = _split(xml)
    assert [p["text"] for p in paragraphs] == ["Art. 1.", "Cuerpo."]
    assert len(notes) == 1 and notes[0]["target_norm_id"] == "BOE-A-2024-1"


def test_classify_separa_en_tres_cubos() -> None:
    kept, notes, dropped = classify_version_paragraphs(etree.fromstring(_VERSION))
    assert [p["text"] for p in kept] == [
        "Artículo 20. Principio.",
        "1. Apartado uno vigente.",
        "2. Apartado dos vigente.",
        "3. Apartado tres vigente.",
    ]
    # «Téngase» + nota_pie → notes (provenance); marcador + redacción derogada → dropped.
    assert any("Téngase en cuenta" in n["text"] for n in notes)
    assert any("Se modifica por el art. 1" in n["text"] for n in notes)
    dropped_texts = [d["text"] for d in dropped]
    assert any("Redacción anterior:" in t for t in dropped_texts)
    assert any("Redacción derogada" in t for t in dropped_texts)
    assert any("Se advierte" in t for t in dropped_texts)  # siempreSeVe: descartado


def test_warning_dispara_para_clase_de_tabla_dentro_de_blockquote() -> None:
    # Una celda de tabla citada en un blockquote debe AFLORAR (warning), no perderse en silencio.
    xml = (
        '<version><p class="parrafo">Cuerpo vigente.</p>'
        '<blockquote class="soloTexto"><p class="cuerpo_tabla_centro">Celda citada</p>'
        "</blockquote></version>"
    )
    assert is_suspicious_blockquote_class("cuerpo_tabla_centro")
    captured: list[str] = []
    sink_id = logger.add(captured.append, level="WARNING", format="{message}")
    try:
        paragraphs, _notes = _parse_version_paragraphs(etree.fromstring(xml))
    finally:
        logger.remove(sink_id)
    assert [p["text"] for p in paragraphs] == ["Cuerpo vigente."]  # la celda no entra al cuerpo
    assert any("cuerpo_tabla_centro" in m for m in captured)


def test_auditor_regex_discrimina_editorial_de_prosa_legitima() -> None:
    assert EDITORIAL_LEAK_RE.search("... Téngase en cuenta que esto entra en vigor ...")
    assert EDITORIAL_LEAK_RE.search('Redacción anterior: "texto"')
    assert EDITORIAL_LEAK_RE.search("Redacciones anteriores:")
    # NO debe marcar la prosa legítima de una transitoria (sin dos puntos tras 'anterior').
    assert not EDITORIAL_LEAK_RE.search("La redacción anterior de este artículo se aplicará")
