"""Refinaciones de la auditoría afloradas al ampliar el corpus 10→92 (offline, núcleo puro).

Al crecer el corpus aparecieron FALSOS POSITIVOS de la auditoría sobre bloques legítimamente
EXCLUIDOS (nota_inicial, encabezados "[Información relacionada]" sin cuerpo → parent_id None,
versión vigente irrelevante) y sobre contenido legítimo (chunk_id de 4 dígitos en anexos con
>1000 chunks; "<P20" — agudeza visual del baremo — confundido con etiqueta). Estos tests fijan
que esos casos ya NO se marcan ERROR, sin aflojar los checks reales sobre bloques indexables.
"""

from src.boe.parser import _full_title
from src.quality.corpus_audit import CHUNK_ID, XML_TAG, _check_blocks, _check_temporal

DID = "BOE-A-2015-11719"
PROC_DATE = "2026-06-22"


# --- XML_TAG: etiqueta bien formada (con '>'), no un '<' literal -------------


def test_xml_tag_no_marca_menor_que_literal() -> None:
    assert XML_TAG.search("AGUDEZAVISUAL | P20 | <P20 | 0") is None
    assert XML_TAG.search("el valor es < que 3") is None


def test_xml_tag_si_marca_etiqueta_real() -> None:
    assert XML_TAG.search('<p class="parrafo">x</p>') is not None
    assert XML_TAG.search("texto</td>") is not None


# --- CHUNK_ID: >=3 dígitos (anexos con >1000 chunks, p. ej. baremo) ----------


def test_chunk_id_admite_cuatro_digitos() -> None:
    assert CHUNK_ID.match("BOE-A-2004-18911__an__c1000")
    assert CHUNK_ID.match("BOE-A-2015-10565__a1-30__c001")


def test_chunk_id_rechaza_menos_de_tres_digitos() -> None:
    assert CHUNK_ID.match("BOE-A-2004-18911__an__c12") is None


# --- _full_title: rótulo en clase de título SIN número -----------------------


def test_full_title_fallback_titulo_sin_numero() -> None:
    # El BOE a veces mal-etiqueta "CAPÍTULO III" como capitulo_tit (sin capitulo_num).
    paragraphs = [{"class": "capitulo_tit", "text": "CAPÍTULO III"}]
    assert _full_title("encabezado", paragraphs) == "CAPÍTULO III"


def test_full_title_par_num_tit_sin_cambios() -> None:
    paragraphs = [
        {"class": "capitulo_num", "text": "CAPÍTULO III"},
        {"class": "capitulo_tit", "text": "Garantías jurídicas"},
    ]
    assert _full_title("encabezado", paragraphs) == "CAPÍTULO III. Garantías jurídicas"


# --- guard block.parent_id: None legítimo en excluidos -----------------------


def test_parent_id_none_no_se_marca() -> None:
    doc = {
        "metadata": {},
        "blocks": [
            {
                "block_id": "co",
                "parent_id": None,
                "block_type": "nota_inicial",
                "retrieval": {"indexable": False},
            }
        ],
    }
    findings = _check_blocks(doc, DID, PROC_DATE)
    assert not any(f["check"] == "block.parent_id" for f in findings)


def test_parent_id_malformado_si_se_marca() -> None:
    doc = {
        "metadata": {},
        "blocks": [
            {
                "block_id": "a1",
                "parent_id": "MAL",
                "block_type": "precepto",
                "retrieval": {"indexable": True},
            }
        ],
    }
    findings = _check_blocks(doc, DID, PROC_DATE)
    assert any(f["check"] == "block.parent_id" for f in findings)


# --- guard block.temporal_mismatch: solo en bloques indexables ---------------


def _temporal_block(*, indexable: bool) -> dict:
    """Bloque resuelto cuya latest_version NO es la vigente por índice (mismatch)."""
    return {
        "block_id": "co",
        "block_type": "nota_inicial",
        "retrieval": {"indexable": indexable},
        "index_last_update_date": "2021-01-01",
        "versions": [
            {"publication_date": "2020-01-01", "source_norm_id": "X", "is_latest": False},
            {"publication_date": "2021-01-01", "source_norm_id": "Y", "is_latest": True},
        ],
        "latest_version": {"publication_date": "2020-01-01", "source_norm_id": "X"},
    }


def test_temporal_mismatch_no_se_marca_en_excluido() -> None:
    findings = _check_temporal(_temporal_block(indexable=False), DID, PROC_DATE)
    assert not any(f["check"] == "block.temporal_mismatch" for f in findings)


def test_temporal_mismatch_si_se_marca_en_indexable() -> None:
    findings = _check_temporal(_temporal_block(indexable=True), DID, PROC_DATE)
    assert any(f["check"] == "block.temporal_mismatch" for f in findings)
