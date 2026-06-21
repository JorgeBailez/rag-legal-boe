"""Tests del índice BM25 (offline): ranking por solape léxico, score 0 y máscara de filtros."""

import numpy as np
import pytest

from src.indexing.lexical_index import LexicalIndex, row_boost_text


def _row(eid: str, parent_id: str) -> dict:
    return {
        "embedding_input_id": eid,
        "document_id": "BOE-A-0001",
        "block_id": parent_id.split("__")[-1],
        "parent_id": parent_id,
        "source": {
            "kind": "chunk_field",
            "chunk_id": f"{parent_id}__c001",
            "field": "retrieval_text",
        },
        "context_anchor": {"paragraph_start": 1, "paragraph_end": 1},
    }


def _index() -> LexicalIndex:
    rows = [_row("e1", "p1"), _row("e2", "p2"), _row("e3", "p3")]
    texts = [
        "Los contratos menores de obras tienen un umbral de 40.000 euros.",
        "El padrón municipal acredita la residencia de los vecinos.",
        "La garantía definitiva es del 5 por 100 del precio del contrato.",
    ]
    return LexicalIndex(rows=rows, texts=texts)


def test_ranking_por_solape_lexico() -> None:
    hits = _index().search("umbral del contrato menor de obras", k=3)
    assert hits[0]["parent_id"] == "p1"
    assert hits[0]["rank"] == 1


def test_sin_solape_no_devuelve_hits() -> None:
    assert _index().search("teléfono extraterrestre galáctico", k=3) == []


def test_mask_excluye_candidatos() -> None:
    mask = np.array([False, True, True])  # excluye la row 0 (la de contratos)
    hits = _index().search("umbral contrato menor obras 40.000", k=3, mask=mask)
    assert all(h["parent_id"] != "p1" for h in hits)


def test_longitudes_desiguales_lanzan_error() -> None:
    with pytest.raises(ValueError):
        LexicalIndex(rows=[_row("e1", "p1")], texts=[])


def test_row_boost_text_incluye_ley_y_titulo() -> None:
    rows = [_row("e1", "p1"), _row("e2", "p2"), _row("e3", "p3")]
    corpus = {
        "parents_by_id": {
            "p1": {"citation": {"label": "Ley 39/2015"}, "full_title": "Artículo 1. Objeto"},
            "p2": {"citation": {"label": "Ley 40/2015"}, "title": "Artículo 2"},  # cae a title
            "p3": {},  # sin ley ni título → ""
        }
    }
    assert row_boost_text(rows, corpus) == [
        "Ley 39/2015 Artículo 1. Objeto",
        "Ley 40/2015 Artículo 2",
        "",
    ]


def test_boost_con_ley_desambigua_articulos_homonimos() -> None:
    # Dos "Artículo 122" de leyes distintas; la pregunta cita la ley. El boost (ley + título) debe
    # desempatar hacia la ley correcta — no solo por el nº, que colisiona (caso real q0077).
    rows = [_row("e1", "L39__a122"), _row("e2", "L40__a122"), _row("e3", "f1"), _row("e4", "f2")]
    texts = [
        "Ley 39/2015. Artículo 122. Recurso de alzada y potestativo de reposición.",
        "Ley 40/2015. Artículo 122. Conflictos de atribuciones entre órganos.",
        "El padrón municipal acredita la residencia de los vecinos.",
        "La garantía definitiva es del cinco por ciento del precio del contrato.",
    ]
    boost = [
        "Ley 39/2015 Artículo 122. Recurso de alzada",
        "Ley 40/2015 Artículo 122. Conflictos de atribuciones",
        "",
        "",
    ]
    idx = LexicalIndex(rows=rows, texts=texts, headings=boost, heading_boost=3)
    hits = idx.search("artículo 122 de la Ley 39/2015", k=2)
    assert hits[0]["parent_id"] == "L39__a122"  # la ley citada desempata hacia el a122 correcto


def test_heading_boost_sube_el_bloque_correcto() -> None:
    # El nº de artículo SOLO vive en la cabecera de p1 (no en su cuerpo); p2 lo repite en el cuerpo.
    # Sin boost, p1 ni es candidato (sin solape); con boost de cabecera, p1 sube al #1.
    rows = [_row("e1", "p1"), _row("e2", "p2"), _row("e3", "p3")]
    texts = [
        "Recursos contra las resoluciones que ponen fin al procedimiento.",  # p1: sin "122"
        "El artículo 122 regula el plazo; el artículo 122 fija recursos.",  # p2: 122 en el cuerpo
        "El padrón municipal acredita la residencia de los vecinos.",  # p3: relleno
    ]
    headings = ["Artículo 122. Recursos", "Artículo 7. Plazos", "Artículo 16. Padrón"]

    sin_boost = LexicalIndex(rows=rows, texts=texts)  # cabeceras por defecto = ""
    hits_sin = sin_boost.search("artículo 122", k=3)
    assert "p1" not in [h["parent_id"] for h in hits_sin]  # sin el nº en el cuerpo no hay solape

    con_boost = LexicalIndex(rows=rows, texts=texts, headings=headings, heading_boost=3)
    hits_con = con_boost.search("artículo 122", k=3)
    assert hits_con[0]["parent_id"] == "p1"  # el boost de cabecera lo coloca el primero


def test_heading_boost_cero_no_cambia_nada() -> None:
    rows = [_row("e1", "p1"), _row("e2", "p2"), _row("e3", "p3")]
    texts = [
        "Los contratos menores de obras tienen un umbral de 40.000 euros.",
        "El padrón municipal acredita la residencia de los vecinos.",
        "La garantía definitiva es del 5 por 100 del precio del contrato.",
    ]
    headings = ["Artículo 118. Contrato menor", "Artículo 17. Padrón", "Artículo 107. Garantía"]
    q = "umbral del contrato menor de obras"
    base = LexicalIndex(rows=rows, texts=texts).search(q, k=3)
    cero = LexicalIndex(rows=rows, texts=texts, headings=headings, heading_boost=0).search(q, k=3)
    assert [h["parent_id"] for h in base] == [h["parent_id"] for h in cero]
    assert [round(h["score"], 6) for h in base] == [round(h["score"], 6) for h in cero]
