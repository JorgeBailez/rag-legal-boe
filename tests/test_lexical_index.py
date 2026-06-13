"""Tests del índice BM25 (offline): ranking por solape léxico, score 0 y máscara de filtros."""

import numpy as np
import pytest

from src.indexing.lexical_index import LexicalIndex


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
