"""Tests del recuperador léxico (offline): devuelve `DenseHit` resuelto, igual que el denso."""

import pytest

from src.indexing.lexical_index import LexicalIndex
from src.retrieval.lexical_retriever import LexicalRetriever


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


def _chunk(parent_id: str, text: str, label: str) -> dict:
    return {
        "chunk_id": f"{parent_id}__c001",
        "retrieval_text": text,
        "citation": {"label": label, "url": f"https://boe.es/{parent_id}"},
    }


def _retriever() -> LexicalRetriever:
    rows = [_row("e1", "BOE-A-0001__a1"), _row("e2", "BOE-A-0001__a2")]
    texts = ["Contrato menor de obras: umbral de 40.000 euros.", "Padrón municipal y residencia."]
    corpus = {
        "chunks": [
            _chunk("BOE-A-0001__a1", texts[0], "Ley 1/2000, artículo 1"),
            _chunk("BOE-A-0001__a2", texts[1], "Ley 1/2000, artículo 2"),
        ],
        "parents_by_id": {},
        "documents_by_id": {},
    }
    return LexicalRetriever(index=LexicalIndex(rows=rows, texts=texts), corpus=corpus)


def test_retrieve_devuelve_densehit_resuelto() -> None:
    hits = _retriever().retrieve("umbral contrato menor obras", top_k=2)
    assert hits[0].parent_id == "BOE-A-0001__a1"
    assert hits[0].rank == 1
    assert hits[0].citation_label == "Ley 1/2000, artículo 1"  # cita autoritativa del corpus
    assert "40.000" in hits[0].retrieval_text  # texto resuelto por join al chunk


def test_retrieve_query_vacia_lanza_error() -> None:
    with pytest.raises(ValueError):
        _retriever().retrieve("   ", top_k=2)
