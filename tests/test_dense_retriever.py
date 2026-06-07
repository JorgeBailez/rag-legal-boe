"""Tests del wrapper de retrieval denso (joins precalculados, perfiles, filtros, fakes)."""

import argparse

import pytest

from src.embeddings.model_registry import get_contract
from src.retrieval.dense_retriever import DenseRetriever, resolve_hit_text_and_citation
from tests.dense_fakes import FakeEncoder
from tests.generation_fakes import build_bundle_retriever
from tests.test_bundle import TEST_CONTRACT


class _SpyEncoder:
    """Envuelve un encoder y registra el `query_profile_id` recibido en cada llamada."""

    def __init__(self, inner: FakeEncoder) -> None:
        self.inner = inner
        self.profiles: list[str | None] = []

    def encode_documents(self, texts, *, batch_size=None, show_progress=False):
        return self.inner.encode_documents(
            texts, batch_size=batch_size, show_progress=show_progress
        )

    def encode_queries(self, queries, *, query_profile_id=None, show_progress=False):
        self.profiles.append(query_profile_id)
        return self.inner.encode_queries(
            queries, query_profile_id=query_profile_id, show_progress=show_progress
        )


class _StubIndex:
    """Índice mínimo (sin búsqueda) para probar resolución de perfil con contrato instruct."""

    rows: list = []
    manifest = {"bundle": {"bundle_id": "stub__j1__0", "model_alias": "e5-large-instruct"}}

    def search(self, query_vector, *, k=5, mask=None):
        return []


def test_retrieve_resolves_text_and_citation_via_joins(tmp_path) -> None:
    encoder = FakeEncoder(dimension=8, contract=TEST_CONTRACT)
    retriever = build_bundle_retriever(tmp_path, TEST_CONTRACT, encoder)
    hits = retriever.retrieve("plazo administrativo", query_profile_id="BASELINE", top_k=3)
    assert hits, "esperaba al menos un hit"
    top = hits[0]
    # La cita (label/url) y el texto provienen del corpus, no del índice.
    assert top.citation_label and top.citation_label != top.parent_id
    assert top.citation_url and top.citation_url.startswith("https://")
    assert top.retrieval_text


def test_retrieve_forwards_query_profile_to_encoder(tmp_path) -> None:
    spy = _SpyEncoder(FakeEncoder(dimension=8, contract=TEST_CONTRACT))
    retriever = build_bundle_retriever(tmp_path, TEST_CONTRACT, spy)
    retriever.retrieve("plazo", query_profile_id="BASELINE", top_k=2)
    assert spy.profiles == ["BASELINE"]


def test_filters_restrict_candidates(tmp_path) -> None:
    encoder = FakeEncoder(dimension=8, contract=TEST_CONTRACT)
    retriever = build_bundle_retriever(tmp_path, TEST_CONTRACT, encoder)
    hits = retriever.retrieve(
        "anexo", query_profile_id="BASELINE", top_k=5, filters={"annex": True}
    )
    assert hits and all(h.block_id == "anexo" for h in hits)


def test_joins_are_precomputed_once_and_reused(tmp_path) -> None:
    encoder = FakeEncoder(dimension=8, contract=TEST_CONTRACT)
    retriever = build_bundle_retriever(tmp_path, TEST_CONTRACT, encoder)
    chunks_map = retriever._chunks_by_id
    assert chunks_map  # join construido una vez en el constructor
    retriever.retrieve("uno", query_profile_id="BASELINE", top_k=1)
    retriever.retrieve("dos", query_profile_id="BASELINE", top_k=1)
    # El mismo objeto de join se reutiliza entre llamadas (no se reconstruye por hit/consulta).
    assert retriever._chunks_by_id is chunks_map


def test_resolved_query_profile_id_with_instruct_contract() -> None:
    contract = get_contract("e5-large-instruct")
    retriever = DenseRetriever(
        index=_StubIndex(),
        encoder=FakeEncoder(dimension=1024, contract=contract),
        contract=contract,
        corpus={"chunks": [], "parents_by_id": {}, "documents_by_id": {}},
    )
    assert retriever.resolved_query_profile_id("I2_CITIZEN_LEGISLATION") == "I2_CITIZEN_LEGISLATION"
    # Acepta índice/encoder falsos sin red ni bundle real.
    assert retriever.retrieve("x", query_profile_id="I2_CITIZEN_LEGISLATION") == []


def _instruct_retriever() -> DenseRetriever:
    contract = get_contract("e5-large-instruct")
    return DenseRetriever(
        index=_StubIndex(),
        encoder=FakeEncoder(dimension=1024, contract=contract),
        contract=contract,
        corpus={"chunks": [], "parents_by_id": {}, "documents_by_id": {}},
    )


def test_retrieve_rejects_non_positive_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        _instruct_retriever().retrieve("plazo", top_k=-1)


def test_retrieve_rejects_blank_query() -> None:
    with pytest.raises(ValueError, match="query"):
        _instruct_retriever().retrieve("   ")


def test_query_dense_index_non_blank_helper_accepts_and_rejects() -> None:
    from scripts.query_dense_index import _non_blank

    assert _non_blank("¿Qué plazo tengo?") == "¿Qué plazo tengo?"
    for bad in ("", "   "):
        with pytest.raises(argparse.ArgumentTypeError):
            _non_blank(bad)


# --- resolución del campo realmente indexado (J1 vs J2 vs derived) ---------


def _chunks_by_id() -> dict[str, dict]:
    return {
        "c1": {
            "text": "texto crudo del child",
            "retrieval_text": "Contexto jurídico. texto crudo del child",
            "citation": {"label": "Ley 1/2000 a1", "url": "https://boe/x#a1"},
        }
    }


def test_resolve_chunk_field_uses_retrieval_text_for_j1() -> None:
    hit = {
        "parent_id": "p1",
        "source": {"kind": "chunk_field", "chunk_id": "c1", "field": "retrieval_text"},
    }
    text, citation = resolve_hit_text_and_citation(
        hit, chunks_by_id=_chunks_by_id(), parents_by_id={}
    )
    assert text == "Contexto jurídico. texto crudo del child"
    assert citation["label"] == "Ley 1/2000 a1"


def test_resolve_chunk_field_uses_text_for_j2() -> None:
    hit = {"parent_id": "p1", "source": {"kind": "chunk_field", "chunk_id": "c1", "field": "text"}}
    text, _ = resolve_hit_text_and_citation(hit, chunks_by_id=_chunks_by_id(), parents_by_id={})
    assert text == "texto crudo del child"


def test_resolve_chunk_field_falls_back_to_text_when_field_missing() -> None:
    chunks = {"c1": {"text": "solo crudo", "citation": {}}}  # sin retrieval_text (bundle antiguo)
    hit = {
        "parent_id": "p1",
        "source": {"kind": "chunk_field", "chunk_id": "c1", "field": "retrieval_text"},
    }
    text, _ = resolve_hit_text_and_citation(hit, chunks_by_id=chunks, parents_by_id={})
    assert text == "solo crudo"


def test_resolve_derived_text_keeps_source_text() -> None:
    hit = {
        "parent_id": "p1",
        "source": {
            "kind": "derived_text",
            "origin": "overflow_repair",
            "chunk_id": "c1",
            "text": "segmento derivado",
        },
    }
    text, _ = resolve_hit_text_and_citation(hit, chunks_by_id=_chunks_by_id(), parents_by_id={})
    assert text == "segmento derivado"
