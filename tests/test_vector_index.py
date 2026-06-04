"""Tests del ExactDenseIndex: búsqueda exacta, ranking estable, recarga mmap y filtros."""

import numpy as np

from src.embeddings.bundle import ExecutionMeta, publish_bundle
from src.embeddings.fingerprints import source_corpus_fingerprint
from src.embeddings.input_preparation import prepare_inputs
from src.indexing.vector_index import ExactDenseIndex, build_filter_mask
from tests.dense_fakes import FakeEncoder, FakeWordTokenizer, synthetic_corpus
from tests.test_bundle import TEST_CONTRACT


def _publish(tmp_path, embeddings=None):
    corpus = synthetic_corpus()
    tok = FakeWordTokenizer(model_max_length=512, special=2)
    prepared = prepare_inputs(
        "J1",
        chunks=corpus["chunks"],
        parents_by_id=corpus["parents_by_id"],
        contract=TEST_CONTRACT,
        tokenizer=tok,
    )
    emb = (
        embeddings
        if embeddings is not None
        else FakeEncoder(dimension=8).encode_documents(prepared.texts)
    )
    scfp = source_corpus_fingerprint(corpus["chunks"], corpus["parents_by_id"])
    result = publish_bundle(
        contract=TEST_CONTRACT,
        view="J1",
        prepared=prepared,
        embeddings=emb,
        source_corpus_fingerprint=scfp,
        n_norms=2,
        execution=ExecutionMeta(encoder_backend="fake"),
        output_root=tmp_path,
    )
    return result["path"], corpus, prepared


def test_self_query_returns_same_row_first(tmp_path) -> None:
    bundle_dir, corpus, prepared = _publish(tmp_path)
    index = ExactDenseIndex.from_bundle(bundle_dir, corpus=corpus)
    assert len(index) == len(prepared.rows)
    assert index.embeddings.dtype == np.float32
    # consultar con el propio vector de la fila 2 → debe salir primera con score ≈ 1.
    q = np.asarray(index.embeddings[2], dtype=np.float32)
    hits = index.search(q, k=3)
    assert hits[0]["row_index"] == 2
    assert hits[0]["score"] > 0.99
    # scores ordenados de forma descendente
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_ranking_is_stable_on_ties(tmp_path) -> None:
    # Dos vectores idénticos (filas 0 y 1) → empate; argsort estable preserva el orden de fila.
    base = np.zeros((4, 8), dtype=np.float32)
    v = np.ones(8, dtype=np.float32)
    v /= np.linalg.norm(v)
    base[0] = v
    base[1] = v
    base[2] = np.eye(8, dtype=np.float32)[3]
    base[3] = np.eye(8, dtype=np.float32)[5]
    bundle_dir, corpus, _ = _publish(tmp_path, embeddings=base)
    index = ExactDenseIndex.from_bundle(bundle_dir, corpus=corpus)
    hits = index.search(v, k=2)
    assert [h["row_index"] for h in hits] == [0, 1]  # empate resuelto por índice (estable)


def test_filter_mask_restricts_candidates(tmp_path) -> None:
    bundle_dir, corpus, prepared = _publish(tmp_path)
    index = ExactDenseIndex.from_bundle(bundle_dir, corpus=corpus)
    mask = build_filter_mask(index.rows, corpus, {"annex": True})
    assert mask.sum() == 1  # solo el bloque anexo del corpus sintético
    q = np.asarray(index.embeddings[0], dtype=np.float32)
    hits = index.search(q, k=5, mask=mask)
    assert len(hits) == 1
    assert hits[0]["block_id"] == "anexo"


def test_filter_subject_codes_membership(tmp_path) -> None:
    bundle_dir, corpus, _ = _publish(tmp_path)
    index = ExactDenseIndex.from_bundle(bundle_dir, corpus=corpus)
    # todas las rows del corpus sintético llevan subject 5703 → pasan todas
    assert build_filter_mask(index.rows, corpus, {"subject_codes": ["5703"]}).all()
    # materia inexistente → ninguna pasa
    assert not build_filter_mask(index.rows, corpus, {"subject_codes": ["9999"]}).any()
