"""Integración extremo a extremo del flujo denso con encoder/tokenizer falsos (offline)."""

import numpy as np

from src.embeddings.bundle import ExecutionMeta, publish_bundle
from src.embeddings.fingerprints import source_corpus_fingerprint
from src.embeddings.input_preparation import prepare_inputs
from src.embeddings.validation import has_errors, run_gate_a
from src.indexing.vector_index import ExactDenseIndex, build_filter_mask
from src.retrieval.context_assembler import P_EXPAND_BOUNDED, assemble_context
from tests.dense_fakes import FakeEncoder, FakeWordTokenizer, synthetic_corpus
from tests.test_bundle import TEST_CONTRACT


def test_end_to_end_prepare_encode_publish_query_assemble(tmp_path) -> None:
    corpus = synthetic_corpus()
    tok = FakeWordTokenizer(model_max_length=512, special=2)

    # 1) preparar inputs (J1)
    prepared = prepare_inputs(
        "J1",
        chunks=corpus["chunks"],
        parents_by_id=corpus["parents_by_id"],
        contract=TEST_CONTRACT,
        tokenizer=tok,
    )
    assert prepared.rows and prepared.report["n_truncated"] == 0

    # 2) Gate A con readiness simulada (lista)
    gate_a = run_gate_a(
        readiness={"ready": True, "blocking_findings": []},
        contract=TEST_CONTRACT,
        allow_unpinned_revision=False,
        prepared=prepared,
    )
    assert not has_errors(gate_a)

    # 3) codificar con encoder falso
    encoder = FakeEncoder(dimension=8, contract=TEST_CONTRACT)
    embeddings = encoder.encode_documents(prepared.texts)

    # 4) publicar bundle (Gate B incluido)
    result = publish_bundle(
        contract=TEST_CONTRACT,
        view="J1",
        prepared=prepared,
        embeddings=embeddings,
        source_corpus_fingerprint=source_corpus_fingerprint(
            corpus["chunks"], corpus["parents_by_id"]
        ),
        n_norms=2,
        execution=ExecutionMeta(encoder_backend="fake"),
        output_root=tmp_path,
        gate_a_findings=gate_a,
    )
    assert result["validation_report"]["gate_b_passed"]

    # 5) cargar índice y consultar
    index = ExactDenseIndex.from_bundle(result["path"], corpus=corpus)
    qv = encoder.encode_queries(["plazo administrativo"])[0]
    hits = index.search(qv, k=3)
    assert hits and all("parent_id" in h for h in hits)

    # 6) filtro (anexo) → restringe candidatos
    mask = build_filter_mask(index.rows, corpus, {"annex": True})
    filtered = index.search(qv, k=5, mask=mask)
    assert all(h["block_id"] == "anexo" for h in filtered)

    # 7) ensamblar contexto del primer hit
    top = hits[0]
    parent = corpus["parents_by_id"][top["parent_id"]]
    ctx = assemble_context(
        strategy=P_EXPAND_BOUNDED,
        parent=parent,
        anchor=top.get("context_anchor"),
        budget_chars=8000,
    )
    assert ctx.text and ctx.char_count > 0
    assert ctx.paragraph_orders == sorted(ctx.paragraph_orders)
    # embeddings recargados por mmap son float32 y solo lectura
    assert np.asarray(index.embeddings).dtype == np.float32
