"""Tests ligeros del script de benchmark/smoke, sin pesos reales."""

import pytest

from scripts.benchmark_dense_models import (
    _budget_runs_for_strategy,
    _hits_for_context_strategy,
    _latency_summary_ms,
    _prepare_smoke_documents,
    _query_result_hit,
    _resolve_hit_text,
    _select_representative_chunks,
)
from scripts.query_dense_index import _resolve
from src.embeddings.model_registry import (
    effective_query_profile_ids,
    format_query_with_profile,
    get_contract,
    get_query_profile,
    query_profile_fingerprint,
    query_profile_metadata,
)
from src.retrieval.context_assembler import K_ONLY, P_EXPAND_BOUNDED, P_EXPAND_FULL
from tests.dense_fakes import FakeWordTokenizer, synthetic_corpus


def test_smoke_documents_apply_document_formatter() -> None:
    corpus = synthetic_corpus()
    texts, report = _prepare_smoke_documents(
        get_contract("e5-base"),
        corpus,
        FakeWordTokenizer(),
        n_docs=3,
    )
    assert texts
    assert report["n_source_chunks"] <= 3
    assert all(t.startswith("passage: ") for t in texts)


def test_representative_sample_is_deterministic() -> None:
    chunks = synthetic_corpus()["chunks"]
    a = _select_representative_chunks(chunks, 3)
    b = _select_representative_chunks(list(reversed(chunks)), 3)
    assert [c["chunk_id"] for c in a] == [c["chunk_id"] for c in b]


def test_resolve_hit_text_prefers_derived_text_over_chunk() -> None:
    corpus = synthetic_corpus()
    hit = {
        "parent_id": "BOE-A-0002__a1",
        "source": {
            "kind": "derived_text",
            "origin": "overflow_repair",
            "chunk_id": "BOE-A-0002__a1__c001",
            "text": "segmento derivado",
        },
    }
    assert _resolve_hit_text(hit, corpus) == "segmento derivado"
    assert _resolve(hit, corpus)[0] == "segmento derivado"


def test_expanded_context_deduplicates_parents_but_k_only_keeps_rows() -> None:
    hits = [
        {"parent_id": "p1", "row_index": 0},
        {"parent_id": "p1", "row_index": 1},
        {"parent_id": "p2", "row_index": 2},
    ]
    assert _hits_for_context_strategy(K_ONLY, hits) == hits
    assert [h["row_index"] for h in _hits_for_context_strategy(P_EXPAND_FULL, hits)] == [0, 2]
    assert [h["row_index"] for h in _hits_for_context_strategy(P_EXPAND_BOUNDED, hits)] == [0, 2]


def test_context_budget_runs_only_vary_for_bounded() -> None:
    assert _budget_runs_for_strategy(K_ONLY) == [(None, 8000)]
    assert _budget_runs_for_strategy(P_EXPAND_FULL) == [(None, 8000)]
    assert [name for name, _ in _budget_runs_for_strategy(P_EXPAND_BOUNDED)] == [
        "B4K",
        "B8K",
        "B12K",
    ]


def test_non_task_models_accept_only_baseline_profile() -> None:
    for alias in ("e5-base", "e5-large", "bge-m3", "gte-multilingual-base"):
        contract = get_contract(alias)
        assert effective_query_profile_ids(contract, None) == ["BASELINE"]
        assert format_query_with_profile(contract, "plazo", "BASELINE") == contract.format_query(
            "plazo"
        )
        with pytest.raises(ValueError, match="BASELINE"):
            effective_query_profile_ids(contract, ["I1_LEGAL"])


def test_instruct_models_keep_distinct_effective_profiles() -> None:
    e5 = get_contract("e5-large-instruct")
    assert effective_query_profile_ids(e5, None) == [
        "I0_GENERIC",
        "I1_LEGAL",
        "I2_CITIZEN_LEGISLATION",
    ]
    e5_inputs = [
        format_query_with_profile(e5, "plazo", p) for p in effective_query_profile_ids(e5, None)
    ]
    assert len(set(e5_inputs)) == 3
    with pytest.raises(ValueError, match="qwen3-0.6b|permitidos"):
        effective_query_profile_ids(e5, ["I_MINUS_NONE"])

    qwen = get_contract("qwen3-0.6b")
    assert effective_query_profile_ids(qwen, None) == [
        "I0_GENERIC",
        "I1_LEGAL",
        "I2_CITIZEN_LEGISLATION",
        "I_MINUS_NONE",
    ]
    assert format_query_with_profile(qwen, "hola", "I_MINUS_NONE") == "hola"
    assert query_profile_metadata(qwen, "I_MINUS_NONE")["query_template"] == "{query}"


def test_i_minus_none_has_distinct_fingerprint_for_qwen() -> None:
    qwen = get_contract("qwen3-0.6b")
    raw = query_profile_fingerprint(qwen, "I_MINUS_NONE")
    instructed = {
        query_profile_fingerprint(qwen, p)
        for p in ("I0_GENERIC", "I1_LEGAL", "I2_CITIZEN_LEGISLATION")
    }
    assert raw not in instructed


def test_query_profile_instructions_are_frozen_exact_strings() -> None:
    assert get_query_profile("I0_GENERIC").instruction == (
        "Given a web search query, retrieve relevant passages that answer the query"
    )
    assert get_query_profile("I1_LEGAL").instruction == (
        "Given a user question about legislation, retrieve the relevant legal passages "
        "that help answer the question"
    )
    assert get_query_profile("I2_CITIZEN_LEGISLATION").instruction == (
        "Given a citizen question about legislation, retrieve the legal passages needed "
        "to answer it accurately"
    )


def test_requested_profiles_are_deduplicated_without_collapsing_effective_inputs() -> None:
    contract = get_contract("e5-large-instruct")
    assert effective_query_profile_ids(
        contract, ["I1_LEGAL", "I1_LEGAL", "I2_CITIZEN_LEGISLATION"]
    ) == [
        "I1_LEGAL",
        "I2_CITIZEN_LEGISLATION",
    ]


def test_latency_summary_uses_milliseconds_and_sample_count() -> None:
    summary = _latency_summary_ms([1.0, 3.0, 5.0], [2.0, 4.0, 6.0])
    assert summary["query_embedding_latency_p50_ms"] == 3.0
    assert summary["query_embedding_latency_p95_ms"] == 4.8
    assert summary["exact_search_latency_p50_ms"] == 4.0
    assert summary["exact_search_latency_p95_ms"] == 5.8
    assert summary["latency_sample_count"] == 3


def test_query_result_hit_has_traceability_without_heavy_text() -> None:
    hit = {
        "rank": 1,
        "row_index": 7,
        "embedding_input_id": "ein_000007",
        "parent_id": "p1",
        "score": 0.123456,
        "source": {"kind": "derived_text", "chunk_id": "c1", "text": "texto pesado"},
        "context_anchor": {"paragraph_start": 2, "paragraph_end": 2},
    }
    out = _query_result_hit(hit)
    assert out == {
        "rank": 1,
        "row_index": 7,
        "embedding_input_id": "ein_000007",
        "parent_id": "p1",
        "source_chunk_id": "c1",
        "context_anchor": {"paragraph_start": 2, "paragraph_end": 2},
        "score": 0.1235,
    }
    assert "text" not in out
