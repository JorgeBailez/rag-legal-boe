"""Tests del evidence builder (dedup, IDs deterministas, contexto acotado, presupuestos)."""

import pytest

from src.core.exceptions import ConfigurationError
from src.retrieval.context_assembler import P_EXPAND_BOUNDED
from src.retrieval.evidence_builder import build_evidences
from tests.generation_fakes import make_corpus_for_parents, make_hit


def _corpus_with_big_paragraphs(parent_ids: list[str], chars: int = 100) -> dict:
    body = "x" * chars
    parents_by_id = {
        pid: {
            "parent_id": pid,
            "document_id": pid.split("__")[0],
            "block_id": pid.split("__")[-1],
            "paragraphs": [{"order": 1, "class": "parrafo", "text": body}],
        }
        for pid in parent_ids
    }
    return {"chunks": [], "parents_by_id": parents_by_id, "documents_by_id": {}}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_evidences": 0},
        {"context_budget_chars": 0},
        {"max_total_context_chars": 0},
        {"context_strategy": "NOPE"},
    ],
)
def test_build_evidences_rejects_invalid_public_params(kwargs: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        build_evidences([], parents_by_id={}, **kwargs)


def test_dedup_by_parent_keeps_best_rank() -> None:
    corpus = make_corpus_for_parents(["BOE-A-0001__a1"])
    hits = [
        make_hit(rank=1, parent_id="BOE-A-0001__a1", score=0.9, text="mejor"),
        make_hit(rank=3, parent_id="BOE-A-0001__a1", score=0.5, text="peor"),
    ]
    sel = build_evidences(hits, parents_by_id=corpus["parents_by_id"])
    assert len(sel.evidences) == 1
    assert sel.evidences[0].retrieval_rank == 1
    assert sel.duplicate_parents_removed == 1


def test_evidence_ids_are_deterministic_in_rank_order() -> None:
    corpus = make_corpus_for_parents(["d__a1", "d__a2", "d__a3"])
    hits = [
        make_hit(rank=2, parent_id="d__a2"),
        make_hit(rank=1, parent_id="d__a1"),
        make_hit(rank=3, parent_id="d__a3"),
    ]
    sel = build_evidences(hits, parents_by_id=corpus["parents_by_id"])
    assert [e.evidence_id for e in sel.evidences] == ["E1", "E2", "E3"]
    assert [e.parent_id for e in sel.evidences] == ["d__a1", "d__a2", "d__a3"]


def test_p_expand_bounded_is_used_and_expands_around_anchor() -> None:
    corpus = make_corpus_for_parents(["d__a1"])  # parent con 2 párrafos
    hits = [make_hit(rank=1, parent_id="d__a1", anchor={"paragraph_start": 1, "paragraph_end": 1})]
    sel = build_evidences(
        hits, parents_by_id=corpus["parents_by_id"], context_strategy=P_EXPAND_BOUNDED
    )
    ev = sel.evidences[0]
    assert ev.context_strategy == P_EXPAND_BOUNDED
    assert ev.paragraph_orders == [1, 2]  # expandió desde el anchor


def test_max_evidences_limits_selection() -> None:
    pids = [f"d__a{i}" for i in range(1, 6)]
    corpus = make_corpus_for_parents(pids)
    hits = [make_hit(rank=i, parent_id=pid) for i, pid in enumerate(pids, start=1)]
    sel = build_evidences(hits, parents_by_id=corpus["parents_by_id"], max_evidences=2)
    assert len(sel.evidences) == 2
    assert [e.evidence_id for e in sel.evidences] == ["E1", "E2"]


def test_aggregate_budget_omits_without_truncating() -> None:
    pids = ["d__a1", "d__a2"]
    corpus = _corpus_with_big_paragraphs(pids, chars=100)
    hits = [make_hit(rank=i, parent_id=pid) for i, pid in enumerate(pids, start=1)]
    sel = build_evidences(
        hits,
        parents_by_id=corpus["parents_by_id"],
        max_total_context_chars=150,  # cabe una evidencia (100), no dos (200)
    )
    assert len(sel.evidences) == 1
    assert sel.evidences[0].char_count == 100  # texto íntegro, sin truncar
    assert sel.total_char_count == 100
    assert sel.omitted and sel.omitted[0]["reason"] == "exceeds_total_context_budget"
    assert sel.omitted[0]["parent_id"] == "d__a2"


def test_missing_parent_is_diagnosed_and_skipped() -> None:
    corpus = make_corpus_for_parents(["d__a1"])
    hits = [
        make_hit(rank=1, parent_id="d__a1"),
        make_hit(rank=2, parent_id="d__missing"),
    ]
    sel = build_evidences(hits, parents_by_id=corpus["parents_by_id"])
    assert [e.parent_id for e in sel.evidences] == ["d__a1"]
    assert any(o["reason"] == "parent_not_found" for o in sel.omitted)


def test_backfill_skips_missing_parent_at_rank1_and_selects_later_ranks() -> None:
    # parent inexistente en rank 1 + max_evidences=2 → selecciona ranks 2 y 3 (backfill).
    corpus = make_corpus_for_parents(["d__a2", "d__a3"])  # falta d__a1
    hits = [
        make_hit(rank=1, parent_id="d__a1"),  # parent ausente
        make_hit(rank=2, parent_id="d__a2"),
        make_hit(rank=3, parent_id="d__a3"),
    ]
    sel = build_evidences(hits, parents_by_id=corpus["parents_by_id"], max_evidences=2)
    assert [e.retrieval_rank for e in sel.evidences] == [2, 3]
    assert [e.evidence_id for e in sel.evidences] == ["E1", "E2"]
    assert any(o["reason"] == "parent_not_found" and o["retrieval_rank"] == 1 for o in sel.omitted)


def test_individual_evidence_over_budget_is_omitted_with_diagnostic() -> None:
    corpus = _corpus_with_big_paragraphs(["d__a1"], chars=200)
    # retrieval_text vacío + párrafo del anchor > presupuesto por evidencia → over_budget.
    hits = [make_hit(rank=1, parent_id="d__a1", text="")]
    sel = build_evidences(hits, parents_by_id=corpus["parents_by_id"], context_budget_chars=50)
    assert sel.evidences == []
    assert sel.omitted and sel.omitted[0]["reason"] == "exceeds_evidence_context_budget"
    assert sel.omitted[0]["char_count"] == 200  # texto íntegro reportado, no truncado


def test_empty_context_is_omitted_with_diagnostic() -> None:
    from src.retrieval.context_assembler import K_ONLY

    corpus = make_corpus_for_parents(["d__a1"])
    # K_ONLY + retrieval_text vacío + anchor que no resuelve párrafos → contexto vacío.
    hits = [
        make_hit(
            rank=1, parent_id="d__a1", text="", anchor={"paragraph_start": 99, "paragraph_end": 99}
        )
    ]
    sel = build_evidences(hits, parents_by_id=corpus["parents_by_id"], context_strategy=K_ONLY)
    assert sel.evidences == []
    assert sel.omitted and sel.omitted[0]["reason"] == "empty_context"


def test_aggregate_budget_backfills_smaller_later_evidence() -> None:
    # Evidencia grande (rank 1) omitida por presupuesto agregado; la pequeña posterior entra.
    parents_by_id = {
        "d__a1": {
            "parent_id": "d__a1",
            "document_id": "d",
            "block_id": "a1",
            "paragraphs": [{"order": 1, "class": "parrafo", "text": "x" * 100}],
        },
        "d__a2": {
            "parent_id": "d__a2",
            "document_id": "d",
            "block_id": "a2",
            "paragraphs": [{"order": 1, "class": "parrafo", "text": "y" * 20}],
        },
    }
    hits = [make_hit(rank=1, parent_id="d__a1"), make_hit(rank=2, parent_id="d__a2")]
    sel = build_evidences(hits, parents_by_id=parents_by_id, max_total_context_chars=50)
    assert [e.parent_id for e in sel.evidences] == ["d__a2"]  # la pequeña
    assert sel.evidences[0].evidence_id == "E1"
    assert sel.total_char_count == 20
    assert any(
        o["reason"] == "exceeds_total_context_budget" and o["parent_id"] == "d__a1"
        for o in sel.omitted
    )
