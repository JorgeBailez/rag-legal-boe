"""Tests del context assembler (K_ONLY, P_EXPAND_FULL, P_EXPAND_BOUNDED) y presupuestos."""

from src.retrieval.context_assembler import (
    B8K,
    K_ONLY,
    P_EXPAND_BOUNDED,
    P_EXPAND_FULL,
    assemble_context,
)


def _parent(n_paras: int, chars_each: int = 50) -> dict:
    body = "x" * chars_each
    return {
        "parent_id": "BOE-A-0001__a1",
        "paragraphs": [
            {"order": i, "class": "parrafo", "text": f"{i}:{body}"} for i in range(1, n_paras + 1)
        ],
    }


def test_k_only_returns_retrieved_text() -> None:
    parent = _parent(6)
    r = assemble_context(
        strategy=K_ONLY,
        parent=parent,
        anchor={"paragraph_start": 2, "paragraph_end": 3},
        retrieved_text="texto recuperado",
        budget_chars=B8K,
    )
    assert r.text == "texto recuperado"
    assert r.char_count == len("texto recuperado")
    assert r.base_char_count == r.char_count  # ExpansionRatio == 1 en K_ONLY


def test_p_expand_full_returns_whole_parent() -> None:
    parent = _parent(4)
    r = assemble_context(
        strategy=P_EXPAND_FULL, parent=parent, anchor={"paragraph_start": 2, "paragraph_end": 2}
    )
    assert r.paragraph_orders == [1, 2, 3, 4]
    assert r.item_count == 4


def test_p_expand_bounded_grows_within_budget_and_keeps_order() -> None:
    parent = _parent(10, chars_each=50)  # cada párrafo ~ 52 chars
    anchor = {"paragraph_start": 5, "paragraph_end": 5}
    small = assemble_context(
        strategy=P_EXPAND_BOUNDED, parent=parent, anchor=anchor, budget_chars=200
    )
    big = assemble_context(
        strategy=P_EXPAND_BOUNDED, parent=parent, anchor=anchor, budget_chars=100000
    )
    # presupuesto mayor ⇒ más párrafos, pero nunca cruza al parent ni desordena
    assert len(big.paragraph_orders) > len(small.paragraph_orders)
    assert small.char_count <= 200
    assert big.paragraph_orders == sorted(big.paragraph_orders)
    assert 5 in small.paragraph_orders  # el anchor siempre está incluido
    assert big.paragraph_orders == list(range(1, 11))  # con presupuesto enorme, todo el parent


def test_bounded_keeps_anchor_even_if_over_budget() -> None:
    parent = _parent(5, chars_each=500)
    anchor = {"paragraph_start": 3, "paragraph_end": 3}
    r = assemble_context(strategy=P_EXPAND_BOUNDED, parent=parent, anchor=anchor, budget_chars=10)
    assert r.paragraph_orders == [3]  # no se trunca el anchor; no se añade nada más
    assert r.over_budget is True
    assert r.fallback_reason == "anchor_exceeds_budget"


def test_bounded_uses_retrieved_text_fallback_when_anchor_exceeds_budget() -> None:
    parent = _parent(5, chars_each=500)
    anchor = {"paragraph_start": 3, "paragraph_end": 3}
    r = assemble_context(
        strategy=P_EXPAND_BOUNDED,
        parent=parent,
        anchor=anchor,
        retrieved_text="segmento compacto",
        budget_chars=100,
    )
    assert r.text == "segmento compacto"
    assert r.paragraph_orders == [3]
    assert r.over_budget is False
    assert r.fallback_reason == "anchor_exceeds_budget_retrieved_text_fallback"


def test_bounded_fallback_keeps_refined_segment_paragraph_orders() -> None:
    parent = _parent(2, chars_each=500)
    r = assemble_context(
        strategy=P_EXPAND_BOUNDED,
        parent=parent,
        anchor={"paragraph_start": 1, "paragraph_end": 1},
        retrieved_text="segmento del parrafo 1",
        budget_chars=100,
    )
    assert r.text == "segmento del parrafo 1"
    assert r.paragraph_orders == [1]
    assert r.fallback_reason == "anchor_exceeds_budget_retrieved_text_fallback"


def test_bounded_requires_anchor() -> None:
    parent = _parent(5)
    try:
        assemble_context(strategy=P_EXPAND_BOUNDED, parent=parent, anchor=None, budget_chars=100)
    except ValueError as exc:
        assert "context_anchor" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("P_EXPAND_BOUNDED debe exigir context_anchor")
