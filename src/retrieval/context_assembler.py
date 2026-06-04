"""Ensamblado de contexto para el LLM a partir de un hit de retrieval (separado de retrieval/gen).

Tres estrategias:
- `K_ONLY`: solo el texto recuperado (chunk/ventana).
- `P_EXPAND_FULL`: el parent jurídico completo.
- `P_EXPAND_BOUNDED`: expande alrededor del `context_anchor` añadiendo párrafos alternando
  posterior/anterior, manteniendo el orden jurídico y sin superar un presupuesto de caracteres.

Presupuestos evaluables (B8K es candidato provisional, no decisión fija):
"""

from __future__ import annotations

from dataclasses import dataclass, field

K_ONLY = "K_ONLY"
P_EXPAND_FULL = "P_EXPAND_FULL"
P_EXPAND_BOUNDED = "P_EXPAND_BOUNDED"
STRATEGIES = (K_ONLY, P_EXPAND_FULL, P_EXPAND_BOUNDED)

B4K = 4000
B8K = 8000
B12K = 12000
BUDGETS = {"B4K": B4K, "B8K": B8K, "B12K": B12K}


@dataclass
class ContextResult:
    """Contexto ensamblado para un hit (un parent)."""

    strategy: str
    parent_id: str
    text: str
    paragraph_orders: list[int] = field(default_factory=list)
    char_count: int = 0
    item_count: int = 0
    base_char_count: int = 0
    over_budget: bool = False
    fallback_reason: str | None = None

    def as_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "parent_id": self.parent_id,
            "paragraph_orders": self.paragraph_orders,
            "char_count": self.char_count,
            "item_count": self.item_count,
            "base_char_count": self.base_char_count,
            "over_budget": self.over_budget,
            "fallback_reason": self.fallback_reason,
        }


def _anchor_orders(
    anchor: dict | None, orders_sorted: list[int], *, required: bool = False
) -> list[int]:
    """Órdenes de párrafo cubiertos por el anchor."""
    if anchor and orders_sorted:
        orders = [
            o for o in orders_sorted if anchor["paragraph_start"] <= o <= anchor["paragraph_end"]
        ]
        if orders:
            return orders
    if required:
        raise ValueError("P_EXPAND_BOUNDED requiere context_anchor")
    return []


def _join(by_order: dict[int, str], orders: list[int]) -> str:
    return "\n".join(by_order[o] for o in sorted(orders) if o in by_order)


def _expand_bounded(
    orders_sorted: list[int], anchor_orders: list[int], by_order: dict[int, str], budget: int
) -> list[int]:
    """Expande desde el anchor alternando posterior/anterior sin superar `budget` caracteres."""
    chosen = list(anchor_orders) or (orders_sorted[:1] if orders_sorted else [])
    if not chosen:
        return []
    lo = orders_sorted.index(chosen[0])
    hi = orders_sorted.index(chosen[-1])
    while True:
        progressed = False
        if hi + 1 < len(orders_sorted):
            cand = chosen + [orders_sorted[hi + 1]]
            if len(_join(by_order, cand)) <= budget:
                chosen, hi, progressed = cand, hi + 1, True
        if lo - 1 >= 0:
            cand = [orders_sorted[lo - 1]] + chosen
            if len(_join(by_order, cand)) <= budget:
                chosen, lo, progressed = cand, lo - 1, True
        if not progressed:
            break
    return sorted(chosen)


def assemble_context(
    *,
    strategy: str,
    parent: dict,
    anchor: dict | None = None,
    retrieved_text: str = "",
    budget_chars: int = B8K,
) -> ContextResult:
    """Ensambla el contexto de un hit según la estrategia. No reescribe el texto legal."""
    if strategy not in STRATEGIES:
        raise ValueError(f"estrategia desconocida: {strategy!r} (esperado {STRATEGIES})")

    paragraphs = parent.get("paragraphs") or []
    by_order = {p["order"]: p.get("text", "") for p in paragraphs}
    orders_sorted = sorted(by_order)
    parent_id = parent.get("parent_id", "")

    anchor_orders = _anchor_orders(anchor, orders_sorted)
    base_text = retrieved_text or _join(by_order, anchor_orders)
    base_chars = len(base_text)
    over_budget = False
    fallback_reason = None

    if strategy == K_ONLY:
        text = base_text
        orders = anchor_orders if not retrieved_text else (anchor_orders if anchor else [])
        item_count = len(orders) if orders else 1
    elif strategy == P_EXPAND_FULL:
        orders = orders_sorted
        text = _join(by_order, orders) or base_text
        item_count = len(orders)
    else:  # P_EXPAND_BOUNDED
        anchor_orders = _anchor_orders(anchor, orders_sorted, required=True)
        anchor_text = _join(by_order, anchor_orders)
        if len(anchor_text) > budget_chars:
            orders = anchor_orders
            if retrieved_text:
                text = retrieved_text
                fallback_reason = "anchor_exceeds_budget_retrieved_text_fallback"
            else:
                text = anchor_text
                fallback_reason = "anchor_exceeds_budget"
            over_budget = len(text) > budget_chars
        else:
            orders = _expand_bounded(orders_sorted, anchor_orders, by_order, budget_chars)
            text = _join(by_order, orders) or base_text
        item_count = len(orders)

    return ContextResult(
        strategy=strategy,
        parent_id=parent_id,
        text=text,
        paragraph_orders=list(orders),
        char_count=len(text),
        item_count=item_count,
        base_char_count=base_chars,
        over_budget=over_budget,
        fallback_reason=fallback_reason,
    )
