"""Preparación de los inputs de embedding por vista (J1 / J2 / C1), sin pérdidas silenciosas.

Convierte el corpus de Fase 1 (chunks + parents) en *rows* derivadas (`dense_embedding_row_v1`) y
sus textos formateados listos para codificar. Es **lógica pura**: el tokenizador se inyecta
(cualquier objeto con `encode`/`decode`/`model_max_length`), de modo que es testeable con un *fake*.

Vistas:
- **J1** (baseline): `chunks[].retrieval_text`.
- **J2** (ablación de contexto jurídico): `chunks[].text`.
- **C1** (chunking clásico controlado): ventanas fijas token-aware dentro de cada parent
  (`parents[].text`), overlap 100 tokens, sin cruzar parents.

Regla innegociable: **truncamiento silencioso prohibido**. Antes de codificar, cada texto base se
formatea con el contrato del modelo, se cuenta con el tokenizador real (special tokens incluidos) y
se valida contra el límite efectivo. Si un input excede el límite se **repara** dividiéndolo en
ventanas token-aware (overlap 100). Si queda algún overflow sin reparar → error bloqueante.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Protocol

from src.embeddings.model_registry import ModelContract
from src.embeddings.tokenizer_profiler import count_tokens, resolve_effective_max

OVERLAP_TOKENS = 100
SAFETY_MARGIN_TOKENS = 16  # margen para roundtrip decode→encode de tokenizadores reales

VIEWS = ("J1", "J2", "C1")
_VIEW_FIELD = {"J1": "retrieval_text", "J2": "text"}


class OverflowNotRepairedError(RuntimeError):
    """Un input excede el límite efectivo incluso tras la reparación token-aware (bloqueante)."""


class AnchorResolutionError(RuntimeError):
    """No se pudo resolver un context_anchor único y accionable."""


class SplittableTokenizer(Protocol):
    """Tokenizador con conteo y decodificación (para dividir token-aware sin truncar)."""

    model_max_length: int

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


@dataclass
class PreparedInputs:
    """Resultado de preparar una vista: rows + textos a codificar (paralelos) + diagnóstico."""

    view: str
    model_alias: str
    rows: list[dict] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    effective_max_tokens: int = 0
    report: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Utilidades de tokens
# --------------------------------------------------------------------------- #


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _special_overhead(tokenizer: SplittableTokenizer) -> int:
    """Tokens especiales que añade el tokenizador ([CLS]/[SEP]…), medidos sobre un texto corto."""
    probe = "texto"
    return len(tokenizer.encode(probe, add_special_tokens=True)) - len(
        tokenizer.encode(probe, add_special_tokens=False)
    )


def _prefix_overhead(tokenizer: SplittableTokenizer, contract: ModelContract) -> int:
    """Tokens que aporta la plantilla documental alrededor del texto (p. ej. 'passage: ')."""
    return len(tokenizer.encode(contract.format_document(""), add_special_tokens=False))


def _token_windows(
    tokenizer: SplittableTokenizer, text: str, max_content_tokens: int, overlap: int
) -> list[tuple[str, int, int]]:
    """Divide `text` en ventanas de ≤`max_content_tokens` tokens de contenido, con `overlap`.

    Devuelve (segment_text, token_start, token_end) sobre los tokens **sin** special. No trunca:
    cubre todo el texto. `overlap` se acota para no estancarse en presupuestos pequeños.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return [(text, 0, 0)]
    if max_content_tokens <= 0:
        raise OverflowNotRepairedError(
            "presupuesto de tokens no positivo tras descontar special tokens y plantilla"
        )
    step_overlap = min(overlap, max_content_tokens - 1) if max_content_tokens > 1 else 0
    windows: list[tuple[str, int, int]] = []
    start = 0
    n = len(ids)
    while start < n:
        end = min(start + max_content_tokens, n)
        segment = tokenizer.decode(ids[start:end])
        windows.append((segment, start, end))
        if end >= n:
            break
        start = end - step_overlap
    return windows


# --------------------------------------------------------------------------- #
# Construcción de rows
# --------------------------------------------------------------------------- #


def _paragraph_anchor(chunk_text: str, parent: dict, *, chunk_id: str | None) -> dict | None:
    """Resuelve un rango de párrafos buscando una secuencia contigua exacta."""
    paragraphs = parent.get("paragraphs") or []
    if not paragraphs:
        return None
    chunk_lines = [line for line in chunk_text.split("\n") if line != ""]
    if not chunk_lines:
        return None
    parent_texts = [p.get("text", "") for p in paragraphs]
    matches: list[int] = []
    n = len(chunk_lines)
    for start in range(0, len(parent_texts) - n + 1):
        if parent_texts[start : start + n] == chunk_lines:
            matches.append(start)
    label = chunk_id or parent.get("parent_id", "<sin parent_id>")
    if not matches:
        raise AnchorResolutionError(f"no se pudo localizar el chunk como secuencia: {label}")
    if len(matches) > 1:
        raise AnchorResolutionError(f"secuencia de párrafos ambigua para chunk: {label}")
    start = matches[0]
    orders = [p.get("order") for p in paragraphs[start : start + n] if p.get("order") is not None]
    if not orders:
        raise AnchorResolutionError(f"secuencia sin orders válidos para chunk: {label}")
    return {"paragraph_start": min(orders), "paragraph_end": max(orders)}


def _paragraph_token_intervals(
    tokenizer: SplittableTokenizer, parent: dict
) -> list[tuple[int, int, int]]:
    """Intervalos [start, end) por párrafo medidos sobre prefijos exactos del parent."""
    intervals: list[tuple[int, int, int]] = []
    parent_text = parent.get("text", "") or ""
    char_cursor = 0
    for p in parent.get("paragraphs") or []:
        para_text = p.get("text", "") or ""
        char_start = parent_text.find(para_text, char_cursor)
        if char_start < 0:
            raise AnchorResolutionError(
                f"no se pudo localizar párrafo {p.get('order')} en {parent.get('parent_id')}"
            )
        char_end = char_start + len(para_text)
        start = len(tokenizer.encode(parent_text[:char_start], add_special_tokens=False))
        end = len(tokenizer.encode(parent_text[:char_end], add_special_tokens=False))
        intervals.append((p["order"], start, end))
        char_cursor = char_end
    return intervals


def _paragraphs_for_anchor(parent: dict, anchor: dict | None) -> list[dict]:
    if anchor is None:
        return []
    return [
        p
        for p in parent.get("paragraphs") or []
        if anchor["paragraph_start"] <= p.get("order", 0) <= anchor["paragraph_end"]
    ]


def _paragraph_token_intervals_in_text(
    tokenizer: SplittableTokenizer,
    text: str,
    paragraphs: list[dict],
    *,
    label: str,
) -> list[tuple[int, int, int]]:
    """Intervalos [start, end) por parrafo medidos sobre prefijos exactos de `text`."""
    intervals: list[tuple[int, int, int]] = []
    char_cursor = 0
    for p in paragraphs:
        para_text = p.get("text", "") or ""
        if not para_text:
            raise AnchorResolutionError(
                f"parrafo {p.get('order')} vacio al resolver anchor token-aware para {label}"
            )
        char_start = text.find(para_text, char_cursor)
        if char_start < 0:
            raise AnchorResolutionError(
                f"no se pudo localizar el parrafo {p.get('order')} dentro del texto base de "
                f"{label}; revisa retrieval_text/chunk.text o ajusta el chunk para resolver "
                "un context_anchor preciso."
            )
        char_end = char_start + len(para_text)
        start = len(tokenizer.encode(text[:char_start], add_special_tokens=False))
        end = len(tokenizer.encode(text[:char_end], add_special_tokens=False))
        intervals.append((p["order"], start, end))
        char_cursor = char_end
    return intervals


def _orders_for_token_window(
    token_start: int,
    token_end: int,
    intervals: list[tuple[int, int, int]] | None,
    *,
    label: str,
) -> list[int]:
    if not intervals:
        raise AnchorResolutionError(f"no hay intervalos de pÃ¡rrafo para {label}")
    return [
        order
        for order, para_start, para_end in intervals
        if token_start < para_end and token_end > para_start
    ]


def _anchor_for_token_window(
    token_start: int,
    token_end: int,
    intervals: list[tuple[int, int, int]] | None,
    *,
    label: str,
) -> dict:
    if not intervals:
        raise AnchorResolutionError(f"no hay intervalos de párrafo para {label}")
    orders = [
        order
        for order, para_start, para_end in intervals
        if token_start < para_end and token_end > para_start
    ]
    if not orders:
        raise AnchorResolutionError(f"ventana sin párrafos solapados para {label}")
    return {"paragraph_start": min(orders), "paragraph_end": max(orders)}


def _emit_for_base_text(
    *,
    base_text: str,
    document_id: str,
    block_id: str,
    parent_id: str,
    chunk_id: str | None,
    field_name: str | None,
    context_anchor: dict | None,
    derived_origin: str | None,
    tokenizer: SplittableTokenizer,
    contract: ModelContract,
    effective_max: int,
    content_budget: int,
    overlap: int,
    rows: list[dict],
    texts: list[str],
    counters: dict,
    paragraph_token_intervals: list[tuple[int, int, int]] | None = None,
    overflow_anchor_paragraphs: list[dict] | None = None,
) -> None:
    """Emite 1 row (cabe) o N rows derivadas (no cabe → reparación token-aware)."""
    formatted = contract.format_document(base_text)
    n_tokens = count_tokens(tokenizer, formatted)

    fits = n_tokens <= effective_max
    is_chunk_field = derived_origin is None

    if fits and is_chunk_field:
        _append_row(
            rows,
            texts,
            formatted=formatted,
            document_id=document_id,
            block_id=block_id,
            parent_id=parent_id,
            source={"kind": "chunk_field", "chunk_id": chunk_id, "field": field_name},
            context_anchor=context_anchor,
            token_count=n_tokens,
        )
        return

    # Caso derivado: C1 (siempre ventanas) u overflow repair de un chunk que no cabe.
    origin = derived_origin or "overflow_repair"
    if origin == "overflow_repair":
        counters["overflow_repaired_inputs"] += 1
        if context_anchor is None:
            raise AnchorResolutionError(
                f"overflow_repair sin context_anchor para {document_id}/{block_id}"
            )
        if paragraph_token_intervals is None:
            paragraph_token_intervals = _paragraph_token_intervals_in_text(
                tokenizer,
                base_text,
                overflow_anchor_paragraphs or [],
                label=f"{document_id}/{block_id}/{chunk_id or 'sin_chunk'}",
            )
    windows = _token_windows(tokenizer, base_text, content_budget, overlap)
    seg_count = len(windows)
    for seg_index, (segment_text, tok_start, tok_end) in enumerate(windows):
        label = f"{document_id}/{block_id}/seg{seg_index}"
        if origin == "fixed_token_window":
            row_anchor = _anchor_for_token_window(
                tok_start, tok_end, paragraph_token_intervals, label=label
            )
        else:
            orders = _orders_for_token_window(
                tok_start, tok_end, paragraph_token_intervals, label=label
            )
            if not orders:
                counters["auxiliary_context_windows_discarded"].append(
                    f"{label} {tok_start}:{tok_end}"
                )
                continue
            row_anchor = {"paragraph_start": min(orders), "paragraph_end": max(orders)}
        seg_formatted = contract.format_document(segment_text)
        seg_tokens = count_tokens(tokenizer, seg_formatted)
        if seg_tokens > effective_max:
            counters["unrepaired_overflow"].append(
                f"{document_id}/{block_id} seg{seg_index} {seg_tokens}>{effective_max}"
            )
        _append_row(
            rows,
            texts,
            formatted=seg_formatted,
            document_id=document_id,
            block_id=block_id,
            parent_id=parent_id,
            source={
                "kind": "derived_text",
                "origin": origin,
                "chunk_id": chunk_id,
                "text": segment_text,
                "token_start": tok_start,
                "token_end": tok_end,
                "segment_index": seg_index,
                "segment_count": seg_count,
            },
            context_anchor=row_anchor,
            token_count=seg_tokens,
        )


def _append_row(
    rows: list[dict],
    texts: list[str],
    *,
    formatted: str,
    document_id: str,
    block_id: str,
    parent_id: str,
    source: dict,
    context_anchor: dict | None,
    token_count: int,
) -> None:
    row_index = len(rows)
    rows.append(
        {
            "row_index": row_index,
            "embedding_input_id": f"ein_{row_index:06d}",
            "document_id": document_id,
            "block_id": block_id,
            "parent_id": parent_id,
            "source": source,
            "context_anchor": context_anchor,
            "token_count": token_count,
            "formatted_input_sha256": _sha256(formatted),
        }
    )
    texts.append(formatted)


# --------------------------------------------------------------------------- #
# API pública
# --------------------------------------------------------------------------- #


def prepare_inputs(
    view: str,
    *,
    chunks: list[dict],
    parents_by_id: dict[str, dict],
    contract: ModelContract,
    tokenizer: SplittableTokenizer,
    overlap: int = OVERLAP_TOKENS,
    safety_margin: int = SAFETY_MARGIN_TOKENS,
) -> PreparedInputs:
    """Prepara los inputs de embedding de una vista. No trunca; repara overflow token-aware.

    `chunks`: chunks v2 de todo el corpus. `parents_by_id`: parent_id → ParentRecord (para anchors
    en J1/J2 y como fuente de texto en C1). Lanza `OverflowNotRepairedError` si tras reparar queda
    algún input por encima del límite efectivo.
    """
    if view not in VIEWS:
        raise ValueError(f"vista desconocida: {view!r} (esperado {VIEWS})")

    effective_max, source_of_limit = resolve_effective_max(
        contract.declared_max_tokens, getattr(tokenizer, "model_max_length", None)
    )
    special = _special_overhead(tokenizer)
    prefix = _prefix_overhead(tokenizer, contract)
    content_budget = effective_max - special - prefix - safety_margin

    rows: list[dict] = []
    texts: list[str] = []
    counters: dict = {
        "overflow_repaired_inputs": 0,
        "unrepaired_overflow": [],
        "auxiliary_context_windows_discarded": [],
    }

    if view in ("J1", "J2"):
        field_name = _VIEW_FIELD[view]
        for ch in chunks:
            parent = parents_by_id.get(ch.get("parent_id"), {})
            anchor = _paragraph_anchor(
                ch.get("text", "") or "", parent, chunk_id=ch.get("chunk_id")
            )
            _emit_for_base_text(
                base_text=ch.get(field_name, "") or "",
                document_id=ch.get("document_id"),
                block_id=ch.get("block_id"),
                parent_id=ch.get("parent_id"),
                chunk_id=ch.get("chunk_id"),
                field_name=field_name,
                context_anchor=anchor,
                derived_origin=None,
                paragraph_token_intervals=None,
                overflow_anchor_paragraphs=_paragraphs_for_anchor(parent, anchor),
                tokenizer=tokenizer,
                contract=contract,
                effective_max=effective_max,
                content_budget=content_budget,
                overlap=overlap,
                rows=rows,
                texts=texts,
                counters=counters,
            )
    else:  # C1: ventanas token-aware del parent, solo parents indexables (los que tienen chunks).
        indexable_parents = {ch.get("parent_id") for ch in chunks}
        for ch_parent_id in sorted(indexable_parents):
            parent = parents_by_id.get(ch_parent_id)
            if parent is None:
                continue
            intervals = _paragraph_token_intervals(tokenizer, parent)
            _emit_for_base_text(
                base_text=parent.get("text", "") or "",
                document_id=parent.get("document_id"),
                block_id=parent.get("block_id"),
                parent_id=parent.get("parent_id"),
                chunk_id=None,
                field_name=None,
                context_anchor=None,
                derived_origin="fixed_token_window",
                paragraph_token_intervals=intervals,
                tokenizer=tokenizer,
                contract=contract,
                effective_max=effective_max,
                content_budget=content_budget,
                overlap=overlap,
                rows=rows,
                texts=texts,
                counters=counters,
            )

    if counters["unrepaired_overflow"]:
        raise OverflowNotRepairedError(
            "overflow sin reparar tras la división token-aware: "
            + "; ".join(counters["unrepaired_overflow"][:5])
        )

    n_derived = sum(1 for r in rows if r["source"]["kind"] == "derived_text")
    report = {
        "view": view,
        "model_alias": contract.alias,
        "n_source_chunks": len(chunks),
        "n_rows": len(rows),
        "n_derived_rows": n_derived,
        "n_overflow_repaired_inputs": counters["overflow_repaired_inputs"],
        "n_auxiliary_context_windows_discarded": len(
            counters["auxiliary_context_windows_discarded"]
        ),
        "auxiliary_context_windows_discarded_sample": counters[
            "auxiliary_context_windows_discarded"
        ][:5],
        "n_truncated": 0,
        "effective_max_tokens": effective_max,
        "source_of_effective_limit": source_of_limit,
        "special_token_overhead": special,
        "template_prefix_overhead": prefix,
        "content_budget_tokens": content_budget,
        "overlap_tokens": overlap,
        "max_token_count": max((r["token_count"] for r in rows), default=0),
    }
    return PreparedInputs(
        view=view,
        model_alias=contract.alias,
        rows=rows,
        texts=texts,
        effective_max_tokens=effective_max,
        report=report,
    )
