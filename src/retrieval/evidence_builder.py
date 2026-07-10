"""Construcción de evidencias para la generación a partir de hits de retrieval denso.

Convierte los `DenseHit` ordenados por relevancia en `GenerationEvidence` deterministas y
acotadas, listas para inyectar en el prompt:

1. deduplica por `parent_id` conservando el mejor hit (rank más alto = número de rank menor);
2. limita el número de evidencias candidatas (`max_evidences`);
3. resuelve el parent en `corpus["parents_by_id"]` y ensambla el contexto con `assemble_context`
   (estrategia configurable; por defecto `P_EXPAND_BOUNDED`);
4. aplica un presupuesto por evidencia (al ensamblar) y un presupuesto AGREGADO de caracteres;
5. NO trunca texto jurídico de forma silenciosa: si una evidencia no cabe en el presupuesto
   agregado se OMITE entera y se registra el diagnóstico, priorizando las mejor rankeadas;
6. asigna IDs compactos deterministas (`E1`, `E2`, ...) solo a las evidencias finalmente incluidas.

No conoce el LLM ni el prompt: solo prepara datos autoritativos del corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.core.exceptions import ConfigurationError
from src.retrieval.context_assembler import P_EXPAND_BOUNDED, STRATEGIES, assemble_context
from src.retrieval.dense_retriever import DenseHit

# Defaults de bajo nivel; la configuración de ejecución puede sobrescribirlos.
DEFAULT_MAX_EVIDENCES = 3
DEFAULT_CONTEXT_STRATEGY = P_EXPAND_BOUNDED
DEFAULT_CONTEXT_BUDGET_CHARS = 4000  # B4K por evidencia
DEFAULT_MAX_TOTAL_CONTEXT_CHARS = 16000  # presupuesto agregado


@dataclass
class GenerationEvidence:
    """Una evidencia jurídica resuelta y acotada, identificada por un ID compacto (`E1`, ...)."""

    evidence_id: str
    parent_id: str
    document_id: str
    block_id: str
    label: str
    url: str | None
    score: float
    retrieval_rank: int
    context_strategy: str
    text: str
    paragraph_orders: list[int] = field(default_factory=list)
    char_count: int = 0


@dataclass
class EvidenceSelection:
    """Resultado del builder: evidencias incluidas + diagnóstico compacto de las omitidas."""

    evidences: list[GenerationEvidence] = field(default_factory=list)
    omitted: list[dict] = field(default_factory=list)
    duplicate_parents_removed: int = 0
    total_char_count: int = 0


def _dedup_by_parent(hits: list[DenseHit]) -> tuple[list[DenseHit], int]:
    """Conserva el mejor hit por `parent_id` (menor rank) preservando el orden de aparición."""
    best: dict[str, DenseHit] = {}
    order: list[str] = []
    duplicates = 0
    for hit in hits:
        prev = best.get(hit.parent_id)
        if prev is None:
            best[hit.parent_id] = hit
            order.append(hit.parent_id)
        else:
            duplicates += 1
            if hit.rank < prev.rank:
                best[hit.parent_id] = hit
    deduped = sorted((best[pid] for pid in order), key=lambda h: h.rank)
    return deduped, duplicates


def build_evidences(
    hits: list[DenseHit],
    *,
    parents_by_id: dict[str, dict],
    max_evidences: int = DEFAULT_MAX_EVIDENCES,
    context_strategy: str = DEFAULT_CONTEXT_STRATEGY,
    context_budget_chars: int = DEFAULT_CONTEXT_BUDGET_CHARS,
    max_total_context_chars: int = DEFAULT_MAX_TOTAL_CONTEXT_CHARS,
) -> EvidenceSelection:
    """Selecciona y ensambla evidencias acotadas a partir de los hits (ver módulo).

    Recorre TODOS los hits deduplicados por orden de ranking (no recorta a `max_evidences` de
    entrada): si un candidato mejor rankeado se descarta (parent ausente, contexto vacío, sobre
    presupuesto por evidencia o agregado), continúa evaluando candidatos posteriores (*backfill*)
    hasta reunir `max_evidences` válidas o agotar los hits. Nunca trunca texto jurídico.
    """
    if max_evidences <= 0:
        raise ConfigurationError(f"max_evidences debe ser > 0 (recibido {max_evidences}).")
    if context_budget_chars <= 0:
        raise ConfigurationError(
            f"context_budget_chars debe ser > 0 (recibido {context_budget_chars})."
        )
    if max_total_context_chars <= 0:
        raise ConfigurationError(
            f"max_total_context_chars debe ser > 0 (recibido {max_total_context_chars})."
        )
    if context_strategy not in STRATEGIES:
        raise ConfigurationError(
            f"context_strategy inválida: {context_strategy!r} (esperado uno de {STRATEGIES})."
        )

    deduped, duplicates = _dedup_by_parent(hits)

    selection = EvidenceSelection(duplicate_parents_removed=duplicates)
    running_total = 0
    next_index = 1
    limit = max_evidences
    for hit in deduped:
        if len(selection.evidences) >= limit:
            break

        parent = parents_by_id.get(hit.parent_id)
        if parent is None:
            _omit(selection, hit, "parent_not_found")
            continue

        try:
            ctx = assemble_context(
                strategy=context_strategy,
                parent=parent,
                anchor=hit.context_anchor,
                retrieved_text=hit.retrieval_text,
                budget_chars=context_budget_chars,
            )
        except ValueError:
            # No se pudo ensamblar un contexto válido (p. ej. anchor irresoluble).
            _omit(selection, hit, "empty_context")
            continue

        if not ctx.text.strip():
            _omit(selection, hit, "empty_context", char_count=ctx.char_count)
            continue
        if ctx.over_budget:
            # La evidencia individual no cabe en su presupuesto; se omite entera (no se trunca).
            _omit(selection, hit, "exceeds_evidence_context_budget", char_count=ctx.char_count)
            continue
        if running_total + ctx.char_count > max_total_context_chars:
            # No cabe en el presupuesto agregado; se omite y se sigue buscando candidatos menores.
            _omit(selection, hit, "exceeds_total_context_budget", char_count=ctx.char_count)
            continue

        running_total += ctx.char_count
        selection.evidences.append(
            GenerationEvidence(
                evidence_id=f"E{next_index}",
                parent_id=hit.parent_id,
                document_id=hit.document_id,
                block_id=hit.block_id,
                label=hit.citation_label,
                url=hit.citation_url,
                score=hit.score,
                retrieval_rank=hit.rank,
                context_strategy=ctx.strategy,
                text=ctx.text,
                paragraph_orders=list(ctx.paragraph_orders),
                char_count=ctx.char_count,
            )
        )
        next_index += 1

    selection.total_char_count = running_total
    return selection


def build_oracle_evidences(
    gold_parents: list[dict],
    *,
    parents_by_id: dict[str, dict],
    max_evidences: int = DEFAULT_MAX_EVIDENCES,
    context_strategy: str = DEFAULT_CONTEXT_STRATEGY,
    context_budget_chars: int = DEFAULT_CONTEXT_BUDGET_CHARS,
    max_total_context_chars: int = DEFAULT_MAX_TOTAL_CONTEXT_CHARS,
) -> EvidenceSelection:
    """Evidencia-*oracle*: inyecta los parents GOLD como si un recuperador perfecto los trajera.

    `gold_parents` es una lista de dicts `{"parent_id", "paragraph_orders", "relevance"}`
    (típicamente los juicios con relevancia >= 1, ordenados por relevancia desc). Fabrica
    `DenseHit` sintéticos anclados en los párrafos de evidencia del gold y los pasa por
    `build_evidences` con la MISMA configuración de producción: el contexto se ensambla igual que
    en la ruta real. Así el *oracle* mide el techo del generador con recuperación perfecta, sin
    cambiar el ensamblado. Los `parent_id` sin parent en el corpus se ignoran.
    """
    hits: list[DenseHit] = []
    for rank, gold in enumerate(gold_parents, start=1):
        parent_id = gold["parent_id"]
        parent = parents_by_id.get(parent_id)
        if parent is None:
            continue
        orders = sorted(o for o in (gold.get("paragraph_orders") or []))
        if not orders:
            # Sin párrafos de evidencia: ancla al primer párrafo existente del parent.
            existing = sorted(p["order"] for p in (parent.get("paragraphs") or []) if "order" in p)
            orders = existing[:1]
        anchor = {"paragraph_start": orders[0], "paragraph_end": orders[-1]} if orders else None
        citation = parent.get("citation") or {}
        hits.append(
            DenseHit(
                rank=rank,
                score=float(gold.get("relevance", 1) or 1),
                row_index=rank - 1,
                embedding_input_id=f"oracle_{rank:06d}",
                document_id=parent.get("document_id", ""),
                block_id=parent.get("block_id", ""),
                parent_id=parent_id,
                source={"kind": "oracle", "parent_id": parent_id},
                context_anchor=anchor,
                retrieval_text=parent.get("text", ""),
                citation_label=citation.get("label") or parent_id,
                citation_url=citation.get("url"),
            )
        )
    return build_evidences(
        hits,
        parents_by_id=parents_by_id,
        max_evidences=max_evidences,
        context_strategy=context_strategy,
        context_budget_chars=context_budget_chars,
        max_total_context_chars=max_total_context_chars,
    )


def _omit(
    selection: EvidenceSelection, hit: DenseHit, reason: str, *, char_count: int | None = None
) -> None:
    """Registra una omisión diagnosticada (sin texto jurídico)."""
    entry: dict = {"parent_id": hit.parent_id, "retrieval_rank": hit.rank, "reason": reason}
    if char_count is not None:
        entry["char_count"] = char_count
    selection.omitted.append(entry)
