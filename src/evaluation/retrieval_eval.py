"""Comparación reproducible de estrategias de retrieval (denso vs BM25 vs híbrido) a nivel L1.

Orquestación **pura y testeable offline**: recibe un conjunto de recuperadores ya construidos (los
fakes en los tests cumplen la misma interfaz `retrieve`) y, sobre un split del banco, calcula las
mismas métricas de retrieval que el benchmark denso (`compute_query_retrieval_metrics`), su IC por
bootstrap y la **diferencia pareada** frente a una baseline (por defecto, el denso).
La construcción de índices/encoder y la escritura del report viven en el CLI; aquí solo se enlazan
datos y métricas, de modo que la comparación se valide con fakes sin pesos ni bundle reales.

Todas las estrategias se evalúan sobre las MISMAS preguntas y juicios, recuperando la misma
profundidad (`retrieve_depth`), para que la comparación sea manzana-con-manzana.
"""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import Protocol

import numpy as np

from src.evaluation.metrics import (
    PRIMARY_METRIC,
    aggregate_metric_groups,
    aggregate_metrics,
    bootstrap_ci,
    compute_query_retrieval_metrics,
    paired_bootstrap,
)

DEFAULT_RETRIEVE_DEPTH = 50


class Retriever(Protocol):
    """Interfaz mínima que consume el bucle (la cumplen denso, léxico, híbrido y los fakes)."""

    def retrieve(
        self,
        query: str,
        *,
        query_profile_id: str | None = ...,
        top_k: int = ...,
        filters: dict | None = ...,
    ) -> list: ...


def _hit_to_dict(hit: object) -> dict:
    """`DenseHit` → dict que consumen las métricas (parent_id + context_anchor + traza)."""
    return {
        "rank": hit.rank,
        "score": hit.score,
        "parent_id": hit.parent_id,
        "embedding_input_id": hit.embedding_input_id,
        "context_anchor": hit.context_anchor,
        "source": hit.source,
    }


def _brief_hit(hit: dict) -> dict:
    """Versión compacta del hit para el detalle por query del report."""
    return {"rank": hit["rank"], "parent_id": hit["parent_id"], "score": round(hit["score"], 4)}


def _percentile_ms(values: list[float], p: float) -> float:
    return float(np.percentile(values, p)) if values else 0.0


def _group_by(field: str, questions: list[dict], per_query: list[dict]) -> dict[str, list[dict]]:
    """Agrupa las métricas por query según un campo de la pregunta (p. ej. `query_style`).

    `per_query` va alineado con `questions` (mismo orden de evaluación). Las preguntas sin el campo
    caen en `(sin)` para no perderlas del recuento.
    """
    groups: dict[str, list[dict]] = {}
    for q, m in zip(questions, per_query, strict=True):
        key = str(q.get(field) or "(sin)")
        groups.setdefault(key, []).append(m)
    return groups


def evaluate_retrieval_strategies(
    *,
    strategies: dict[str, Retriever],
    split_questions: list[dict],
    judgments_by_query: dict[str, list[dict]],
    retrieve_depth: int = DEFAULT_RETRIEVE_DEPTH,
    query_profile_id: str | None = None,
    baseline: str = "dense",
    seed: int = 12345,
    on_progress: Callable[[dict], None] | None = None,
) -> dict:
    """Evalúa cada estrategia sobre el split y devuelve filas de métricas, detalle y resumen.

    `strategies` es un dict ordenado `{nombre: recuperador}`. La baseline para las diferencias
    pareadas es `baseline` si está presente, si no la primera estrategia. Devuelve
    `{metrics_rows, query_results, summary}`; `summary` incluye el IC por estrategia y el bootstrap
    pareado de cada estrategia frente a la baseline sobre la métrica primaria (ParentnDCG@10).
    `summary["stratified"]` desglosa cada estrategia por `query_style` y `difficulty` (n + medias +
    IC por estrato) para ver DÓNDE gana o pierde cada una (p. ej. el léxico en `directa_articulo`).
    """
    notify = on_progress or (lambda _info: None)
    total = len(strategies) * len(split_questions)
    done = 0

    metrics_rows: list[dict] = []
    query_results: list[dict] = []
    per_query: dict[str, list[dict]] = {}
    primary_by_strategy: dict[str, list[float]] = {}

    for name, retriever in strategies.items():
        if split_questions:  # warmup: descarta el coste de carga perezosa de la 1ª query
            retriever.retrieve(
                split_questions[0]["query"], query_profile_id=query_profile_id, top_k=1
            )
        pq: list[dict] = []
        latencies_ms: list[float] = []
        for q in split_questions:
            t0 = perf_counter()
            hits = retriever.retrieve(
                q["query"], query_profile_id=query_profile_id, top_k=retrieve_depth
            )
            latencies_ms.append((perf_counter() - t0) * 1000.0)
            hit_dicts = [_hit_to_dict(h) for h in hits]
            judgments_q = judgments_by_query.get(q["query_id"], [])
            m = compute_query_retrieval_metrics(hit_dicts, judgments_q)
            pq.append(m)
            query_results.append(
                {
                    "strategy": name,
                    "query_id": q["query_id"],
                    "split": q.get("split"),
                    "hits": [_brief_hit(h) for h in hit_dicts],
                    "metrics": m,
                }
            )
            done += 1
            notify(
                {
                    "event": "query",
                    "strategy": name,
                    "query_id": q["query_id"],
                    "done": done,
                    "total": total,
                }
            )

        per_query[name] = pq
        primary_by_strategy[name] = [d[PRIMARY_METRIC] for d in pq if PRIMARY_METRIC in d]
        metrics_rows.append(
            {
                "strategy": name,
                "model_alias": getattr(retriever, "model_alias", name),
                "n_queries": len(pq),
                "retrieve_latency_p50_ms": round(_percentile_ms(latencies_ms, 50), 3),
                "retrieve_latency_p95_ms": round(_percentile_ms(latencies_ms, 95), 3),
                **aggregate_metrics(pq),
            }
        )

    base = baseline if baseline in primary_by_strategy else next(iter(primary_by_strategy), None)
    paired_vs_baseline = [
        {
            "strategy": name,
            "baseline": base,
            "metric": PRIMARY_METRIC,
            "diff": paired_bootstrap(
                primary_by_strategy[name], primary_by_strategy[base], seed=seed
            ),
        }
        for name in strategies
        if base is not None and name != base
    ]
    stratified = {
        field_key: {
            name: aggregate_metric_groups(
                _group_by(field, split_questions, per_query[name]), seed=seed
            )
            for name in strategies
        }
        for field, field_key in (("query_style", "by_query_style"), ("difficulty", "by_difficulty"))
    }
    summary = {
        "primary_metric": PRIMARY_METRIC,
        "baseline": base,
        "retrieve_depth": retrieve_depth,
        "query_profile_id": query_profile_id,
        "strategies": [
            {
                "strategy": name,
                "n_queries": len(per_query[name]),
                "primary_ci": bootstrap_ci(primary_by_strategy[name], seed=seed),
            }
            for name in strategies
        ],
        "paired_vs_baseline": paired_vs_baseline,
        "stratified": stratified,
    }
    return {"metrics_rows": metrics_rows, "query_results": query_results, "summary": summary}
