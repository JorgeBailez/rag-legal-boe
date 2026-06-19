"""Recuperador híbrido: fusiona un recuperador denso y uno léxico en una sola lista ordenada.

Combina lo mejor de ambos mundos (significado del denso + vocabulario exacto del BM25). Dos métodos:

- **RRF** (Reciprocal Rank Fusion, por defecto): `score(d) = Σ_r 1/(k + rank_r(d))`. **Sin
  hiperparámetros sensibles** (solo `k`, robusto ≈60) y no necesita normalizar escalas heterogéneas
  (coseno vs BM25). Es el método estándar y la opción recomendada para la comparación.
- **`weighted`**: combina los scores **normalizados** (min-max por recuperador) como
  `α·denso + (1-α)·léxico`. Útil para estudiar la sensibilidad al peso relativo.

La fusión se hace por `embedding_input_id`, que identifica la misma unidad de recuperación en ambos
índices (se construyen sobre las mismas rows del bundle). Mantiene la interfaz del `DenseRetriever`
(`retrieve` → `list[DenseHit]`) para ser intercambiable en el runner de comparación.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal, Protocol

from src.embeddings.corpus_loader import load_processed_corpus
from src.retrieval.dense_retriever import DenseHit, DenseRetriever
from src.retrieval.lexical_retriever import LexicalRetriever
from src.retrieval.text_analysis import SpanishAnalyzer

FusionMethod = Literal["rrf", "weighted"]


class Retriever(Protocol):
    """Interfaz común de los recuperadores (la cumplen el denso, el léxico y los fakes)."""

    def retrieve(
        self,
        query: str,
        *,
        query_profile_id: str | None = ...,
        top_k: int = ...,
        filters: dict | None = ...,
    ) -> list[DenseHit]: ...

    def resolved_query_profile_id(self, query_profile_id: str | None = ...) -> str: ...


def _min_max(scores: dict[str, float]) -> dict[str, float]:
    """Normaliza scores a [0, 1] por min-max; si todos son iguales, 1.0 (o 0.0 si son ≤ 0)."""
    if not scores:
        return {}
    values = scores.values()
    lo, hi = min(values), max(values)
    if hi <= lo:
        fill = 1.0 if hi > 0 else 0.0
        return dict.fromkeys(scores, fill)
    span = hi - lo
    return {key: (value - lo) / span for key, value in scores.items()}


class HybridRetriever:
    """Fusiona denso + léxico (RRF o ponderada). Drop-in del `DenseRetriever` en el runner."""

    def __init__(
        self,
        *,
        dense: Retriever,
        lexical: Retriever,
        fusion: FusionMethod = "rrf",
        rrf_k: int = 60,
        alpha: float = 0.5,
        candidates: int = 100,
    ) -> None:
        if fusion not in ("rrf", "weighted"):
            raise ValueError(f"fusion debe ser 'rrf' o 'weighted' (recibido {fusion!r}).")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha debe estar en [0, 1] (recibido {alpha}).")
        self.dense = dense
        self.lexical = lexical
        self.fusion = fusion
        self.rrf_k = rrf_k
        self.alpha = alpha
        self.candidates = candidates
        self.bundle_id = getattr(dense, "bundle_id", "")
        self.model_alias = f"hybrid_{fusion}"

    @classmethod
    def from_bundle(
        cls,
        bundle_dir: str | Path,
        *,
        corpus: dict | None = None,
        encoder: object | None = None,
        analyzer: SpanishAnalyzer | None = None,
        fusion: FusionMethod = "rrf",
        rrf_k: int = 60,
        alpha: float = 0.5,
        candidates: int = 100,
    ) -> HybridRetriever:
        """Construye denso + léxico sobre el MISMO bundle/corpus (mismas rows) y los fusiona."""
        corpus = corpus if corpus is not None else load_processed_corpus()
        dense = DenseRetriever.from_bundle(bundle_dir, corpus=corpus, encoder=encoder)
        lexical = LexicalRetriever.from_bundle(bundle_dir, corpus=corpus, analyzer=analyzer)
        return cls(
            dense=dense,
            lexical=lexical,
            fusion=fusion,
            rrf_k=rrf_k,
            alpha=alpha,
            candidates=candidates,
        )

    def resolved_query_profile_id(self, query_profile_id: str | None = None) -> str:
        return self.dense.resolved_query_profile_id(query_profile_id)

    def retrieve(
        self,
        query: str,
        *,
        query_profile_id: str | None = None,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[DenseHit]:
        """Recupera candidatos de ambos recuperadores, los fusiona y devuelve el top-k final."""
        if top_k <= 0:
            raise ValueError(f"top_k debe ser > 0 (recibido {top_k}).")
        pool = max(self.candidates, top_k)
        dense_hits = self.dense.retrieve(
            query, query_profile_id=query_profile_id, top_k=pool, filters=filters
        )
        lexical_hits = self.lexical.retrieve(query, top_k=pool, filters=filters)
        fused = self._fuse(dense_hits, lexical_hits)
        return fused[:top_k]

    def _fuse(self, dense_hits: list[DenseHit], lexical_hits: list[DenseHit]) -> list[DenseHit]:
        if self.fusion == "rrf":
            return self._fuse_rrf(dense_hits, lexical_hits)
        return self._fuse_weighted(dense_hits, lexical_hits)

    def _representatives(
        self, dense_hits: list[DenseHit], lexical_hits: list[DenseHit]
    ) -> dict[str, DenseHit]:
        """Un `DenseHit` por unidad (mismo metadato en ambos índices); se prioriza el del denso."""
        rep: dict[str, DenseHit] = {}
        for hit in (*lexical_hits, *dense_hits):  # el denso pisa al léxico → gana su representante
            rep[hit.embedding_input_id] = hit
        return rep

    def _ranked(self, fused: dict[str, float], rep: dict[str, DenseHit]) -> list[DenseHit]:
        """Ordena por score fusionado (desc) con desempate estable por id, y reasigna rank/score."""
        order = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
        return [
            replace(rep[key], rank=rank, score=score)
            for rank, (key, score) in enumerate(order, start=1)
        ]

    def _fuse_rrf(self, dense_hits: list[DenseHit], lexical_hits: list[DenseHit]) -> list[DenseHit]:
        scores: dict[str, float] = {}
        for hits in (dense_hits, lexical_hits):
            for hit in hits:
                key = hit.embedding_input_id
                scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + hit.rank)
        return self._ranked(scores, self._representatives(dense_hits, lexical_hits))

    def _fuse_weighted(
        self, dense_hits: list[DenseHit], lexical_hits: list[DenseHit]
    ) -> list[DenseHit]:
        dense_norm = _min_max({h.embedding_input_id: h.score for h in dense_hits})
        lexical_norm = _min_max({h.embedding_input_id: h.score for h in lexical_hits})
        scores = {
            key: self.alpha * dense_norm.get(key, 0.0)
            + (1.0 - self.alpha) * lexical_norm.get(key, 0.0)
            for key in dense_norm.keys() | lexical_norm.keys()
        }
        return self._ranked(scores, self._representatives(dense_hits, lexical_hits))
