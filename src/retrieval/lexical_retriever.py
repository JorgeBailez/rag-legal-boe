"""Recuperador léxico (BM25): *drop-in* del `DenseRetriever` con índice léxico en vez de denso.

Misma interfaz pública (`retrieve` → `list[DenseHit]`, `resolved_query_profile_id`, `bundle_id`,
`model_alias`) y mismo tipo de hit, de modo que el runner de comparación trate denso, léxico e
híbrido de forma intercambiable. No necesita encoder ni perfil de consulta: la query se tokeniza con
el mismo `SpanishAnalyzer` del índice. Reutiliza `build_filter_mask` (mismos filtros que el denso) y
`hit_from_raw` (misma resolución de texto + cita autoritativa del corpus).
"""

from __future__ import annotations

from pathlib import Path

from src.embeddings.corpus_loader import load_processed_corpus
from src.indexing.lexical_index import LexicalIndex
from src.indexing.vector_index import build_filter_mask
from src.retrieval.dense_retriever import DenseHit, hit_from_raw
from src.retrieval.text_analysis import SpanishAnalyzer


class LexicalRetriever:
    """Recuperador BM25 sobre un bundle, con joins al corpus precalculados (como el denso)."""

    def __init__(self, *, index: LexicalIndex, corpus: dict) -> None:
        self.index = index
        self.corpus = corpus
        self.manifest = index.manifest
        self.bundle_id = index.manifest.get("bundle", {}).get("bundle_id", "")
        self.model_alias = "bm25"
        self._chunks_by_id: dict[str, dict] = {c["chunk_id"]: c for c in corpus.get("chunks", [])}
        self._parents_by_id: dict[str, dict] = corpus.get("parents_by_id", {})

    @classmethod
    def from_bundle(
        cls,
        bundle_dir: str | Path,
        *,
        corpus: dict | None = None,
        analyzer: SpanishAnalyzer | None = None,
        heading_boost: int = 0,
    ) -> LexicalRetriever:
        """Carga corpus + rows del bundle y construye el índice BM25 + el retriever."""
        corpus = corpus if corpus is not None else load_processed_corpus()
        index = LexicalIndex.from_bundle(
            Path(bundle_dir), corpus=corpus, analyzer=analyzer, heading_boost=heading_boost
        )
        return cls(index=index, corpus=corpus)

    def resolved_query_profile_id(self, query_profile_id: str | None = None) -> str:
        """El léxico no usa perfil de consulta; devuelve una etiqueta fija para las trazas."""
        return "lexical"

    def retrieve(
        self,
        query: str,
        *,
        query_profile_id: str | None = None,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[DenseHit]:
        """Recupera top-k por BM25 y resuelve texto + cita de cada hit (mismo `DenseHit`).

        `query_profile_id` se acepta por compatibilidad de interfaz pero se ignora (no hay encoder).
        """
        if top_k <= 0:
            raise ValueError(f"top_k debe ser > 0 (recibido {top_k}).")
        if not query or not query.strip():
            raise ValueError("la query no puede estar vacía ni contener solo espacios.")
        mask = build_filter_mask(self.index.rows, self.corpus, filters) if filters else None
        raw_hits = self.index.search(query, k=top_k, mask=mask)
        return [
            hit_from_raw(h, chunks_by_id=self._chunks_by_id, parents_by_id=self._parents_by_id)
            for h in raw_hits
        ]
