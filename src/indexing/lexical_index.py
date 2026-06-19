"""Índice léxico BM25 sobre las MISMAS unidades de recuperación que el índice denso.

El scoring se delega en `rank_bm25` (BM25Okapi, vectorizado con numpy). Lo propio del proyecto:

1. **Indexa exactamente las mismas `rows` que el bundle denso** (misma `embedding_input_id` y mismo
   `row_index`): la comparación denso vs léxico y la fusión RRF son así manzana-con-manzana.
2. Resuelve el texto de cada row con la **misma función pura** que el retriever denso
   (`resolve_hit_text_and_citation`) → se indexa el `retrieval_text` real, sin el prefijo de
   instrucción que solo necesita el modelo de embeddings.
3. Usa el `SpanishAnalyzer` para tokenizar/stemming.

`search` devuelve hits con el **mismo esquema** que `ExactDenseIndex.search`, para que el retriever
léxico reutilice la construcción de hits. No se persiste binario (nada de pickle): reconstruir el
índice desde el corpus es barato frente al coste de los embeddings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from src.embeddings.bundle import load_validated_bundle
from src.retrieval.dense_retriever import resolve_hit_text_and_citation
from src.retrieval.text_analysis import SpanishAnalyzer

_NEG_INF = float("-inf")


def row_texts(rows: list[dict], corpus: dict) -> list[str]:
    """Texto recuperado (K_ONLY) de cada row, por join ligero al corpus (orden = orden de rows)."""
    chunks_by_id = {c["chunk_id"]: c for c in corpus.get("chunks", [])}
    parents_by_id = corpus.get("parents_by_id", {})
    return [
        resolve_hit_text_and_citation(row, chunks_by_id=chunks_by_id, parents_by_id=parents_by_id)[
            0
        ]
        for row in rows
    ]


class LexicalIndex:
    """Índice BM25 sobre las rows de un bundle. `search` espeja `ExactDenseIndex.search`."""

    def __init__(
        self,
        *,
        rows: list[dict],
        texts: list[str],
        manifest: dict | None = None,
        analyzer: SpanishAnalyzer | None = None,
    ) -> None:
        if len(rows) != len(texts):
            raise ValueError(
                f"rows ({len(rows)}) y texts ({len(texts)}) deben tener igual longitud."
            )
        self.rows = rows
        self.manifest = manifest or {}
        self.analyzer = analyzer or SpanishAnalyzer()
        tokenized = [self.analyzer.analyze(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized)
        # Conjunto de tokens por doc: el "match" se decide por SOLAPE léxico real, no por el signo
        # del score (el IDF de Okapi puede ser 0 o negativo y dejar a 0 un solape legítimo cuando un
        # término aparece en ~la mitad de los docs o el corpus es pequeño).
        self._doc_tokens: list[set[str]] = [set(toks) for toks in tokenized]

    @classmethod
    def from_bundle(
        cls, bundle_dir: str | Path, *, corpus: dict, analyzer: SpanishAnalyzer | None = None
    ) -> LexicalIndex:
        """Construye el índice léxico sobre las rows del bundle denso (mismo orden/ids)."""
        manifest, rows, _embeddings = load_validated_bundle(Path(bundle_dir), corpus=corpus)
        return cls(rows=rows, texts=row_texts(rows, corpus), manifest=manifest, analyzer=analyzer)

    def __len__(self) -> int:
        return len(self.rows)

    def search(self, query: str, *, k: int = 5, mask: np.ndarray | None = None) -> list[dict]:
        """Top-k por BM25 entre los docs con SOLAPE léxico real con la query (≥1 token en común).

        `mask` (opcional) restringe las filas candidatas, igual que el índice denso; los hits llevan
        el mismo esquema que `ExactDenseIndex.search`. Un doc sin ningún token de la query NO es
        match aunque el ranking lo empate a 0, de modo que BM25 devuelve solo coincidencias reales
        (y puede devolver menos de k si hay pocas).
        """
        if k <= 0:
            raise ValueError(f"k debe ser > 0 (recibido {k}).")
        tokens = self.analyzer.analyze(query)
        if not tokens:
            return []
        query_set = set(tokens)
        scores = np.asarray(self._bm25.get_scores(tokens), dtype=np.float32)
        if mask is not None:
            scores = np.where(mask, scores, _NEG_INF)
        hits: list[dict] = []
        for idx in np.argsort(-scores, kind="stable"):
            if len(hits) >= k:
                break
            if scores[idx] == _NEG_INF:
                break  # zona enmascarada (el resto del orden también lo está)
            if not (query_set & self._doc_tokens[idx]):
                continue  # sin solape léxico real → no es un match
            row = self.rows[idx]
            hits.append(
                {
                    "rank": len(hits) + 1,
                    "score": float(scores[idx]),
                    "row_index": int(idx),
                    "embedding_input_id": row["embedding_input_id"],
                    "document_id": row["document_id"],
                    "block_id": row["block_id"],
                    "parent_id": row["parent_id"],
                    "source": row["source"],
                    "context_anchor": row.get("context_anchor"),
                }
            )
        return hits
