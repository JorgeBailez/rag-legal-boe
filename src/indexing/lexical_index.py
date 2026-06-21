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


def row_boost_text(rows: list[dict], corpus: dict) -> list[str]:
    """Texto a boostear por fila: nombre de la ley + título del bloque, por join al parent.

    Incluye la LEY (`parent.citation.label`, "Ley 39/2015") junto al título (`full_title`→`title`,
    "Artículo 122. Recursos") a propósito: boostear solo el nº de artículo hace colisionar dos
    "Artículo 122" de leyes distintas (el barrido lo mostró en q0077); añadir la ley refuerza el
    discriminador para que la ley citada desempate. Devuelve "" si no hay ni ley ni título.
    """
    parents_by_id = corpus.get("parents_by_id", {})
    out: list[str] = []
    for row in rows:
        parent = parents_by_id.get(row["parent_id"], {})
        law = (parent.get("citation") or {}).get("label") or ""
        title = parent.get("full_title") or parent.get("title") or ""
        out.append(f"{law} {title}".strip())
    return out


class LexicalIndex:
    """Índice BM25 sobre las rows de un bundle. `search` espeja `ExactDenseIndex.search`."""

    def __init__(
        self,
        *,
        rows: list[dict],
        texts: list[str],
        headings: list[str] | None = None,
        heading_boost: int = 0,
        manifest: dict | None = None,
        analyzer: SpanishAnalyzer | None = None,
    ) -> None:
        if len(rows) != len(texts):
            raise ValueError(
                f"rows ({len(rows)}) y texts ({len(texts)}) deben tener igual longitud."
            )
        if heading_boost < 0:
            raise ValueError(f"heading_boost debe ser >= 0 (recibido {heading_boost}).")
        headings = headings if headings is not None else [""] * len(texts)
        if len(headings) != len(texts):
            raise ValueError(
                f"headings ({len(headings)}) y texts ({len(texts)}) deben tener igual longitud."
            )
        self.rows = rows
        self.manifest = manifest or {}
        self.analyzer = analyzer or SpanishAnalyzer()
        # BM25F-lite: a los tokens de cada doc se añaden `heading_boost` copias EXTRA de los de su
        # cabecera (título del artículo). Como la cabecera ya está 1× en el `retrieval_text`,
        # `heading_boost=0` reproduce el comportamiento sin boost; subirlo pesa más el nº de
        # artículo y la rúbrica (tokens raros), que es donde el léxico debe ganar.
        self.heading_boost = heading_boost
        tokenized: list[list[str]] = []
        for text, heading in zip(texts, headings, strict=True):
            toks = self.analyzer.analyze(text)
            if heading_boost and heading:
                toks = toks + self.analyzer.analyze(heading) * heading_boost
            tokenized.append(toks)
        self._bm25 = BM25Okapi(tokenized)
        # Conjunto de tokens por doc: el "match" se decide por SOLAPE léxico real, no por el signo
        # del score (el IDF de Okapi puede ser 0 o negativo y dejar a 0 un solape legítimo cuando un
        # término aparece en ~la mitad de los docs o el corpus es pequeño). El boost no altera el
        # conjunto (set), solo el ranking → la elegibilidad de match se mantiene.
        self._doc_tokens: list[set[str]] = [set(toks) for toks in tokenized]

    @classmethod
    def from_bundle(
        cls,
        bundle_dir: str | Path,
        *,
        corpus: dict,
        analyzer: SpanishAnalyzer | None = None,
        heading_boost: int = 0,
    ) -> LexicalIndex:
        """Construye el índice léxico sobre las rows del bundle denso (mismo orden/ids)."""
        manifest, rows, _embeddings = load_validated_bundle(Path(bundle_dir), corpus=corpus)
        return cls(
            rows=rows,
            texts=row_texts(rows, corpus),
            headings=row_boost_text(rows, corpus),
            heading_boost=heading_boost,
            manifest=manifest,
            analyzer=analyzer,
        )

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
