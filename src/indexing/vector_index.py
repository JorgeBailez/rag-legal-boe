"""Índice vectorial exacto (dense-only): producto escalar sobre vectores L2-normalizados.

Carga las rows de un bundle a memoria y la matriz `embeddings.npy` por mmap de solo lectura
(`allow_pickle=False`). La búsqueda es **exacta** (no ANN): `scores = embeddings @ q` y top-k por
`argsort` estable. Los filtros son opcionales y se resuelven por **join ligero** contra los chunks
o descriptors oficiales (no se duplican en el bundle). El benchmark principal corre sin filtros.

Sin interfaz abstracta: una sola implementación concreta para esta fase.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.embeddings.bundle import load_validated_bundle

# Claves de filtro soportadas (proyección de Fase 1).
BOOL_FILTERS = ("without_content", "annex", "table", "image")
SCALAR_FILTERS = ("rank_code", "scope_code", "semantic_role")


def _row_filter_attrs(
    row: dict,
    *,
    chunks_by_id: dict[str, dict],
    parents_by_id: dict[str, dict],
    documents_by_id: dict[str, dict],
) -> dict:
    """Resuelve los atributos de filtro de una row por join (chunk si existe; si no, parent+doc)."""
    source = row.get("source", {})
    chunk_id = source.get("chunk_id")
    chunk = chunks_by_id.get(chunk_id) if chunk_id else None
    if chunk is not None:
        return dict(chunk.get("filters") or {})

    # Row derivada de parent (C1): se reconstruye desde parent + metadatos del documento.
    parent = parents_by_id.get(row.get("parent_id"), {})
    doc = documents_by_id.get(row.get("document_id"), {})
    meta = doc.get("metadata") or {}
    subjects = (doc.get("analysis") or {}).get("subjects", [])
    return {
        "rank_code": (meta.get("rank") or {}).get("code"),
        "scope_code": (meta.get("scope") or {}).get("code"),
        "subject_codes": [s.get("code") for s in subjects if s.get("code")],
        "semantic_role": parent.get("semantic_role"),
        "without_content": parent.get("is_without_content", False),
        "annex": parent.get("is_annex", False),
        "table": parent.get("contains_table", False),
        "image": parent.get("contains_image", False),
    }


def _matches(attrs: dict, filters: dict) -> bool:
    for key in SCALAR_FILTERS:
        if key in filters and attrs.get(key) != filters[key]:
            return False
    for key in BOOL_FILTERS:
        if key in filters and bool(attrs.get(key, False)) != bool(filters[key]):
            return False
    if "subject_codes" in filters:
        wanted = set(filters["subject_codes"])
        if wanted and not wanted & set(attrs.get("subject_codes") or []):
            return False
    return True


def build_filter_mask(rows: list[dict], corpus: dict, filters: dict) -> np.ndarray:
    """Máscara booleana (n_rows,) en memoria; True = la row pasa todos los filtros indicados."""
    if not filters:
        return np.ones(len(rows), dtype=bool)
    chunks_by_id = {c["chunk_id"]: c for c in corpus.get("chunks", [])}
    parents_by_id = corpus.get("parents_by_id", {})
    documents_by_id = corpus.get("documents_by_id", {})
    mask = np.zeros(len(rows), dtype=bool)
    for i, row in enumerate(rows):
        attrs = _row_filter_attrs(
            row,
            chunks_by_id=chunks_by_id,
            parents_by_id=parents_by_id,
            documents_by_id=documents_by_id,
        )
        mask[i] = _matches(attrs, filters)
    return mask


class ExactDenseIndex:
    """Índice denso exacto sobre un bundle publicado (mmap + dot product + argsort estable)."""

    def __init__(self, bundle_dir: Path, *, corpus: dict | None = None) -> None:
        self.bundle_dir = Path(bundle_dir)
        self.manifest, self.rows, self.embeddings = load_validated_bundle(
            self.bundle_dir, corpus=corpus
        )
        self.dimension = self.manifest["artifacts"]["embedding_dimension"]

    @classmethod
    def from_bundle(cls, bundle_dir: Path, *, corpus: dict | None = None) -> ExactDenseIndex:
        """Carga un bundle. Sin `corpus` se omiten las comprobaciones bundle↔corpus (ver
        `load_validated_bundle`): apto para búsqueda sin filtros ni resolución de texto."""
        return cls(bundle_dir, corpus=corpus)

    def __len__(self) -> int:
        return len(self.rows)

    def search(
        self, query_vector: np.ndarray, *, k: int = 5, mask: np.ndarray | None = None
    ) -> list[dict]:
        """Top-k por producto escalar. `mask` (opcional) restringe las filas candidatas."""
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dimension:
            raise ValueError(f"dim query {q.shape[0]} != dim índice {self.dimension}")
        scores = np.asarray(self.embeddings, dtype=np.float32) @ q
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores, kind="stable")[:k]
        hits: list[dict] = []
        for rank, idx in enumerate(order, start=1):
            score = float(scores[idx])
            if score == float("-inf"):
                break  # no quedan candidatos tras el filtro
            row = self.rows[idx]
            hits.append(
                {
                    "rank": rank,
                    "score": score,
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
