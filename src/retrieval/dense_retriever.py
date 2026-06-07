"""Wrapper reutilizable del retrieval denso exacto (Fase 2 → Fase 3).

Extrae la lógica común que antes vivía incrustada en `scripts/query_dense_index.py`: cargar el
bundle, resolver el contrato del manifest, validar compatibilidad, codificar la query con un perfil
reproducible, filtrar (opcional) y resolver cada hit a su texto recuperado + cita autoritativa
(etiqueta + URL del corpus, nunca generadas por el LLM).

Decisiones de diseño:
- Los joins (`chunk_id → chunk`) se precalculan UNA vez al construir el retriever, no dentro del
  bucle de hits.
- El índice (`ExactDenseIndex`) y el encoder (`DenseEncoder` / fake) son **inyectables** para
  poder probar offline sin pesos reales ni bundle real.
- No introduce BM25, híbrido ni reranking: es dense-only, concreto para `ExactDenseIndex`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src.embeddings.corpus_loader import load_processed_corpus
from src.embeddings.model_registry import (
    ModelContract,
    assert_bundle_compatible,
    get_contract,
    query_profile_metadata,
)
from src.indexing.vector_index import ExactDenseIndex, build_filter_mask


def resolve_hit_text_and_citation(
    hit: dict,
    *,
    chunks_by_id: dict[str, dict],
    parents_by_id: dict[str, dict],
) -> tuple[str, dict]:
    """Texto recuperado (K_ONLY) + cita de un hit por join al chunk (si existe) o al parent.

    Para rows `chunk_field` se respeta el campo realmente indexado (`source.field`, p. ej.
    `retrieval_text` en J1 o `text` en J2), con *fallback* a `chunk["text"]` solo si ese campo no
    existe (compatibilidad con bundles antiguos). Para rows `derived_text` se conserva
    `source["text"]` (texto nuevo persistido en la propia row).

    Función pura (recibe los joins ya construidos): la usa el retriever con sus mapas
    precalculados y también el shim de `scripts/query_dense_index.py`.
    """
    source = hit["source"]
    chunk_id = source.get("chunk_id")
    if source.get("kind") == "derived_text" and source.get("text") is not None:
        if chunk_id:
            chunk = chunks_by_id.get(chunk_id, {})
            return source["text"], chunk.get("citation", {})
        parent = parents_by_id.get(hit["parent_id"], {})
        return source["text"], parent.get("citation", {})
    if chunk_id:
        chunk = chunks_by_id.get(chunk_id, {})
        field_name = source.get("field") or "text"
        text = chunk.get(field_name)
        if text is None:
            text = chunk.get("text", "")  # fallback para bundles sin ese campo
        return text, chunk.get("citation", {})
    parent = parents_by_id.get(hit["parent_id"], {})
    return source.get("text", parent.get("text", "")), parent.get("citation", {})


class QueryEncoder(Protocol):
    """Interfaz mínima que necesita el retriever (la cumplen `DenseEncoder` y `FakeEncoder`)."""

    def encode_queries(
        self,
        queries: list[str],
        *,
        query_profile_id: str | None = ...,
        show_progress: bool = ...,
    ) -> np.ndarray: ...


@dataclass
class DenseHit:
    """Hit denso resuelto: score + relaciones + texto recuperado + cita autoritativa."""

    rank: int
    score: float
    row_index: int
    embedding_input_id: str
    document_id: str
    block_id: str
    parent_id: str
    source: dict[str, Any]
    context_anchor: dict[str, Any] | None
    retrieval_text: str
    citation_label: str
    citation_url: str | None


class DenseRetriever:
    """Recuperador denso exacto sobre un bundle publicado, con joins precalculados."""

    def __init__(
        self,
        *,
        index: ExactDenseIndex,
        encoder: QueryEncoder,
        contract: ModelContract,
        corpus: dict,
    ) -> None:
        self.index = index
        self.encoder = encoder
        self.contract = contract
        self.corpus = corpus
        self.manifest = index.manifest
        self.bundle_id = index.manifest["bundle"]["bundle_id"]
        self.model_alias = index.manifest["bundle"]["model_alias"]
        # Joins precalculados una sola vez (no por hit ni por query).
        self._chunks_by_id: dict[str, dict] = {c["chunk_id"]: c for c in corpus.get("chunks", [])}
        self._parents_by_id: dict[str, dict] = corpus.get("parents_by_id", {})

    @classmethod
    def from_bundle(
        cls,
        bundle_dir: str | Path,
        *,
        corpus: dict | None = None,
        encoder: QueryEncoder | None = None,
        batch_size: int = 32,
        device: str = "cpu",
        allow_unpinned_revision: bool = False,
    ) -> DenseRetriever:
        """Carga corpus + índice + contrato y construye el retriever (encoder inyectable)."""
        corpus = corpus if corpus is not None else load_processed_corpus()
        index = ExactDenseIndex.from_bundle(Path(bundle_dir), corpus=corpus)
        contract = get_contract(index.manifest["bundle"]["model_alias"])
        assert_bundle_compatible(contract, index.manifest)
        if encoder is None:
            # Import perezoso: el encoder real arrastra torch/transformers (no en tests offline).
            from src.embeddings.encoder import DenseEncoder

            encoder = DenseEncoder(
                contract,
                device=device,
                batch_size=batch_size,
                allow_unpinned_revision=allow_unpinned_revision,
            )
        return cls(index=index, encoder=encoder, contract=contract, corpus=corpus)

    def resolved_query_profile_id(self, query_profile_id: str | None) -> str:
        """ID del perfil de consulta efectivamente aplicado (resuelve el default del contrato)."""
        return query_profile_metadata(self.contract, query_profile_id)["query_profile_id"]

    def retrieve(
        self,
        query: str,
        *,
        query_profile_id: str | None = None,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[DenseHit]:
        """Codifica la query, recupera top-k y resuelve texto + cita de cada hit por join ligero."""
        if top_k <= 0:
            raise ValueError(f"top_k debe ser > 0 (recibido {top_k}).")
        if not query or not query.strip():
            raise ValueError("la query no puede estar vacía ni contener solo espacios.")
        qv = self.encoder.encode_queries(
            [query], query_profile_id=query_profile_id, show_progress=False
        )[0]
        mask = build_filter_mask(self.index.rows, self.corpus, filters) if filters else None
        raw_hits = self.index.search(qv, k=top_k, mask=mask)
        return [self._resolve_hit(h) for h in raw_hits]

    def _resolve_hit(self, hit: dict) -> DenseHit:
        """Resuelve el texto recuperado (K_ONLY) y la cita autoritativa (label/url) de un hit."""
        text, citation = self._resolve_text_and_citation(hit)
        return DenseHit(
            rank=hit["rank"],
            score=hit["score"],
            row_index=hit["row_index"],
            embedding_input_id=hit["embedding_input_id"],
            document_id=hit["document_id"],
            block_id=hit["block_id"],
            parent_id=hit["parent_id"],
            source=hit["source"],
            context_anchor=hit.get("context_anchor"),
            retrieval_text=text,
            citation_label=citation.get("label") or hit["parent_id"],
            citation_url=citation.get("url"),
        )

    def _resolve_text_and_citation(self, hit: dict) -> tuple[str, dict]:
        """Texto recuperado + cita usando los joins precalculados del retriever."""
        return resolve_hit_text_and_citation(
            hit, chunks_by_id=self._chunks_by_id, parents_by_id=self._parents_by_id
        )


# Dataclass auxiliar para mantener la firma de los campos de filtro en un solo lugar reutilizable.
@dataclass
class RetrievalFilters:
    """Filtros opcionales de retrieval (proyección de los flags de Fase 1)."""

    rank_code: str | None = None
    scope_code: str | None = None
    semantic_role: str | None = None
    subject_codes: list[str] = field(default_factory=list)
    annex: bool = False
    table: bool = False
    image: bool = False
    without_content: bool = False

    def as_filter_dict(self) -> dict:
        """Construye el dict que consume `build_filter_mask` (omite claves no activadas)."""
        filters: dict = {}
        if self.rank_code:
            filters["rank_code"] = self.rank_code
        if self.scope_code:
            filters["scope_code"] = self.scope_code
        if self.semantic_role:
            filters["semantic_role"] = self.semantic_role
        if self.subject_codes:
            filters["subject_codes"] = self.subject_codes
        for flag in ("annex", "table", "image", "without_content"):
            if getattr(self, flag):
                filters[flag] = True
        return filters
