"""Tests de la fusión del recuperador híbrido (offline, sin índices reales)."""

import pytest

from src.retrieval.dense_retriever import DenseHit
from src.retrieval.hybrid_retriever import HybridRetriever


def _hit(eid: str, parent: str, rank: int, score: float) -> DenseHit:
    return DenseHit(
        rank=rank,
        score=score,
        row_index=rank - 1,
        embedding_input_id=eid,
        document_id="BOE-A-0001",
        block_id="a1",
        parent_id=parent,
        source={},
        context_anchor=None,
        retrieval_text="texto",
        citation_label=parent,
        citation_url=None,
    )


class _FixedRetriever:
    """Recuperador de prueba: devuelve una lista fija (cumple el `Retriever` Protocol)."""

    def __init__(self, hits: list[DenseHit]) -> None:
        self._hits = hits

    def resolved_query_profile_id(self, query_profile_id: str | None = None) -> str:
        return "fake"

    def retrieve(
        self,
        query: str,
        *,
        query_profile_id: str | None = None,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[DenseHit]:
        return self._hits[:top_k]


def _hybrid(fusion: str, **kwargs) -> HybridRetriever:
    dense = _FixedRetriever([_hit("e_a", "p_a", 1, 0.9), _hit("e_b", "p_b", 2, 0.8)])
    lexical = _FixedRetriever([_hit("e_b", "p_b", 1, 5.0), _hit("e_c", "p_c", 2, 3.0)])
    return HybridRetriever(dense=dense, lexical=lexical, fusion=fusion, candidates=10, **kwargs)


def test_rrf_fusiona_y_reordena() -> None:
    hits = _hybrid("rrf").retrieve("q", top_k=3)
    # e_b aparece en ambos (rank2 denso + rank1 léxico) → suma RRF mayor → primero.
    assert [h.embedding_input_id for h in hits] == ["e_b", "e_a", "e_c"]
    assert [h.rank for h in hits] == [1, 2, 3]
    assert hits[0].score > hits[1].score > hits[2].score


def test_weighted_normaliza_y_combina() -> None:
    ids = [h.embedding_input_id for h in _hybrid("weighted", alpha=0.5).retrieve("q", top_k=3)]
    assert ids[-1] == "e_c"  # solo en léxico, score normalizado mínimo
    assert set(ids[:2]) == {"e_a", "e_b"}


def test_top_k_recorta_el_resultado() -> None:
    assert len(_hybrid("rrf").retrieve("q", top_k=1)) == 1


def test_fusion_invalida_lanza_error() -> None:
    with pytest.raises(ValueError):
        _hybrid("nope")


def test_alpha_fuera_de_rango_lanza_error() -> None:
    with pytest.raises(ValueError):
        _hybrid("weighted", alpha=1.5)
