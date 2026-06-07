"""Fakes deterministas y corpus sintético para los tests densos (offline, sin pesos reales)."""

from __future__ import annotations

import hashlib

import numpy as np

from src.embeddings.model_registry import format_query_with_profile


class FakeWordTokenizer:
    """Tokenizador reversible por palabras: 1 token/palabra + `special` tokens; decode invertible.

    Mantiene un vocab por instancia (word↔id) de modo que encode→decode es estable dentro de un
    test (suficiente para validar la división token-aware y la prohibición de truncado).
    """

    def __init__(self, model_max_length: int = 512, special: int = 2) -> None:
        self.model_max_length = model_max_length
        self.special = special
        self._vocab: dict[str, int] = {}
        self._inv: dict[int, str] = {}

    def _id(self, word: str) -> int:
        if word not in self._vocab:
            i = len(self._vocab) + 10  # ids de palabra ≥10; el 0 queda para special tokens
            self._vocab[word] = i
            self._inv[i] = word
        return self._vocab[word]

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [self._id(w) for w in text.split()]
        if add_special_tokens:
            ids = [0] * self.special + ids
        return ids

    def decode(self, ids: list[int]) -> str:
        return " ".join(self._inv[i] for i in ids if i in self._inv)


class FakeEncoder:
    """Encoder determinista: texto → vector float32 L2-normalizado reproducible (hash → RNG).

    Misma API pública que `DenseEncoder` (`encode_documents`/`encode_queries`). No formatea
    documentos (recibe los inputs ya formateados por `prepare_inputs`); sí formatea queries si se
    le pasa un contrato.
    """

    backend = "fake"

    def __init__(self, dimension: int = 8, contract=None) -> None:
        self.dimension = dimension
        self.contract = contract

    def _vec(self, text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.dimension).astype(np.float32)
        norm = float(np.linalg.norm(v))
        return v / norm if norm else v

    def encode_documents(
        self, texts: list[str], *, batch_size: int | None = None, show_progress: bool = False
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.vstack([self._vec(t) for t in texts]).astype(np.float32)

    def encode_queries(
        self,
        queries: list[str],
        *,
        query_profile_id: str | None = None,
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Misma firma que `DenseEncoder.encode_queries` (perfil de query reproducible)."""
        formatted = [
            format_query_with_profile(self.contract, q, query_profile_id)
            if self.contract is not None
            else q
            for q in queries
        ]
        if not formatted:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.vstack([self._vec(t) for t in formatted]).astype(np.float32)


def _paragraphs(texts: list[str]) -> list[dict]:
    return [{"order": i, "class": "parrafo", "text": t} for i, t in enumerate(texts, start=1)]


def synthetic_corpus() -> dict:
    """Corpus sintético: 2 docs, 4 parents, chunk largo, overflow, parent largo y un filtro (anexo).

    Devuelve {chunks, parents_by_id, documents_by_id} con la misma forma que el corpus real.
    """
    d1, d2 = "BOE-A-0001", "BOE-A-0002"

    # Parent corto (2 párrafos) → 1 chunk.
    p_a1_paras = ["Articulo uno apartado primero.", "Apartado segundo del articulo uno."]
    # Parent medio (3 párrafos).
    p_a2_paras = [
        "Articulo dos sobre plazos administrativos.",
        "El plazo general sera de tres meses.",
        "Salvo norma especial que indique otro plazo.",
    ]
    # Parent largo (muchas palabras) → C1 produce varias ventanas; overflow con tokenizer pequeño.
    long_words = " ".join(f"palabra{i}" for i in range(120))
    p_long_paras = [long_words, "Cierre del articulo largo."]
    # Anexo (para filtro).
    p_anex_paras = ["Anexo tecnico con tabla descriptiva.", "Fila uno; fila dos."]

    parents = {
        f"{d1}__a1": _parent(d1, "a1", p_a1_paras),
        f"{d1}__a2": _parent(d1, "a2", p_a2_paras),
        f"{d2}__a1": _parent(d2, "a1", p_long_paras),
        f"{d2}__anexo": _parent(d2, "anexo", p_anex_paras, is_annex=True),
    }

    chunks = [
        _chunk(d1, "a1", 1, 1, p_a1_paras),
        _chunk(d1, "a2", 1, 1, p_a2_paras),
        _chunk(d2, "a1", 1, 1, p_long_paras),  # chunk "largo"
        _chunk(d2, "anexo", 1, 1, p_anex_paras, annex=True),
    ]

    documents = {
        d1: {"document_id": d1, "metadata": {"short_title": "Ley 1/2000"}},
        d2: {"document_id": d2, "metadata": {"short_title": "Ley 2/2000"}},
    }
    return {"chunks": chunks, "parents_by_id": parents, "documents_by_id": documents}


def _parent(doc: str, bid: str, paras: list[str], *, is_annex: bool = False) -> dict:
    return {
        "parent_id": f"{doc}__{bid}",
        "document_id": doc,
        "block_id": bid,
        "order": 1,
        "text": "\n".join(paras),
        "paragraphs": _paragraphs(paras),
        "is_annex": is_annex,
    }


def _chunk(
    doc: str, bid: str, index: int, count: int, paras: list[str], *, annex: bool = False
) -> dict:
    text = "\n".join(paras)
    return {
        "chunk_id": f"{doc}__{bid}__c{index:03d}",
        "parent_id": f"{doc}__{bid}",
        "document_id": doc,
        "block_id": bid,
        "position": {"index": index, "count_for_parent": count},
        "text": text,
        "retrieval_text": f"Contexto. {text}",
        "citation": {"label": f"{doc} {bid}", "url": f"https://boe/{doc}#{bid}"},
        "filters": {
            "rank_code": "1300",
            "scope_code": "1",
            "subject_codes": ["5703"],
            "semantic_role": "annex" if annex else "precept",
            "without_content": False,
            "annex": annex,
            "table": annex,
            "image": False,
        },
    }
