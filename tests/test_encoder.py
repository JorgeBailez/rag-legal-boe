"""Tests del DenseEncoder con un modelo inyectado (offline, sin pesos reales)."""

import numpy as np
import pytest

from src.embeddings.encoder import (
    DenseEncoder,
    RemoteCodeNotReviewedError,
    RevisionUnpinnedError,
    load_tokenizer,
    read_hf_token,
)
from src.embeddings.model_registry import get_contract


class DummyModel:
    """Modelo ST falso: registra los textos recibidos y devuelve vectores deterministas."""

    def __init__(self, dim: int, captured: dict) -> None:
        self.dim = dim
        self.captured = captured

    def encode(
        self, texts, *, batch_size, show_progress_bar, normalize_embeddings, convert_to_numpy
    ):
        self.captured["texts"] = list(texts)
        self.captured["normalize"] = normalize_embeddings
        # float64 a propósito: el encoder debe castear a float32.
        return np.ones((len(texts), self.dim), dtype=np.float64)


def test_revision_unpinned_blocks_without_flag() -> None:
    # model=None fuerza la carga real; el guard de revisión salta ANTES de importar ST.
    with pytest.raises(RevisionUnpinnedError):
        DenseEncoder(get_contract("e5-base"), allow_unpinned_revision=False)


def test_remote_code_not_reviewed_blocks_tokenizer_even_when_unpinned_allowed() -> None:
    with pytest.raises(RemoteCodeNotReviewedError, match="remote_code_reviewed=True"):
        load_tokenizer(get_contract("gte-multilingual-base"), allow_unpinned_revision=True)


def test_remote_code_not_reviewed_blocks_model_even_when_unpinned_allowed() -> None:
    with pytest.raises(RemoteCodeNotReviewedError, match="remote_code_reviewed=True"):
        DenseEncoder(get_contract("gte-multilingual-base"), allow_unpinned_revision=True)


def test_encode_documents_verbatim_float32_and_dim() -> None:
    captured: dict = {}
    contract = get_contract("e5-base")  # dim 768
    enc = DenseEncoder(contract, model=DummyModel(768, captured))
    emb = enc.encode_documents(["passage: hola", "passage: mundo"], show_progress=False)
    assert emb.dtype == np.float32
    assert emb.shape == (2, 768)
    assert captured["texts"] == ["passage: hola", "passage: mundo"]  # verbatim, sin reformatear
    assert captured["normalize"] is True


def test_encode_queries_applies_query_profile() -> None:
    captured: dict = {}
    contract = get_contract("e5-base")
    enc = DenseEncoder(contract, model=DummyModel(768, captured))
    enc.encode_queries(["¿plazo?"], show_progress=False)
    assert captured["texts"] == ["query: ¿plazo?"]  # query profile del contrato e5


def test_dimension_mismatch_raises() -> None:
    contract = get_contract("e5-base")  # espera 768
    enc = DenseEncoder(contract, model=DummyModel(16, {}))
    with pytest.raises(ValueError):
        enc.encode_documents(["x"], show_progress=False)


def test_empty_inputs_return_empty_matrix() -> None:
    contract = get_contract("e5-base")
    enc = DenseEncoder(contract, model=DummyModel(768, {}))
    assert enc.encode_documents([]).shape == (0, 768)
    assert enc.encode_queries([]).shape == (0, 768)


def test_read_hf_token_optional_and_silent(monkeypatch, capsys) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert read_hf_token() is None
    monkeypatch.setenv("HF_TOKEN", "secreto-de-prueba")
    assert read_hf_token() == "secreto-de-prueba"
    out = capsys.readouterr()
    assert "secreto-de-prueba" not in out.out + out.err  # nunca se imprime
