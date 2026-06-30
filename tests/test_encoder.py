"""Tests del DenseEncoder con un modelo inyectado (offline, sin pesos reales)."""

import numpy as np
import pytest

from src.embeddings.encoder import (
    DenseEncoder,
    RemoteCodeNotReviewedError,
    RevisionUnpinnedError,
    align_max_seq_length,
    load_tokenizer,
    read_hf_token,
)
from src.embeddings.model_registry import ModelContract, get_contract


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


# Contrato sintético SIN revisión pinneada: aísla el guard de revisión del estado real del registry
# (todos los modelos reales están pinneados tras el cierre del MVP, así que ninguno sirve ya para
# este caso). trust_remote_code por defecto False → el guard que salta es el de revisión, no el de
# código remoto.
_SIN_REVISION = ModelContract(
    alias="fake-unpinned",
    model_id="fake/unpinned-model",
    declared_max_tokens=512,
    expected_embedding_dimension=768,
)


def test_revision_unpinned_blocks_without_flag() -> None:
    # model=None fuerza la carga real; el guard de revisión salta ANTES de importar/instanciar ST.
    with pytest.raises(RevisionUnpinnedError):
        DenseEncoder(_SIN_REVISION, allow_unpinned_revision=False)


# Contrato sintético de código remoto SIN revisar: aísla el guard del estado real del registry
# (gte-multilingual-base ya está revisado y pinneado tras el cierre del MVP). Queda unpinned a
# propósito para verificar que el guard de código remoto salta ANTES que el de revisión.
_REMOTE_SIN_REVISAR = ModelContract(
    alias="fake-remote-unreviewed",
    model_id="fake/remote-code-model",
    declared_max_tokens=512,
    expected_embedding_dimension=768,
    trust_remote_code=True,
    remote_code_reviewed=False,
)


def test_remote_code_not_reviewed_blocks_tokenizer_even_when_unpinned_allowed() -> None:
    with pytest.raises(RemoteCodeNotReviewedError, match="remote_code_reviewed=True"):
        load_tokenizer(_REMOTE_SIN_REVISAR, allow_unpinned_revision=True)


def test_remote_code_not_reviewed_blocks_model_even_when_unpinned_allowed() -> None:
    with pytest.raises(RemoteCodeNotReviewedError, match="remote_code_reviewed=True"):
        DenseEncoder(_REMOTE_SIN_REVISAR, allow_unpinned_revision=True)


class _FakeSeqModel:
    """Modelo ST falso: expone tokenizer.model_max_length y un max_seq_length ajustable."""

    def __init__(self, tokenizer_max_length: int, max_seq_length: int) -> None:
        self.tokenizer = type("_Tok", (), {"model_max_length": tokenizer_max_length})()
        self.max_seq_length = max_seq_length


def test_align_max_seq_length_corrige_limite_bajo_del_empaquetado() -> None:
    # Footgun real: modelo de contexto largo cuyo ST viene con max_seq_length=512. El guard debe
    # subirlo al effective_max derivado del tokenizer para no truncar en silencio.
    bge = get_contract("bge-m3")  # declared 8192
    model = _FakeSeqModel(tokenizer_max_length=8192, max_seq_length=512)
    assert align_max_seq_length(model, bge) == 8192
    assert model.max_seq_length == 8192


def test_align_max_seq_length_usa_declarado_si_tokenizer_es_sentinel() -> None:
    # Tokenizer sin límite real (HF reporta ~1e30): se cae al declared_max_tokens del contrato.
    qwen = get_contract("qwen3-0.6b")  # declared 32768
    model = _FakeSeqModel(tokenizer_max_length=10**30, max_seq_length=512)
    assert align_max_seq_length(model, qwen) == 32768
    assert model.max_seq_length == 32768


def test_align_max_seq_length_sin_atributo_devuelve_menos_uno() -> None:
    # Un objeto sin max_seq_length (p. ej. el DummyModel inyectado) no se toca.
    assert align_max_seq_length(object(), get_contract("e5-base")) == -1


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
