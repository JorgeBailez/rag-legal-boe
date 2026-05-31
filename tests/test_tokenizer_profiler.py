"""Tests del perfilador de tokenizadores (sin red, con fake tokenizer determinista).

Validan los contratos de modelo, la resolución del límite efectivo (incl. caso *sentinel*), el
conteo con special tokens, el truncado/margen/exceso (sin pérdidas silenciosas) y la agregación,
separando el input de embedding del contexto del padre.
"""

from src.embeddings.model_registry import CANDIDATES, ModelContract, get_contract
from src.embeddings.tokenizer_profiler import (
    SENTINEL_THRESHOLD,
    aggregate_embedding_inputs,
    count_tokens,
    profile_chunk,
    profile_model,
    profile_text,
    resolve_effective_max,
)


class FakeTokenizer:
    """Tokenizer determinista: 1 token por palabra + `special` tokens si add_special_tokens."""

    def __init__(self, model_max_length: int, special: int = 2) -> None:
        self.model_max_length = model_max_length
        self.special = special

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        n = len(text.split()) + (self.special if add_special_tokens else 0)
        return list(range(n))


def _chunk(chunk_id: str, retrieval_text: str, block_type: str = "precepto") -> dict:
    return {
        "chunk_id": chunk_id,
        "block_id": chunk_id.split("__")[1] if "__" in chunk_id else chunk_id,
        "text": retrieval_text,
        "retrieval_text": retrieval_text,
        "parent_text": retrieval_text + " contexto del bloque padre completo",
        "metadata": {"block_type": block_type},
    }


# --- contratos de modelo ----------------------------------------------------


def test_registry_has_five_candidates() -> None:
    assert len(CANDIDATES) == 5
    assert "intfloat/multilingual-e5-large" in CANDIDATES  # baseline histórico


def test_e5_document_and_query_formatters() -> None:
    c = get_contract("intfloat/multilingual-e5-base")
    assert c.format_document("Artículo 1.") == "passage: Artículo 1."
    assert c.format_query("¿plazo?") == "query: ¿plazo?"


def test_instruct_query_formatter_uses_instruction() -> None:
    c = get_contract("intfloat/multilingual-e5-large-instruct")
    assert c.format_document("texto") == "texto"  # documentos sin prefijo
    q = c.format_query("¿plazo de recurso?")
    assert q.startswith("Instruct: ")
    assert "Query: ¿plazo de recurso?" in q


def test_bge_and_qwen_have_no_prefix() -> None:
    assert get_contract("BAAI/bge-m3").format_document("x") == "x"
    assert get_contract("Qwen/Qwen3-Embedding-0.6B").format_document("x") == "x"


def test_formatter_is_literal_replacement_safe_with_braces() -> None:
    # Texto jurídico con llaves no debe romper el formateo.
    c = get_contract("BAAI/bge-m3")
    assert c.format_document("conjunto {a, b}") == "conjunto {a, b}"


# --- límite efectivo (sentinel) ---------------------------------------------


def test_effective_max_uses_declared_on_sentinel() -> None:
    eff, source = resolve_effective_max(512, int(1e30))
    assert eff == 512
    assert source == "declared"


def test_effective_max_uses_declared_when_none() -> None:
    eff, source = resolve_effective_max(8192, None)
    assert (eff, source) == (8192, "declared")


def test_effective_max_uses_tokenizer_when_valid() -> None:
    eff, source = resolve_effective_max(512, 514)
    assert (eff, source) == (514, "tokenizer")


def test_effective_max_threshold_boundary() -> None:
    assert resolve_effective_max(512, SENTINEL_THRESHOLD + 1)[1] == "declared"
    assert resolve_effective_max(512, SENTINEL_THRESHOLD)[1] == "tokenizer"


# --- conteo y perfil de un texto --------------------------------------------


def test_count_tokens_includes_special_tokens() -> None:
    tok = FakeTokenizer(model_max_length=512, special=2)
    assert count_tokens(tok, "una dos tres") == 5  # 3 palabras + 2 special


def test_profile_text_truncation_margin_excess() -> None:
    tok = FakeTokenizer(model_max_length=10, special=0)
    p = profile_text(tok, "a b c d e f g h i j k l", effective_max=10)  # 12 tokens
    assert p["n_tokens"] == 12
    assert p["truncated"] is True
    assert p["excess_tokens"] == 2  # sin pérdida silenciosa
    assert p["margin"] == -2


def test_profile_text_within_limit() -> None:
    tok = FakeTokenizer(model_max_length=10, special=0)
    p = profile_text(tok, "a b c", effective_max=10)
    assert p["truncated"] is False
    assert p["excess_tokens"] == 0
    assert p["margin"] == 7


# --- perfil de chunk: embedding vs contexto del padre -----------------------


def test_profile_chunk_separates_embedding_and_parent() -> None:
    c = get_contract("intfloat/multilingual-e5-base")  # añade "passage: "
    tok = FakeTokenizer(model_max_length=512, special=0)
    prof = profile_chunk(c, tok, _chunk("D__a1__c001", "Artículo uno dos"), effective_max=512)
    # embedding input añade el prefijo "passage:" -> 1 token más que el texto crudo.
    assert prof["embedding_input"]["n_tokens"] == 4  # passage: + Artículo + uno + dos
    assert "text" in prof["parent_context"] and "parent_text" in prof["parent_context"]
    assert prof["block_type"] == "precepto"


# --- agregación y perfil de modelo ------------------------------------------


def test_aggregate_counts_truncations_and_by_type() -> None:
    c = get_contract("BAAI/bge-m3")  # sin prefijo
    tok = FakeTokenizer(model_max_length=5, special=0)
    chunks = [
        _chunk("D__a1__c001", "una dos tres", "precepto"),  # 3 tokens, ok
        _chunk("D__a2__c001", "una dos tres cuatro cinco seis", "precepto"),  # 6 -> truncado
        _chunk("D__pr__c001", "uno", "preambulo"),  # 1 token, ok
    ]
    profiles = [profile_chunk(c, tok, ch, 5) for ch in chunks]
    agg = aggregate_embedding_inputs(profiles)
    assert agg["overall"]["n_items"] == 3
    assert agg["overall"]["n_truncated"] == 1
    assert agg["truncated_chunk_ids"] == ["D__a2__c001"]
    assert "precepto" in agg["by_block_type"] and "preambulo" in agg["by_block_type"]
    assert agg["by_block_type"]["precepto"]["n_truncated"] == 1


def test_profile_model_end_to_end_records_limit_metadata() -> None:
    # Tokenizer con sentinel -> usa declared (512) del contrato e5-large.
    contract = get_contract("intfloat/multilingual-e5-large")
    tok = FakeTokenizer(model_max_length=int(1e30), special=2)
    chunks = [_chunk("D__a1__c001", "uno dos tres", "precepto")]
    rep = profile_model(contract, tok, chunks)
    assert rep["model_id"] == "intfloat/multilingual-e5-large"
    assert rep["declared_max_tokens"] == 512
    assert rep["effective_max_tokens"] == 512
    assert rep["source_of_effective_limit"] == "declared"
    assert rep["expected_embedding_dimension"] == 1024
    assert "embedding_input_profile" in rep and "parent_context_profile" in rep


def test_profile_model_long_context_no_truncation() -> None:
    contract = ModelContract(
        model_id="fake/long", declared_max_tokens=8192, expected_embedding_dimension=1024
    )
    tok = FakeTokenizer(model_max_length=8192, special=2)
    chunks = [_chunk(f"D__a{i}__c001", "palabra " * 50) for i in range(5)]
    rep = profile_model(contract, tok, chunks)
    assert rep["embedding_input_profile"]["overall"]["n_truncated"] == 0
    assert rep["source_of_effective_limit"] == "tokenizer"
