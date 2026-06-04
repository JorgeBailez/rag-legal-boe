"""Perfilado de tokenizadores: mide los tokens del input de embedding y cuantifica truncamientos.

Resuelve `H3_oversized_token_measurement`: el chunker mide **caracteres** (límite blando), pero
los modelos limitan **tokens**. Aquí se mide, por contrato de modelo, cuántos tokens ocupa el
input que realmente se embeberá (`document_formatter(retrieval_text)`) y se compara con el límite
**efectivo** del modelo, registrando margen y exceso **sin pérdidas silenciosas**.

Lógica pura: no importa `transformers` ni hace red. El tokenizador se inyecta (cualquier objeto
con `model_max_length` y `encode(text, add_special_tokens=...) -> list[int]`), de modo que es
testeable con un *fake tokenizer*.
"""

from __future__ import annotations

from collections import Counter
from typing import Protocol

from src.embeddings.model_registry import ModelContract

# Límites por encima de este umbral se consideran *sentinel* (HF usa ~1e30 si no hay límite real).
SENTINEL_THRESHOLD = 100_000


class TokenizerLike(Protocol):
    """Interfaz mínima del tokenizador que necesita el perfilador."""

    model_max_length: int

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...


def resolve_effective_max(
    declared_max_tokens: int, tokenizer_model_max_length: int | None
) -> tuple[int, str]:
    """Determina el límite de tokens **efectivo** y su origen.

    Si el tokenizador no declara límite o reporta un *sentinel* enorme (o no positivo), se usa el
    `declared_max_tokens` del contrato. Si reporta un valor válido, se usa ese (y la divergencia
    con el declarado queda visible al registrarse ambos por separado).
    """
    tml = tokenizer_model_max_length
    if tml is None or tml <= 0 or tml > SENTINEL_THRESHOLD:
        return declared_max_tokens, "declared"
    return tml, "tokenizer"


def count_tokens(tokenizer: TokenizerLike, text: str) -> int:
    """Cuenta tokens **incluyendo special tokens**, sin truncar."""
    return len(tokenizer.encode(text, add_special_tokens=True))


def profile_text(tokenizer: TokenizerLike, text: str, effective_max: int) -> dict:
    """Perfila un único texto: tokens, truncado, margen y exceso (sin pérdida silenciosa)."""
    n = count_tokens(tokenizer, text)
    excess = max(0, n - effective_max)
    return {
        "n_tokens": n,
        "truncated": n > effective_max,
        "margin": effective_max - n,
        "excess_tokens": excess,
    }


def profile_chunk(
    contract: ModelContract,
    tokenizer: TokenizerLike,
    chunk: dict,
    effective_max: int,
    parent_text: str = "",
    block_type: str | None = None,
) -> dict:
    """Perfila un chunk separando el input de embedding del contexto del padre (H3 vs LLM).

    `parent_text` se resuelve por **join** al parent store (chunks v2 no lo llevan). `block_type`
    procede del descriptor del documento. Ambos son informativos: NO afectan al riesgo de truncado
    del embedding (que depende solo de `retrieval_text` formateado).
    """
    retrieval_text = chunk.get("retrieval_text", "") or ""
    formatted = contract.format_document(retrieval_text)
    return {
        "chunk_id": chunk.get("chunk_id"),
        "block_id": chunk.get("block_id"),
        "block_type": block_type,
        # Input REAL de embedding (lo que decide H3).
        "embedding_input": profile_text(tokenizer, formatted, effective_max),
        # Solo informativo; NO se mezcla con el riesgo de truncado del embedding.
        "parent_context": {
            "text": profile_text(tokenizer, chunk.get("text", "") or "", effective_max),
            "parent_text": profile_text(tokenizer, parent_text or "", effective_max),
        },
    }


def _percentile(values: list[int], p: float) -> float:
    """Percentil por interpolación lineal (sin numpy)."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return float(s[f])
    return s[f] + (s[c] - s[f]) * (k - f)


def _aggregate(token_counts: list[int], truncations: list[bool]) -> dict:
    """Estadísticos agregados de una colección de perfiles (tokens + truncados)."""
    n = len(token_counts)
    n_trunc = sum(1 for t in truncations if t)
    return {
        "n_items": n,
        "n_truncated": n_trunc,
        "pct_truncated": round(100.0 * n_trunc / n, 2) if n else 0.0,
        "max_tokens": max(token_counts) if token_counts else 0,
        "p50_tokens": round(_percentile(token_counts, 50), 1),
        "p90_tokens": round(_percentile(token_counts, 90), 1),
        "p95_tokens": round(_percentile(token_counts, 95), 1),
        "p99_tokens": round(_percentile(token_counts, 99), 1),
    }


def aggregate_embedding_inputs(chunk_profiles: list[dict]) -> dict:
    """Agrega los perfiles del input de embedding, global y por `block_type`."""
    counts = [p["embedding_input"]["n_tokens"] for p in chunk_profiles]
    truncs = [p["embedding_input"]["truncated"] for p in chunk_profiles]
    overall = _aggregate(counts, truncs)

    by_type: dict[str, dict] = {}
    types = Counter(p.get("block_type") for p in chunk_profiles)
    for bt in types:
        sub = [p for p in chunk_profiles if p.get("block_type") == bt]
        by_type[str(bt)] = _aggregate(
            [p["embedding_input"]["n_tokens"] for p in sub],
            [p["embedding_input"]["truncated"] for p in sub],
        )
    truncated_chunk_ids = [
        p["chunk_id"] for p in chunk_profiles if p["embedding_input"]["truncated"]
    ]
    return {
        "overall": overall,
        "by_block_type": by_type,
        "truncated_chunk_ids": truncated_chunk_ids,
    }


def aggregate_parent_context(chunk_profiles: list[dict]) -> dict:
    """Agrega los perfiles de `text` y `parent_text` (informativo, separado del embedding)."""
    out = {}
    for field in ("text", "parent_text"):
        counts = [p["parent_context"][field]["n_tokens"] for p in chunk_profiles]
        truncs = [p["parent_context"][field]["truncated"] for p in chunk_profiles]
        out[field] = _aggregate(counts, truncs)
    return out


def profile_model(
    contract: ModelContract,
    tokenizer: TokenizerLike,
    chunks: list[dict],
    *,
    parent_text_by_id: dict[str, str] | None = None,
    block_type_by_id: dict[str, str] | None = None,
    keep_per_chunk: bool = False,
) -> dict:
    """Perfila todos los chunks para un modelo y devuelve su sección del informe.

    `parent_text_by_id` (parent_id → texto del parent) y `block_type_by_id` (block_id → tipo)
    resuelven por join el contexto informativo; si no se pasan, se usan valores vacíos.
    """
    parent_text_by_id = parent_text_by_id or {}
    block_type_by_id = block_type_by_id or {}
    effective_max, source = resolve_effective_max(
        contract.declared_max_tokens, getattr(tokenizer, "model_max_length", None)
    )
    chunk_profiles = [
        profile_chunk(
            contract,
            tokenizer,
            ch,
            effective_max,
            parent_text=parent_text_by_id.get(ch.get("parent_id"), ""),
            block_type=block_type_by_id.get(ch.get("block_id")),
        )
        for ch in chunks
    ]
    result = {
        "model_id": contract.model_id,
        "model_revision": contract.model_revision,
        "tokenizer_revision": contract.tokenizer_revision,
        "expected_embedding_dimension": contract.expected_embedding_dimension,
        "declared_max_tokens": contract.declared_max_tokens,
        "tokenizer_model_max_length": getattr(tokenizer, "model_max_length", None),
        "effective_max_tokens": effective_max,
        "source_of_effective_limit": source,
        "embedding_input_profile": aggregate_embedding_inputs(chunk_profiles),
        "parent_context_profile": aggregate_parent_context(chunk_profiles),
    }
    if keep_per_chunk:
        result["per_chunk"] = chunk_profiles
    return result
