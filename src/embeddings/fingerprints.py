"""Fingerprints deterministas (JSON canónico interno → SHA-256).

Tres fingerprints distintos (no se mezclan):
- `source_corpus_fingerprint`: identidad del corpus de Fase 1 consumido (chunks + parents).
- `embedding_inputs_fingerprint`: identidad de los inputs preparados (view-específica).
- `document_contract_fingerprint`: identidad de los embeddings documentales (modelo + plantilla +
  límite + vista); sus 12 primeros hex forman parte del `bundle_id`.

El JSON canónico usa `sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False`
y UTF-8. El JSON legible (manifest) puede ir indentado: el fingerprint se calcula sobre la forma
canónica interna, no sobre el fichero humano.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from src.embeddings.model_registry import ModelContract


def canonical_json(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def fingerprint(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj)).hexdigest()


def _sha(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def source_corpus_fingerprint(chunks: list[dict], parents_by_id: dict[str, dict]) -> str:
    """Huella estable del corpus de Fase 1 (independiente de la vista)."""
    chunk_sig = sorted(
        [c.get("chunk_id"), _sha(c.get("text", "")), _sha(c.get("retrieval_text", ""))]
        for c in chunks
    )
    parent_sig = sorted(
        [
            pid,
            _sha(p.get("text", "")),
            [
                [para.get("order"), para.get("class"), _sha(para.get("text", ""))]
                for para in (p.get("paragraphs") or [])
            ],
        ]
        for pid, p in parents_by_id.items()
    )
    return fingerprint({"chunks": chunk_sig, "parents": parent_sig})


def embedding_inputs_fingerprint(rows: list[dict]) -> str:
    """Huella de los inputs preparados (orden incluido) y su trazabilidad mínima."""
    sig = [
        {
            "embedding_input_id": r["embedding_input_id"],
            "document_id": r["document_id"],
            "block_id": r["block_id"],
            "parent_id": r["parent_id"],
            "formatted_input_sha256": r["formatted_input_sha256"],
            "token_count": r["token_count"],
            "source": r["source"],
            "context_anchor": r.get("context_anchor"),
        }
        for r in rows
    ]
    return fingerprint(sig)


def bundle_identity_fingerprint(
    *,
    document_contract_fingerprint: str,
    source_corpus_fingerprint: str,
    embedding_inputs_fingerprint: str,
) -> str:
    """Huella que identifica un bundle publicado."""
    return fingerprint(
        {
            "document_contract_fingerprint": document_contract_fingerprint,
            "source_corpus_fingerprint": source_corpus_fingerprint,
            "embedding_inputs_fingerprint": embedding_inputs_fingerprint,
        }
    )


def document_contract_fingerprint(
    contract: ModelContract,
    *,
    view: str,
    effective_max_tokens: int,
    overflow_policy: str,
    overlap: int,
) -> str:
    """Huella de la identidad de los embeddings documentales (define el hash del bundle_id)."""
    return fingerprint(
        {
            "model_id": contract.model_id,
            "model_revision": contract.model_revision,
            "tokenizer_id": contract.effective_tokenizer_id,
            "tokenizer_revision": contract.tokenizer_revision,
            "declared_max_tokens": contract.declared_max_tokens,
            "effective_max_tokens": effective_max_tokens,
            "expected_embedding_dimension": contract.expected_embedding_dimension,
            "document_template": contract.document_template,
            "pooling": contract.pooling,
            "normalize_embeddings": contract.normalize_embeddings,
            "trust_remote_code": contract.trust_remote_code,
            "view": view,
            "overflow_policy": overflow_policy,
            "overlap_tokens": overlap,
        }
    )
