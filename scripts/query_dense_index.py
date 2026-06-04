"""Consulta manual de un bundle de índice denso (búsqueda exacta + cita + contexto K_ONLY).

Uso habitual (limpio):
    uv run python scripts/query_dense_index.py \
      --bundle data/indexes/dense/<bundle_id> \
      --query "¿Cuánto tiempo tiene la Administración para responder a mi solicitud?"

Carga el bundle, construye el encoder del modelo registrado en el manifest, codifica la query,
recupera el top-k por producto escalar y muestra cita + contexto. Requiere los pesos del modelo
(se descargan la primera vez) y el corpus procesado de Fase 1 para resolver citas/contexto.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings.corpus_loader import load_processed_corpus  # noqa: E402
from src.embeddings.encoder import DenseEncoder  # noqa: E402
from src.embeddings.model_registry import (  # noqa: E402
    assert_bundle_compatible,
    get_contract,
    query_profile_metadata,
)
from src.indexing.vector_index import ExactDenseIndex, build_filter_mask  # noqa: E402


def _build_filters(args: argparse.Namespace) -> dict:
    filters: dict = {}
    if args.rank_code:
        filters["rank_code"] = args.rank_code
    if args.scope_code:
        filters["scope_code"] = args.scope_code
    if args.semantic_role:
        filters["semantic_role"] = args.semantic_role
    if args.subject_code:
        filters["subject_codes"] = args.subject_code
    for flag in ("annex", "table", "image", "without_content"):
        if getattr(args, flag):
            filters[flag] = True
    return filters


def _resolve(hit: dict, corpus: dict) -> tuple[str, dict]:
    """Devuelve (texto K_ONLY, citation) de un hit por join al chunk o al parent."""
    source = hit["source"]
    chunk_id = source.get("chunk_id")
    if source.get("kind") == "derived_text" and source.get("text") is not None:
        if chunk_id:
            chunk = {c["chunk_id"]: c for c in corpus["chunks"]}.get(chunk_id, {})
            return source["text"], chunk.get("citation", {})
        parent = corpus["parents_by_id"].get(hit["parent_id"], {})
        return source["text"], parent.get("citation", {})
    if chunk_id:
        chunk = {c["chunk_id"]: c for c in corpus["chunks"]}.get(chunk_id, {})
        return chunk.get("text", ""), chunk.get("citation", {})
    parent = corpus["parents_by_id"].get(hit["parent_id"], {})
    return source.get("text", parent.get("text", "")), parent.get("citation", {})


def main() -> int:
    parser = argparse.ArgumentParser(description="Consulta un bundle de índice denso.")
    parser.add_argument("--bundle", required=True, help="ruta al directorio del bundle publicado.")
    parser.add_argument("--query", required=True, help="pregunta en lenguaje natural.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--query-profile-id",
        default=None,
        help="perfil de query reproducible (opción avanzada).",
    )
    parser.add_argument(
        "--context-strategy",
        default="K_ONLY",
        choices=["K_ONLY"],
        help="estrategia de contexto (K_ONLY; expansión en el benchmark).",
    )
    parser.add_argument("--rank-code")
    parser.add_argument("--scope-code")
    parser.add_argument("--semantic-role")
    parser.add_argument("--subject-code", action="append", help="repetible; materia (código).")
    parser.add_argument("--annex", action="store_true")
    parser.add_argument("--table", action="store_true")
    parser.add_argument("--image", action="store_true")
    parser.add_argument("--without-content", action="store_true")
    args = parser.parse_args()

    corpus = load_processed_corpus()
    index = ExactDenseIndex.from_bundle(args.bundle, corpus=corpus)
    contract = get_contract(index.manifest["bundle"]["model_alias"])
    assert_bundle_compatible(contract, index.manifest)

    encoder = DenseEncoder(contract)
    q = encoder.encode_queries(
        [args.query],
        query_profile_id=args.query_profile_id,
        show_progress=False,
    )[0]

    filters = _build_filters(args)
    mask = build_filter_mask(index.rows, corpus, filters) if filters else None
    hits = index.search(q, k=args.top_k, mask=mask)

    print(f"bundle: {index.manifest['bundle']['bundle_id']}")
    qp = query_profile_metadata(contract, args.query_profile_id)
    print(f"query_profile: {qp['query_profile_id']} ({qp['query_profile_fingerprint'][:12]})")
    print(f"query : {args.query}")
    if filters:
        print(f"filtros: {filters}")
    print(f"top-{args.top_k} (estrategia de contexto: {args.context_strategy}):\n")
    for hit in hits:
        text, citation = _resolve(hit, corpus)
        snippet = (text[:280] + " …") if len(text) > 280 else text
        label = citation.get("label", hit["parent_id"])
        print(f"#{hit['rank']}  score={hit['score']:.4f}  {label}")
        print(f"    {citation.get('url', '')}")
        print(f"    {snippet}\n")
    if not hits:
        print("Sin resultados (revisa filtros o corpus).")
    print(
        "Aviso: texto consolidado de carácter informativo, sin valor jurídico oficial. "
        "Remítase a la publicación oficial en el BOE."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
