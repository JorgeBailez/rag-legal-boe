"""Consulta manual de un bundle de índice denso (búsqueda exacta + cita + contexto K_ONLY).

Uso habitual (limpio):
    uv run python scripts/query_dense_index.py \
      --bundle data/indexes/dense/<bundle_id> \
      --query "¿Cuánto tiempo tiene la Administración para responder a mi solicitud?"

Carga el bundle, construye el encoder del modelo registrado en el manifest, codifica la query,
recupera el top-k por producto escalar y muestra cita + contexto. Requiere los pesos del modelo
(se descargan la primera vez) y el corpus procesado de Fase 1 para resolver citas/contexto.

La lógica reutilizable vive en `src.retrieval.dense_retriever` (compartida con la generación de
Fase 3); este script es una capa delgada de presentación.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings.corpus_loader import load_processed_corpus  # noqa: E402
from src.embeddings.model_registry import query_profile_metadata  # noqa: E402
from src.retrieval.dense_retriever import (  # noqa: E402
    DenseRetriever,
    RetrievalFilters,
    resolve_hit_text_and_citation,
)


def _resolve(hit: dict, corpus: dict) -> tuple[str, dict]:
    """Shim de compatibilidad: resuelve (texto, cita) de un hit por join al chunk o al parent."""
    chunks_by_id = {c["chunk_id"]: c for c in corpus.get("chunks", [])}
    return resolve_hit_text_and_citation(
        hit, chunks_by_id=chunks_by_id, parents_by_id=corpus.get("parents_by_id", {})
    )


def _positive_int(value: str) -> int:
    """Tipo argparse: entero estrictamente positivo (falla en el parseo del CLI, no más tarde)."""
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"valor entero inválido: {value!r}") from None
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"debe ser un entero > 0 (recibido {ivalue}).")
    return ivalue


def _non_blank(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError(
            "la pregunta no puede estar vacía ni contener solo espacios."
        )
    return value


def _build_filters(args: argparse.Namespace) -> RetrievalFilters:
    return RetrievalFilters(
        rank_code=args.rank_code,
        scope_code=args.scope_code,
        semantic_role=args.semantic_role,
        subject_codes=list(args.subject_code or []),
        annex=args.annex,
        table=args.table,
        image=args.image,
        without_content=args.without_content,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Consulta un bundle de índice denso.")
    parser.add_argument("--bundle", required=True, help="ruta al directorio del bundle publicado.")
    parser.add_argument(
        "--query", required=True, type=_non_blank, help="pregunta en lenguaje natural."
    )
    parser.add_argument("--top-k", type=_positive_int, default=5)
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
    retriever = DenseRetriever.from_bundle(args.bundle, corpus=corpus)

    filters = _build_filters(args).as_filter_dict()
    hits = retriever.retrieve(
        args.query,
        query_profile_id=args.query_profile_id,
        top_k=args.top_k,
        filters=filters or None,
    )

    print(f"bundle: {retriever.bundle_id}")
    qp = query_profile_metadata(retriever.contract, args.query_profile_id)
    print(f"query_profile: {qp['query_profile_id']} ({qp['query_profile_fingerprint'][:12]})")
    print(f"query : {args.query}")
    if filters:
        print(f"filtros: {filters}")
    print(f"top-{args.top_k} (estrategia de contexto: {args.context_strategy}):\n")
    for hit in hits:
        text = hit.retrieval_text
        snippet = (text[:280] + " …") if len(text) > 280 else text
        print(f"#{hit.rank}  score={hit.score:.4f}  {hit.citation_label}")
        print(f"    {hit.citation_url or ''}")
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
