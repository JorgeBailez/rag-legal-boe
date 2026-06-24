"""Pooling de candidatos para anotar el gold de relevancia (Fase D). Carga pesada → servidor.

Para cada pregunta del dataset de evaluación, recupera los top-k parents candidatos desde VARIOS
bundles densos publicados (y sus perfiles de query) y los reúne en un *pool* por pregunta (método de
TREC, anti-sesgo). Escribe, por split: un pool máquina (`<split>.jsonl`) y un *worksheet* legible
(`<split>.md`) con cita, procedencia (`found_by`) y el texto vigente del parent con párrafos
numerados, para que el anotador gradúe relevancia (0/1/2) y elija `evidence.paragraph_orders`.

NO valida ni escribe gold: solo produce material para anotar (no toca `judgments.jsonl`). La lógica
pura vive en `src/evaluation/candidate_pool.py` (testeable offline). Requiere los bundles
(`data/indexes/dense/`) y el corpus procesado (`data/processed/`); no corre en CI.

Uso:
    uv run python scripts/build_eval_candidates.py --split all --pool-depth 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings.corpus_loader import load_processed_corpus  # noqa: E402
from src.embeddings.encoder import set_cpu_threads  # noqa: E402
from src.embeddings.model_registry import effective_query_profile_ids  # noqa: E402
from src.evaluation.candidate_pool import (  # noqa: E402
    build_pool,
    pool_to_jsonl_records,
    render_worksheet,
)
from src.evaluation.dataset import (  # noqa: E402
    DATASET_DIR,
    JUDGMENTS_FILE,
    QUESTIONS_FILE,
    load_jsonl,
)
from src.retrieval.dense_retriever import DenseRetriever  # noqa: E402
from src.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from src.retrieval.lexical_retriever import LexicalRetriever  # noqa: E402
from src.retrieval.text_analysis import SpanishAnalyzer  # noqa: E402

SPLITS = ("development", "test", "out_of_corpus")


def _resolve_bundles(args: argparse.Namespace) -> list[Path]:
    if args.bundle:
        return [Path(b) for b in args.bundle]
    root = Path(args.bundles_root)
    return sorted(p for p in root.glob("*") if (p / "manifest.json").is_file())


def _judged_by_qid(judgments: list[dict]) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for j in judgments:
        out.setdefault(j["query_id"], {})[j["parent_id"]] = j
    return out


def _collect_systems(
    bundle_dirs: list[Path], target_questions: list[dict], corpus: dict, args: argparse.Namespace
) -> list[dict]:
    """Recupera top-k por (bundle, perfil, pregunta) y devuelve la lista de 'sistemas' del pool."""
    systems: list[dict] = []
    for bundle_dir in bundle_dirs:
        retriever = DenseRetriever.from_bundle(
            bundle_dir, corpus=corpus, batch_size=args.batch_size
        )
        bundle_id = retriever.bundle_id
        try:
            profile_ids = effective_query_profile_ids(retriever.contract, args.query_profile_id)
        except (KeyError, ValueError) as exc:
            print(f"  {bundle_id}: perfil inválido ({exc}); se omite.", file=sys.stderr)
            continue
        for profile_id in profile_ids:
            print(f"  [{bundle_id}] perfil={profile_id}: recuperando {len(target_questions)}…")
            hits_by_qid: dict[str, list[dict]] = {}
            for q in target_questions:
                hits = retriever.retrieve(
                    q["query"], query_profile_id=profile_id, top_k=args.pool_depth
                )
                hits_by_qid[q["query_id"]] = [
                    {"parent_id": h.parent_id, "rank": h.rank, "score": h.score} for h in hits
                ]
            systems.append(
                {"bundle_id": bundle_id, "query_profile_id": profile_id, "hits_by_qid": hits_by_qid}
            )
    return systems


def _lexical_hybrid_systems(
    args: argparse.Namespace,
    corpus: dict,
    target_questions: list[dict],
    bundle_dirs: list[Path],
) -> list[dict]:
    """Añade BM25 y/o híbrido RRF como sistemas del pool → de-sesga el gold (pooleado dense-only).

    Es el paso anti-sesgo del flagship: sin esto, los parents que solo BM25 recupera (p. ej. el
    artículo exacto de las preguntas `directa_articulo`) nunca se juzgan y se contarían como rel=0.
    """
    if not (args.with_bm25 or args.with_hybrid):
        return []
    bdir = Path(args.lexical_bundle) if args.lexical_bundle else bundle_dirs[0]
    lexical = LexicalRetriever.from_bundle(
        bdir, corpus=corpus, analyzer=SpanishAnalyzer(), heading_boost=args.bm25_heading_boost
    )

    def _hits(retriever: object, profile_id: str | None) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for q in target_questions:
            hits = retriever.retrieve(
                q["query"], query_profile_id=profile_id, top_k=args.pool_depth
            )
            out[q["query_id"]] = [
                {"parent_id": h.parent_id, "rank": h.rank, "score": h.score} for h in hits
            ]
        return out

    systems: list[dict] = []
    if args.with_bm25:
        print(f"  [bm25] heading_boost={args.bm25_heading_boost}: {len(target_questions)} q")
        systems.append(
            {
                "bundle_id": "bm25",
                "query_profile_id": "lexical",
                "hits_by_qid": _hits(lexical, None),
            }
        )
    if args.with_hybrid:
        dense = DenseRetriever.from_bundle(bdir, corpus=corpus, batch_size=args.batch_size)
        prof = effective_query_profile_ids(dense.contract, args.query_profile_id)[0]
        hybrid = HybridRetriever(
            dense=dense,
            lexical=lexical,
            fusion="rrf",
            rrf_k=args.rrf_k,
            candidates=max(100, args.pool_depth),
        )
        print(f"  [hybrid_rrf] rrf_k={args.rrf_k} perfil={prof}: {len(target_questions)} q")
        systems.append(
            {
                "bundle_id": "hybrid_rrf",
                "query_profile_id": prof,
                "hits_by_qid": _hits(hybrid, prof),
            }
        )
    return systems


def _write_split(
    split: str,
    systems: list[dict],
    questions: list[dict],
    judged_by_qid: dict[str, dict[str, dict]],
    corpus: dict,
    args: argparse.Namespace,
) -> None:
    questions_by_id = {q["query_id"]: q for q in questions}
    qids = [q["query_id"] for q in questions if q["split"] == split]
    if not qids:
        return
    pool = build_pool(systems, judged_by_qid, qids)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = pool_to_jsonl_records(pool, questions_by_id)
    (out_dir / f"{split}.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8"
    )
    md = render_worksheet(
        pool,
        corpus["parents_by_id"],
        questions_by_id,
        worksheet_top=args.worksheet_top,
        max_chars=args.max_paragraph_chars,
    )
    (out_dir / f"{split}.md").write_text(md, encoding="utf-8")

    n_cand = sum(len(p["candidates"]) for p in pool.values())
    n_not_pooled = sum(len(p["judged_not_pooled"]) for p in pool.values())
    print(
        f"{split}: {len(qids)} preguntas · {n_cand} candidatos · "
        f"{n_not_pooled} juzgados-no-pooled → {out_dir / f'{split}.{{jsonl,md}}'}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Pooling de candidatos para anotar el gold.")
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--bundles-root", default="data/indexes/dense")
    parser.add_argument("--bundle", action="append", help="bundle concreto (repetible).")
    parser.add_argument("--query-profile-id", action="append", help="perfil(es) de query.")
    parser.add_argument("--split", default="all", choices=[*SPLITS, "all"])
    parser.add_argument("--pool-depth", type=int, default=20, help="top-k por (bundle, perfil).")
    parser.add_argument("--worksheet-top", type=int, default=10, help="candidatos volcados al .md.")
    parser.add_argument("--output-dir", default=None, help="default <dataset-dir>/_candidates.")
    parser.add_argument("--max-paragraph-chars", type=int, default=0, help="recorte de párrafo.")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--with-bm25", action="store_true", help="añade BM25 como sistema del pool."
    )
    parser.add_argument("--with-hybrid", action="store_true", help="añade híbrido RRF al pool.")
    parser.add_argument(
        "--lexical-bundle", default=None, help="bundle para BM25/híbrido (default: el primero)."
    )
    parser.add_argument("--bm25-heading-boost", type=int, default=0)
    parser.add_argument("--rrf-k", type=int, default=60)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = str(Path(args.dataset_dir) / "_candidates")

    corpus = load_processed_corpus()
    if not corpus["parents_by_id"]:
        print("Corpus procesado vacío (ejecuta scripts/process_mvp_corpus.py).", file=sys.stderr)
        return 1
    questions = load_jsonl(Path(args.dataset_dir) / QUESTIONS_FILE)
    judgments = load_jsonl(Path(args.dataset_dir) / JUDGMENTS_FILE)
    if not questions:
        print("No hay questions.jsonl en el dataset.", file=sys.stderr)
        return 1

    bundle_dirs = _resolve_bundles(args)
    if not bundle_dirs:
        print("No hay bundles que consultar (genera alguno o pasa --bundle).", file=sys.stderr)
        return 1

    target_splits = list(SPLITS) if args.split == "all" else [args.split]
    target_questions = [q for q in questions if q["split"] in target_splits]
    judged_by_qid = _judged_by_qid(judgments)

    set_cpu_threads(args.threads)
    print(f"Bundles: {len(bundle_dirs)} | preguntas objetivo: {len(target_questions)} | "
          f"pool-depth={args.pool_depth}")
    systems = _collect_systems(bundle_dirs, target_questions, corpus, args)
    systems += _lexical_hybrid_systems(args, corpus, target_questions, bundle_dirs)
    if not systems:
        print("Ningún sistema produjo candidatos (revisa perfiles/bundles).", file=sys.stderr)
        return 1

    for split in target_splits:
        _write_split(split, systems, questions, judged_by_qid, corpus, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
