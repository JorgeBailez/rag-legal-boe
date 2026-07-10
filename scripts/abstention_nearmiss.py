"""Abstención (L6) con desglose far-domain vs near-miss — corrida ligera.

Calcula la separabilidad de la señal de recuperación (score top-1) entre preguntas in-corpus
(respondibles) y out_of_corpus (no respondibles), **desglosando** las OOC en dos subconjuntos:

  * far_domain: materia ajena al corpus (negativos fáciles; las 30 OOC originales),
  * near_miss:  misma materia, respuesta ausente (query_id con prefijo `q92nm_`).

El AUC global mezcla ambos y sobreestima la abstención, porque el near-miss es el caso difícil
(cierra el caveat del capítulo: "el AUC probablemente sobreestima"). Para cada split in-corpus
(development, test) se reporta AUC + balanced accuracy contra all / far_domain / near_miss.

A diferencia de `benchmark_dense_models.py`, NO necesita `data/processed/` (no calcula ParentnDCG):
solo el bundle (embeddings + rows) para el score top-1 y el modelo para codificar las consultas. La
codificación de ~120 consultas cortas es barata (CPU basta; GPU opcional).

Uso:
    python scripts/abstention_nearmiss.py \
        --bundle data/indexes/dense/e5-large-instruct__j1__c46c7042f563 \
        --dataset-dir data/evaluation/corpus92_v1 \
        --query-profile-id I1_LEGAL \
        --out data/processed/reports/dense/abstention_nearmiss.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings import model_registry as reg  # noqa: E402
from src.embeddings.encoder import DenseEncoder, set_cpu_threads  # noqa: E402
from src.embeddings.model_registry import assert_bundle_compatible  # noqa: E402
from src.evaluation.metrics import abstention_threshold_analysis  # noqa: E402

# El índice se carga SIN corpus: `load_validated_bundle(corpus=None)` valida la integridad interna
# del bundle (esquema, gates, checksums, Gate B) pero omite las comprobaciones que exigen tener el
# corpus en disco (source_corpus_fingerprint, existencia de parents). La abstención solo necesita el
# score top-1 (embeddings + rows); no necesita arrastrar data/processed/.
from src.indexing.vector_index import ExactDenseIndex  # noqa: E402


def _load_questions(dataset_dir: Path) -> list[dict]:
    path = dataset_dir / "questions.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _top1_scores(
    enc: DenseEncoder, index: ExactDenseIndex, queries: list[str], profile_id: str
) -> list[float]:
    if not queries:
        return []
    vecs = enc.encode_queries(queries, query_profile_id=profile_id, show_progress=True)
    scores: list[float] = []
    for v in vecs:
        h = index.search(v, k=1)
        scores.append(float(h[0]["score"]) if h else 0.0)
    return scores


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--dataset-dir", default="data/evaluation/corpus92_v1", type=Path)
    ap.add_argument("--query-profile-id", default="I1_LEGAL")
    ap.add_argument("--splits", default="development,test", help="splits in-corpus (coma)")
    ap.add_argument("--near-prefix", default="q92nm_", help="prefijo de query_id de los near-miss")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--threads", type=int, default=0, help="hilos CPU (0 = por defecto)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.threads:
        set_cpu_threads(args.threads)

    index = ExactDenseIndex.from_bundle(args.bundle)
    contract = reg.get_contract(index.manifest["bundle"]["model_alias"])
    assert_bundle_compatible(contract, index.manifest)
    enc = DenseEncoder(contract, batch_size=args.batch_size, allow_unpinned_revision=False)
    profile_id = args.query_profile_id

    questions = _load_questions(args.dataset_dir)
    ooc = [q for q in questions if q["split"] == "out_of_corpus"]
    far = [q for q in ooc if not q["query_id"].startswith(args.near_prefix)]
    near = [q for q in ooc if q["query_id"].startswith(args.near_prefix)]
    print(f"OOC: {len(ooc)} (far_domain={len(far)}, near_miss={len(near)})")

    far_scores = _top1_scores(enc, index, [q["query"] for q in far], profile_id)
    near_scores = _top1_scores(enc, index, [q["query"] for q in near], profile_id)
    all_scores = far_scores + near_scores

    result: dict = {
        "kind": "abstention_nearmiss",
        "bundle_id": index.manifest["bundle"]["bundle_id"],
        "query_profile_id": profile_id,
        "n_far_domain": len(far),
        "n_near_miss": len(near),
        "by_split": {},
        "ooc_top1_scores": {
            **{q["query_id"]: s for q, s in zip(far, far_scores, strict=True)},
            **{q["query_id"]: s for q, s in zip(near, near_scores, strict=True)},
        },
    }

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        ans = [q for q in questions if q["split"] == split]
        ans_scores = _top1_scores(enc, index, [q["query"] for q in ans], profile_id)
        entry = {
            "n_answerable": len(ans),
            "all": abstention_threshold_analysis(ans_scores, all_scores),
        }
        if far_scores:
            entry["far_domain"] = abstention_threshold_analysis(ans_scores, far_scores)
        if near_scores:
            entry["near_miss"] = abstention_threshold_analysis(ans_scores, near_scores)
        result["by_split"][split] = entry

    # salida legible
    print(f"\n=== Abstención far vs near — bundle {result['bundle_id']} · perfil {profile_id} ===")
    for split, e in result["by_split"].items():
        print(f"\n[{split}] answerable n={e['n_answerable']}")
        for sub in ("all", "far_domain", "near_miss"):
            if sub in e:
                a = e[sub]
                print(
                    f"  {sub:<11} AUC={a['auc']:.3f}  bal_acc={a['balanced_accuracy']:.3f}  "
                    f"(n_ooc={a['n_unanswerable']})"
                )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[guardado] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
