"""Auditoría de *grounding* del dataset de evaluación contra el corpus procesado REAL.

Verifica, para CADA pregunta, que todo lo anotado está respaldado por el texto vigente de las normas
(no por memoria ni por los borradores antiguos):

- cada `parent_id` juzgado y cada `expected_citation_parent` EXISTE en el corpus, es **recuperable**
  (tiene chunks) y su `content_status` es coherente;
- cada `key_fact` de un answer_key aparece **literalmente** (normalizado: minúsculas, sin tildes) en
  el texto vigente de los parents citados → prueba de que el hecho está respaldado por la norma;
- coherencia answer_keys ↔ judgments (las citas esperadas son parents juzgados relevantes);
- las trampas temporales (`failure_mode = temporal_without_content`) apuntan a un bloque
  «(Sin contenido)».

Solo lectura. Requiere el corpus procesado en `data/processed/` (no versionado; se genera con
`scripts/process_mvp_corpus.py`). Es el chequeo que permite confiar en el dataset sin revisarlo a
mano una a una: lo verde está respaldado por el texto; lo marcado [FLAG] es lo único a corregir.

Uso:
    uv run python scripts/audit_eval_dataset.py                 # informe + flags
    uv run python scripts/audit_eval_dataset.py --show-text     # añade el texto citado por pregunta
    uv run python scripts/audit_eval_dataset.py --strict        # exit≠0 si hay flags
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings.corpus_loader import load_processed_corpus  # noqa: E402
from src.evaluation.dataset import (  # noqa: E402
    ANSWER_KEYS_FILE,
    DATASET_DIR,
    JUDGMENTS_FILE,
    QUESTIONS_FILE,
    load_jsonl,
)
from src.evaluation.generation_metrics import normalize_text  # noqa: E402


def _parent_text(parent: dict) -> str:
    """Texto vigente del parent (campo `text`; si está vacío, se reconstruye de los párrafos)."""
    text = (parent.get("text") or "").strip()
    if text:
        return text
    return " ".join(p.get("text", "") for p in parent.get("paragraphs") or [])


def main() -> int:  # noqa: C901 - auditoría lineal con varias comprobaciones
    parser = argparse.ArgumentParser(description="Auditoría de grounding del dataset.")
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--show-text", action="store_true", help="imprime el texto citado.")
    parser.add_argument("--max-text", type=int, default=240, help="caracteres de snippet.")
    parser.add_argument("--strict", action="store_true", help="exit≠0 si hay flags.")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    corpus = load_processed_corpus()
    parents = corpus["parents_by_id"]
    if not parents:
        print("ERROR: corpus procesado vacío. Genera data/processed/ con process_mvp_corpus.py.")
        return 2
    chunks_by_parent: dict[str, int] = {}
    for c in corpus["chunks"]:
        chunks_by_parent[c["parent_id"]] = chunks_by_parent.get(c["parent_id"], 0) + 1

    questions = {q["query_id"]: q for q in load_jsonl(dataset_dir / QUESTIONS_FILE)}
    judgments = load_jsonl(dataset_dir / JUDGMENTS_FILE)
    answer_keys = {a["query_id"]: a for a in load_jsonl(dataset_dir / ANSWER_KEYS_FILE)}

    relevant_by_qid: dict[str, set[str]] = {}
    for j in judgments:
        if j.get("relevance", 0) >= 1:
            relevant_by_qid.setdefault(j["query_id"], set()).add(j["parent_id"])

    flags: list[str] = []
    checked_parents = 0
    checked_facts = 0

    # 1) Todos los parents juzgados existen.
    for j in judgments:
        pid = j["parent_id"]
        checked_parents += 1
        if pid not in parents:
            flags.append(f"{j['query_id']}: judgment.parent_id inexistente en corpus: {pid}")
        elif j.get("relevance", 0) >= 1 and pid not in chunks_by_parent:
            flags.append(f"{j['query_id']}: parent relevante sin chunks (no recuperable): {pid}")

    # 2) Por pregunta: citas esperadas + key_facts respaldados por el texto.
    for qid, q in sorted(questions.items()):
        ak = answer_keys.get(qid)
        answerable = (
            ak.get("answerable", q["split"] != "out_of_corpus")
            if ak
            else (q["split"] != "out_of_corpus")
        )
        line_flags: list[str] = []
        expected = (ak or {}).get("expected_citation_parents", []) if ak else []
        cited_texts: list[str] = []
        for pid in expected:
            checked_parents += 1
            if pid not in parents:
                line_flags.append(f"cita esperada inexistente: {pid}")
                continue
            cited_texts.append(_parent_text(parents[pid]))
            if pid not in chunks_by_parent:
                line_flags.append(f"cita esperada no recuperable (sin chunks): {pid}")
            if parents[pid].get("is_without_content"):
                line_flags.append(f"cita esperada (Sin contenido) vigente: {pid}")
            if pid not in relevant_by_qid.get(qid, set()):
                line_flags.append(f"cita esperada sin juicio relevante: {pid}")

        haystack = normalize_text(" \n ".join(cited_texts))
        for kf in (ak or {}).get("key_facts", []) if ak else []:
            checked_facts += 1
            if normalize_text(kf) not in haystack:
                line_flags.append(f"key_fact NO aparece en el texto citado: {kf!r}")

        # Trampa temporal: el parent juzgado debe ser (Sin contenido).
        if (q.get("failure_mode") or "") == "temporal_without_content":
            judged = relevant_by_qid.get(qid, set()) | {
                j["parent_id"] for j in judgments if j["query_id"] == qid
            }
            if not any(parents.get(p, {}).get("is_without_content") for p in judged):
                line_flags.append("trampa temporal sin bloque (Sin contenido) entre los juzgados")

        status = "FLAG" if line_flags else "ok"
        mark = "answerable" if answerable else "abstención"
        print(f"[{status}] {qid} ({mark}) {q.get('failure_mode') or q['query_style']}")
        for lf in line_flags:
            print(f"        - {lf}")
            flags.append(f"{qid}: {lf}")
        if args.show_text and cited_texts:
            for pid, txt in zip(expected, cited_texts, strict=False):
                label = parents.get(pid, {}).get("citation", {}).get("label", pid)
                snippet = " ".join(txt.split())[: args.max_text]
                print(f"        · {label} [{pid}]: {snippet}…")

    print("\n===== RESUMEN =====")
    print(
        f"preguntas: {len(questions)} | judgments: {len(judgments)} "
        f"| answer_keys: {len(answer_keys)}"
    )
    print(f"parents comprobados: {checked_parents} | key_facts comprobados: {checked_facts}")
    print(f"normas en corpus: {corpus['n_norms']}")
    if flags:
        print(f"\n[FLAGS] {len(flags)} incidencias a revisar:")
        for f in flags:
            print(f"  - {f}")
    else:
        print("\nSin flags: todo lo anotado está respaldado por el texto del corpus.")

    if args.strict and flags:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
