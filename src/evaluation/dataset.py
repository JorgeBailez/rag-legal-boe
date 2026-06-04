"""Dataset de evaluación de retrieval denso: carga, contratos estrictos y reglas (Gate C).

Estructura simple y revisable a mano (dos ficheros JSONL):
- `questions.jsonl`: una pregunta por línea (identidad, split, familia, estilo, alcance, estado).
- `judgments.jsonl`: un juicio (query_id, parent_id, relevance 0/1/2, evidencia, cita, estado).

Semántica de relevancia: 2 = parent central/suficiente · 1 = apoyo/matiz · 0 = revisado y
descartado · ausencia = todavía no juzgado. La revisión jurídica humana es posterior: el scaffold
trae solo ejemplos marcados (`review_status="example"`), que **no** habilitan el benchmark formal.

Lógica pura (sin red ni disco salvo `load_jsonl`), testeable con datos sintéticos.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

DATASET_DIR = Path("data/evaluation/dense_retrieval_v1")
SPLITS = ("development", "test", "out_of_corpus")
REVIEWED_STATUSES = ("reviewed", "final")
GATE_C_LEVELS = {
    "checkpoint": {"development": 40, "test": 20, "out_of_corpus": 10},
    "formal": {"development": 40, "test": 80, "out_of_corpus": 20},
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvalQuestion(_Strict):
    query_id: str
    query: str
    split: Literal["development", "test", "out_of_corpus"]
    issue_family_id: str
    query_style: str
    answer_scope: Literal["single_parent", "multi_parent", "none"]
    review_status: Literal["example", "draft", "reviewed", "final"] = "draft"
    notes: str | None = None


class Evidence(_Strict):
    paragraph_orders: list[int] = []


class EvalJudgment(_Strict):
    query_id: str
    parent_id: str
    relevance: Literal[0, 1, 2]
    evidence: Evidence | None = None
    quote: str | None = None
    review_status: Literal["example", "draft", "reviewed", "final"] = "draft"
    notes: str | None = None


def load_jsonl(path: Path) -> list[dict]:
    """Carga un fichero JSONL en una lista de dicts (ignora líneas en blanco)."""
    path = Path(path)
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def validate_dataset(
    questions: list[dict],
    judgments: list[dict],
    *,
    corpus: dict | None = None,
    gate_c_level: str = "formal",
) -> dict:
    """Valida estructura + reglas y calcula la disponibilidad de Gate C (benchmark formal)."""
    if gate_c_level not in GATE_C_LEVELS:
        raise ValueError(f"gate_c_level desconocido: {gate_c_level!r}")
    errors: list[str] = []
    warnings: list[str] = []

    # 1) Contratos por registro.
    q_models: list[EvalQuestion] = []
    for i, q in enumerate(questions):
        try:
            q_models.append(EvalQuestion.model_validate(q))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"question[{i}]: contrato inválido ({type(exc).__name__})")
    j_models: list[EvalJudgment] = []
    for i, j in enumerate(judgments):
        try:
            j_models.append(EvalJudgment.model_validate(j))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"judgment[{i}]: contrato inválido ({type(exc).__name__})")

    # 2) query_id único.
    ids = [q.query_id for q in q_models]
    if len(set(ids)) != len(ids):
        errors.append("query_id duplicado en questions")
    qid_set = set(ids)
    split_by_qid = {q.query_id: q.split for q in q_models}
    scope_by_qid = {q.query_id: q.answer_scope for q in q_models}
    parents_by_id = (corpus or {}).get("parents_by_id") or {}
    parent_orders = {
        pid: {p.get("order") for p in parent.get("paragraphs") or []}
        for pid, parent in parents_by_id.items()
    }

    # 3) Juicios sin pregunta.
    for q in q_models:
        if not q.query.strip():
            errors.append(f"query vacía: {q.query_id}")
        if not q.issue_family_id.strip():
            errors.append(f"issue_family_id vacío: {q.query_id}")
        if (q.answer_scope == "none") != (q.split == "out_of_corpus"):
            errors.append(
                f"answer_scope='none' debe coincidir con split='out_of_corpus': {q.query_id}"
            )

    for j in j_models:
        if j.query_id not in qid_set:
            errors.append(f"judgment.query_id sin pregunta: {j.query_id}")

    # 4) Juicios duplicados y referencias al corpus.
    judgment_keys = [(j.query_id, j.parent_id) for j in j_models]
    if len(set(judgment_keys)) != len(judgment_keys):
        errors.append("(query_id, parent_id) duplicado en judgments")
    if parents_by_id:
        for j in j_models:
            if j.parent_id not in parents_by_id:
                errors.append(f"judgment.parent_id inexistente en corpus: {j.parent_id}")
                continue
            orders = (j.evidence.paragraph_orders if j.evidence else []) or []
            bad_orders = [o for o in orders if o not in parent_orders[j.parent_id]]
            if bad_orders:
                errors.append(
                    f"paragraph_orders inexistentes para {j.query_id}/{j.parent_id}: {bad_orders}"
                )

    # 5) Reglas de evidencia y justificación.
    for j in j_models:
        orders = (j.evidence.paragraph_orders if j.evidence else []) or []
        if j.relevance == 2 and not orders:
            errors.append(f"evidence obligatoria para relevance=2: {j.query_id}/{j.parent_id}")
        if j.relevance >= 1 and not ((j.quote or "").strip() or (j.notes or "").strip()):
            errors.append(
                f"justificación obligatoria para relevance>=1: {j.query_id}/{j.parent_id}"
            )

    # 6) Fuga de familia entre development y test.
    fam_split: dict[str, set[str]] = {}
    for q in q_models:
        fam_split.setdefault(q.issue_family_id, set()).add(q.split)
    for fam, splits in fam_split.items():
        if "development" in splits and "test" in splits:
            errors.append(f"issue_family_id en development y test a la vez: {fam}")

    # 7) out_of_corpus no puede tener juicios relevantes.
    for j in j_models:
        if split_by_qid.get(j.query_id) == "out_of_corpus" and j.relevance >= 1:
            errors.append(f"juicio relevante en out_of_corpus: {j.query_id}/{j.parent_id}")

    # 8) multi_parent exige al menos dos parents relevantes.
    relevant_by_qid: dict[str, set[str]] = {}
    for j in j_models:
        if j.relevance >= 1:
            relevant_by_qid.setdefault(j.query_id, set()).add(j.parent_id)
    for qid, scope in scope_by_qid.items():
        if scope == "multi_parent" and len(relevant_by_qid.get(qid, set())) < 2:
            errors.append(f"multi_parent con menos de dos parents relevantes: {qid}")

    # Conteos.
    by_split = {s: sum(1 for q in q_models if q.split == s) for s in SPLITS}

    # Gate C: benchmark formal solo si hay anotación revisada suficiente y sin fugas.
    reviewed_rel_qids = {
        j.query_id for j in j_models if j.relevance >= 1 and j.review_status in REVIEWED_STATUSES
    }
    reviewed_ready_by_split: dict[str, int] = {}
    for split in SPLITS:
        if split == "out_of_corpus":
            reviewed_ready_by_split[split] = sum(
                1 for q in q_models if q.split == split and q.review_status in REVIEWED_STATUSES
            )
        else:
            reviewed_ready_by_split[split] = sum(
                1
                for q in q_models
                if q.split == split
                and q.review_status in REVIEWED_STATUSES
                and q.query_id in reviewed_rel_qids
            )
    reasons: list[str] = []
    if errors:
        reasons.append("hay errores estructurales")
    minimums = GATE_C_LEVELS[gate_c_level]
    for split, minimum in minimums.items():
        count = reviewed_ready_by_split[split]
        if count < minimum:
            reasons.append(
                f"split {split}: {count} revisadas listas < mínimo {minimum} ({gate_c_level})"
            )
    gate_c_ready = not reasons

    return {
        "n_questions": len(q_models),
        "n_judgments": len(j_models),
        "by_split": by_split,
        "n_reviewed_relevant_queries": len(reviewed_rel_qids),
        "reviewed_ready_by_split": reviewed_ready_by_split,
        "errors": errors,
        "warnings": warnings,
        "gate_c": {
            "ready": gate_c_ready,
            "level": gate_c_level,
            "minimums": minimums,
            "reasons": reasons,
        },
    }


def load_and_validate(
    dataset_dir: Path = DATASET_DIR,
    *,
    corpus: dict | None = None,
    gate_c_level: str = "formal",
) -> dict:
    """Carga questions/judgments del directorio y valida. Útil para el CLI y los notebooks."""
    dataset_dir = Path(dataset_dir)
    questions = load_jsonl(dataset_dir / "questions.jsonl")
    judgments = load_jsonl(dataset_dir / "judgments.jsonl")
    report = validate_dataset(questions, judgments, corpus=corpus, gate_c_level=gate_c_level)
    report["dataset_dir"] = str(dataset_dir)
    return report
