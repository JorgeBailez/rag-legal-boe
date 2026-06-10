"""Dataset de evaluación del sistema RAG: carga, contratos estrictos y reglas (Gate C).

Estructura simple y revisable a mano (tres ficheros JSONL):
- `questions.jsonl`: una pregunta por línea (identidad, split, familia, estilo, dificultad,
  modo de fallo, alcance, procedencia, estado).
- `judgments.jsonl`: un juicio de retrieval (query_id, parent_id, relevance 0/1/2, evidencia, cita).
- `answer_keys.jsonl`: gold de generación por pregunta (respuesta de referencia, hechos clave,
  citas esperadas, respondible o no). Modelo-agnóstico.

Semántica de relevancia: 2 = parent central/suficiente · 1 = apoyo/matiz · 0 = revisado y
descartado · ausencia = todavía no juzgado. La revisión jurídica humana es posterior: solo
`review_status` ∈ {reviewed, final} habilita los mínimos de Gate C; los borradores pueden estar
incompletos sin ser un error (ver `validate_dataset`).

Lógica pura (sin red ni disco salvo `load_jsonl`), testeable con datos sintéticos.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

DATASET_DIR = Path("data/evaluation/dense_retrieval_v1")
QUESTIONS_FILE = "questions.jsonl"
JUDGMENTS_FILE = "judgments.jsonl"
ANSWER_KEYS_FILE = "answer_keys.jsonl"
SPLITS = ("development", "test", "out_of_corpus")
REVIEWED_STATUSES = ("reviewed", "final")
GATE_C_LEVELS = {
    "checkpoint": {"development": 40, "test": 20, "out_of_corpus": 10},
    "formal": {"development": 40, "test": 80, "out_of_corpus": 20},
}

# Vocabulario recomendado (no contractual; fuera de la lista solo genera aviso, no error).
QUERY_STYLES = (
    "directa_articulo",
    "ciudadana",
    "conceptual",
    "procedimental",
    "lexica",
    "comparativa",
    "sin_respuesta",
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvalQuestion(_Strict):
    query_id: str
    query: str
    split: Literal["development", "test", "out_of_corpus"]
    issue_family_id: str
    query_style: str
    answer_scope: Literal["single_parent", "multi_parent", "none"]
    difficulty: Literal["facil", "media", "dificil"] = "media"
    failure_mode: str | None = None
    provenance: Literal["auto_draft", "human_authored"] = "auto_draft"
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


class EvalAnswerKey(_Strict):
    """Gold de generación de una pregunta: respuesta de referencia, hechos clave y citas esperadas.

    Modelo-agnóstico (no depende del LLM generador): es la verdad del corpus contra la que se miden
    fidelidad, corrección, citas y abstención. `answerable=false` ⇒ gold de abstención (sin
    respuesta ni citas esperadas). Las reglas de completitud solo se exigen a `reviewed`/`final`.
    """

    query_id: str
    answerable: bool
    reference_answer: str = ""
    key_facts: list[str] = []
    forbidden_facts: list[str] = []
    expected_citation_parents: list[str] = []
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
    answer_keys: list[dict] | None = None,
    corpus: dict | None = None,
    gate_c_level: str = "formal",
) -> dict:
    """Valida estructura + reglas y calcula la disponibilidad de Gate C (benchmark formal).

    Honestidad: las reglas *estructurales* (contrato, ids únicos, sin fugas dev/test, referencias
    existentes al corpus) se exigen SIEMPRE; las reglas de *completitud* de anotación (evidencia
    para relevance=2, justificación, multi_parent≥2, answer_keys completos) se exigen SOLO a
    `reviewed`/`final`. Un borrador puede estar incompleto sin ser un error.
    """
    if gate_c_level not in GATE_C_LEVELS:
        raise ValueError(f"gate_c_level desconocido: {gate_c_level!r}")
    answer_keys = answer_keys or []
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
    ak_models: list[EvalAnswerKey] = []
    for i, a in enumerate(answer_keys):
        try:
            ak_models.append(EvalAnswerKey.model_validate(a))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"answer_key[{i}]: contrato inválido ({type(exc).__name__})")

    # 2) query_id único.
    ids = [q.query_id for q in q_models]
    if len(set(ids)) != len(ids):
        errors.append("query_id duplicado en questions")
    qid_set = set(ids)
    split_by_qid = {q.query_id: q.split for q in q_models}
    scope_by_qid = {q.query_id: q.answer_scope for q in q_models}
    review_by_qid = {q.query_id: q.review_status for q in q_models}
    parents_by_id = (corpus or {}).get("parents_by_id") or {}
    parent_orders = {
        pid: {p.get("order") for p in parent.get("paragraphs") or []}
        for pid, parent in parents_by_id.items()
    }

    # 3) Preguntas: contenido mínimo, coherencia de scope/split y vocabulario de estilo.
    for q in q_models:
        if not q.query.strip():
            errors.append(f"query vacía: {q.query_id}")
        if not q.issue_family_id.strip():
            errors.append(f"issue_family_id vacío: {q.query_id}")
        if (q.answer_scope == "none") != (q.split == "out_of_corpus"):
            errors.append(
                f"answer_scope='none' debe coincidir con split='out_of_corpus': {q.query_id}"
            )
        if q.query_style not in QUERY_STYLES:
            warnings.append(f"query_style fuera del vocabulario recomendado: {q.query_id}")

    for j in j_models:
        if j.query_id not in qid_set:
            errors.append(f"judgment.query_id sin pregunta: {j.query_id}")

    # 4) Juicios duplicados y referencias al corpus (estructural, siempre).
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

    # 5) Completitud de la evidencia (solo reviewed/final).
    for j in j_models:
        if j.review_status not in REVIEWED_STATUSES:
            continue
        orders = (j.evidence.paragraph_orders if j.evidence else []) or []
        if j.relevance == 2 and not orders:
            errors.append(
                f"evidence obligatoria para relevance=2 revisado: {j.query_id}/{j.parent_id}"
            )
        if j.relevance >= 1 and not ((j.quote or "").strip() or (j.notes or "").strip()):
            errors.append(
                f"justificación obligatoria para relevance>=1 revisado: {j.query_id}/{j.parent_id}"
            )

    # 6) Fuga de familia entre development y test (estructural, siempre).
    fam_split: dict[str, set[str]] = {}
    for q in q_models:
        fam_split.setdefault(q.issue_family_id, set()).add(q.split)
    for fam, splits in fam_split.items():
        if "development" in splits and "test" in splits:
            errors.append(f"issue_family_id en development y test a la vez: {fam}")

    # 7) out_of_corpus no puede tener juicios relevantes (estructural, siempre).
    for j in j_models:
        if split_by_qid.get(j.query_id) == "out_of_corpus" and j.relevance >= 1:
            errors.append(f"juicio relevante en out_of_corpus: {j.query_id}/{j.parent_id}")

    # 8) multi_parent exige ≥2 parents relevantes (solo en preguntas revisadas).
    relevant_by_qid: dict[str, set[str]] = {}
    for j in j_models:
        if j.relevance >= 1:
            relevant_by_qid.setdefault(j.query_id, set()).add(j.parent_id)
    for qid, scope in scope_by_qid.items():
        reviewed_q = review_by_qid.get(qid) in REVIEWED_STATUSES
        if scope == "multi_parent" and reviewed_q and len(relevant_by_qid.get(qid, set())) < 2:
            errors.append(f"multi_parent con menos de dos parents relevantes: {qid}")

    # 9) answer_keys (gold de generación): estructura siempre, completitud solo reviewed/final.
    ak_qids = [a.query_id for a in ak_models]
    if len(set(ak_qids)) != len(ak_qids):
        errors.append("query_id duplicado en answer_keys")
    for a in ak_models:
        if a.query_id not in qid_set:
            errors.append(f"answer_key.query_id sin pregunta: {a.query_id}")
            continue
        is_ooc = split_by_qid.get(a.query_id) == "out_of_corpus"
        if is_ooc and a.answerable:
            errors.append(f"answer_key answerable=true en out_of_corpus: {a.query_id}")
        if not a.answerable and (a.reference_answer.strip() or a.expected_citation_parents):
            errors.append(f"answer_key no answerable con respuesta/citas esperadas: {a.query_id}")
        if parents_by_id:
            missing_corpus = [p for p in a.expected_citation_parents if p not in parents_by_id]
            if missing_corpus:
                errors.append(
                    f"expected_citation_parents inexistentes en corpus para {a.query_id}: "
                    f"{missing_corpus}"
                )
        if a.review_status in REVIEWED_STATUSES and a.answerable:
            if not a.reference_answer.strip():
                errors.append(f"reference_answer obligatoria (answerable revisado): {a.query_id}")
            if not a.expected_citation_parents:
                errors.append(
                    f"expected_citation_parents obligatorio (answerable revisado): {a.query_id}"
                )
            # Coherencia entre capas: las citas esperadas deben ser parents juzgados relevantes.
            rel = relevant_by_qid.get(a.query_id, set())
            not_relevant = [p for p in a.expected_citation_parents if rel and p not in rel]
            if not_relevant:
                warnings.append(
                    f"expected_citation_parents sin juicio relevante para {a.query_id}: "
                    f"{not_relevant}"
                )

    # Conteos.
    by_split = {s: sum(1 for q in q_models if q.split == s) for s in SPLITS}

    # Gate C (retrieval): solo si hay anotación de relevancia revisada suficiente y sin fugas.
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

    # Gate C (generación): además, answer_keys revisados que cubren las preguntas revisadas.
    reviewed_ak_qids = {a.query_id for a in ak_models if a.review_status in REVIEWED_STATUSES}
    gen_ready_by_split = {
        split: sum(
            1
            for q in q_models
            if q.split == split
            and q.review_status in REVIEWED_STATUSES
            and q.query_id in reviewed_ak_qids
        )
        for split in SPLITS
    }
    gen_reasons: list[str] = list(reasons)
    if not ak_models:
        gen_reasons.append("sin answer_keys (gold de generación ausente)")
    for split, minimum in minimums.items():
        if gen_ready_by_split[split] < minimum:
            gen_reasons.append(
                f"split {split}: {gen_ready_by_split[split]} answer_keys revisadas < {minimum}"
            )
    generation_ready = not gen_reasons

    return {
        "n_questions": len(q_models),
        "n_judgments": len(j_models),
        "n_answer_keys": len(ak_models),
        "by_split": by_split,
        "n_reviewed_relevant_queries": len(reviewed_rel_qids),
        "n_reviewed_answer_keys": len(reviewed_ak_qids),
        "reviewed_ready_by_split": reviewed_ready_by_split,
        "errors": errors,
        "warnings": warnings,
        "gate_c": {
            "ready": gate_c_ready,
            "level": gate_c_level,
            "minimums": minimums,
            "reasons": reasons,
            "generation_ready": generation_ready,
            "generation_reasons": gen_reasons,
        },
    }


def load_and_validate(
    dataset_dir: Path = DATASET_DIR,
    *,
    corpus: dict | None = None,
    gate_c_level: str = "formal",
) -> dict:
    """Carga questions/judgments/answer_keys del directorio y valida (CLI y notebooks)."""
    dataset_dir = Path(dataset_dir)
    questions = load_jsonl(dataset_dir / QUESTIONS_FILE)
    judgments = load_jsonl(dataset_dir / JUDGMENTS_FILE)
    answer_keys = load_jsonl(dataset_dir / ANSWER_KEYS_FILE)
    report = validate_dataset(
        questions,
        judgments,
        answer_keys=answer_keys,
        corpus=corpus,
        gate_c_level=gate_c_level,
    )
    report["dataset_dir"] = str(dataset_dir)
    return report
