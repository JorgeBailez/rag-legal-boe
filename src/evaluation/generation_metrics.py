"""Métricas de generación (L3–L6): fidelidad, citas, corrección y abstención.

Funciones **puras y deterministas**, sin red ni LLM:

- Las que NO dependen del juez (key-fact recall, hechos prohibidos, attribution vs gold, abstención)
  operan solo sobre la respuesta y el answer_key del gold.
- Las que dependen del juez (fidelidad L3, corrección L5) reciben los **veredictos ya calculados**
  como tipos primitivos (`list[bool]`, etiquetas), de modo que se testean sin invocar al LLM. La
  llamada al juez vive en `src/evaluation/judge.py`; aquí solo se reducen a números.

Convención de abstención (dos errores NO simétricos): un *false answer* sobre una pregunta no
respondible es el error peligroso en legal; una *over-abstention* sobre una respondible es molesta
pero segura. Se reportan por separado y como balanced accuracy.
"""

from __future__ import annotations

import unicodedata

ABSTENTION_OUTCOMES = ("answered", "correct_abstention", "over_abstention", "false_answer")
CORRECTNESS_LABELS = ("correct", "partial", "incorrect")
_CORRECTNESS_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
_JUDGE_NUMERIC_KEYS = (
    "key_fact_recall",
    "citation_precision",
    "citation_recall",
    "citation_f1",
    "faithfulness",
    "correctness",
)


# --------------------------------------------------------------------------- #
# Normalización de texto (para casar hechos clave robusta a tildes/mayúsculas)
# --------------------------------------------------------------------------- #


def normalize_text(text: str) -> str:
    """Minúsculas, sin tildes y con espacios colapsados (para comparación robusta de hechos)."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(stripped.lower().split())


def _contains(haystack_norm: str, needle: str) -> bool:
    n = normalize_text(needle)
    return bool(n) and n in haystack_norm


# --------------------------------------------------------------------------- #
# L5 — corrección barata por hechos clave (sin juez)
# --------------------------------------------------------------------------- #


def key_fact_recall(answer: str, key_facts: list[str]) -> dict:
    """Fracción de `key_facts` presentes en la respuesta (match normalizado). None si no hay."""
    norm = normalize_text(answer)
    present = [k for k in key_facts if _contains(norm, k)]
    missing = [k for k in key_facts if not _contains(norm, k)]
    recall = (len(present) / len(key_facts)) if key_facts else None
    return {"key_fact_recall": recall, "key_facts_present": present, "key_facts_missing": missing}


def forbidden_fact_hits(answer: str, forbidden_facts: list[str]) -> list[str]:
    """Hechos prohibidos (trampas) presentes en la respuesta → señal de alucinación."""
    norm = normalize_text(answer)
    return [f for f in forbidden_facts if _contains(norm, f)]


# --------------------------------------------------------------------------- #
# L4 — atribución de citas vs gold (sin juez)
# --------------------------------------------------------------------------- #


def citation_attribution(cited_parents: list[str], expected_parents: list[str]) -> dict:
    """Precision/recall/F1 del conjunto de parents citados vs los esperados por el gold.

    Si el gold no fija citas esperadas (p. ej. no respondible), devuelve None (no aplica).
    """
    cited = set(cited_parents)
    expected = set(expected_parents)
    if not expected:
        return {"citation_precision": None, "citation_recall": None, "citation_f1": None}
    tp = len(cited & expected)
    precision = tp / len(cited) if cited else 0.0
    recall = tp / len(expected)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"citation_precision": precision, "citation_recall": recall, "citation_f1": f1}


# --------------------------------------------------------------------------- #
# L6 — abstención (clasificación por query; sin juez)
# --------------------------------------------------------------------------- #


def abstention_outcome(answered: bool, answerable: bool) -> str:
    """Categoría de abstención por pregunta (ver ABSTENTION_OUTCOMES)."""
    if answerable:
        return "answered" if answered else "over_abstention"
    return "false_answer" if answered else "correct_abstention"


def abstention_point(answered: bool, has_generation_metrics: bool) -> str:
    """Dónde se decidió: respondida, abstención determinista pre-LLM o decidida por el LLM."""
    if answered:
        return "answered"
    return "llm_decided" if has_generation_metrics else "pre_llm"


# --------------------------------------------------------------------------- #
# Reducción de veredictos del juez (L3 fidelidad, L5 corrección)
# --------------------------------------------------------------------------- #


def faithfulness_score(claim_supported: list[bool]) -> float | None:
    """Fracción de afirmaciones soportadas por la evidencia. None si no hay afirmaciones."""
    if not claim_supported:
        return None
    return sum(1 for s in claim_supported if s) / len(claim_supported)


def correctness_score(label: str) -> float:
    """Mapea la etiqueta del juez (correct/partial/incorrect) a 1.0/0.5/0.0."""
    if label not in _CORRECTNESS_SCORE:
        raise ValueError(f"etiqueta de corrección desconocida: {label!r}")
    return _CORRECTNESS_SCORE[label]


# --------------------------------------------------------------------------- #
# Métricas por query + agregación
# --------------------------------------------------------------------------- #


def compute_query_generation_metrics(
    *,
    answered: bool,
    answer_text: str,
    has_generation_metrics: bool,
    cited_parents: list[str],
    answerable: bool,
    key_facts: list[str],
    forbidden_facts: list[str],
    expected_citation_parents: list[str],
    faithfulness_claims: list[bool] | None = None,
    correctness_label: str | None = None,
) -> dict:
    """Métricas de UNA pregunta. Solo evalúa contenido si respondió y era respondible."""
    out: dict = {
        "answered": answered,
        "answerable": answerable,
        "abstention_outcome": abstention_outcome(answered, answerable),
        "abstention_point": abstention_point(answered, has_generation_metrics),
    }
    if answerable and answered:
        out.update(key_fact_recall(answer_text, key_facts))
        hits = forbidden_fact_hits(answer_text, forbidden_facts)
        out["forbidden_fact_hits"] = hits
        out["hallucinated_forbidden"] = bool(hits)
        out.update(citation_attribution(cited_parents, expected_citation_parents))
        if faithfulness_claims is not None:
            out["faithfulness"] = faithfulness_score(faithfulness_claims)
        if correctness_label is not None:
            out["correctness"] = correctness_score(correctness_label)
    return out


def _class_accuracy(correct: int, total: int) -> float | None:
    return (correct / total) if total else None


def aggregate_generation_metrics(per_query: list[dict]) -> dict:
    """Agrega métricas por query: medias (ignorando None) + bloque de abstención + balanced acc."""
    n = len(per_query)
    agg: dict = {"n_queries": n}
    for key in _JUDGE_NUMERIC_KEYS:
        vals = [d[key] for d in per_query if isinstance(d.get(key), int | float)]
        agg[f"{key}_mean"] = (sum(vals) / len(vals)) if vals else None
        agg[f"{key}_n"] = len(vals)

    outcomes = [d.get("abstention_outcome") for d in per_query]
    counts = {o: outcomes.count(o) for o in ABSTENTION_OUTCOMES}
    answerable_total = sum(1 for d in per_query if d.get("answerable"))
    unanswerable_total = n - answerable_total
    acc_answerable = _class_accuracy(counts["answered"], answerable_total)
    acc_unanswerable = _class_accuracy(counts["correct_abstention"], unanswerable_total)
    class_accs = [a for a in (acc_answerable, acc_unanswerable) if a is not None]
    balanced = (sum(class_accs) / len(class_accs)) if class_accs else None

    agg["abstention"] = {
        "counts": counts,
        "answerable_total": answerable_total,
        "unanswerable_total": unanswerable_total,
        "answer_rate_on_answerable": acc_answerable,
        "abstention_rate_on_unanswerable": acc_unanswerable,
        "over_abstention_rate": _class_accuracy(counts["over_abstention"], answerable_total),
        "false_answer_rate": _class_accuracy(counts["false_answer"], unanswerable_total),
        "balanced_accuracy": balanced,
        "hallucinated_forbidden_count": sum(
            1 for d in per_query if d.get("hallucinated_forbidden")
        ),
        "abstention_points": {
            point: sum(1 for d in per_query if d.get("abstention_point") == point)
            for point in ("answered", "pre_llm", "llm_decided")
        },
    }
    return agg
