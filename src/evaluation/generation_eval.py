"""Bucle de evaluación de generación: pregunta → RagAnswerV1 → métricas L3–L6.

Orquestación **pura y testeable offline**: recibe el `AnswerGenerator` y un juez inyectables (fakes
en los tests), recorre las preguntas de un split y produce el detalle por query + filas escalares
para CSV + el agregado. La llamada real al LLM/juez la encapsulan el generador y el juez; aquí solo
se enlazan datos y métricas.

Para la fidelidad (L3) se reconstruye el bloque de evidencias que vio el generador con el mismo
pipeline determinista (retrieve + build_evidences con la misma config), de modo que el juez evalúa
exactamente el contexto entregado al modelo.
"""

from __future__ import annotations

from typing import Protocol

from src.contracts.generation_models import RagAnswerV1
from src.evaluation.dataset import EvalAnswerKey, EvalQuestion
from src.evaluation.generation_metrics import (
    aggregate_generation_metrics,
    compute_query_generation_metrics,
)
from src.generation.answer_generator import AnswerGenerator
from src.generation.prompt import build_evidences_block
from src.retrieval.evidence_builder import build_evidences

_SCALAR_KEYS = (
    "key_fact_recall",
    "citation_precision",
    "citation_recall",
    "citation_f1",
    "faithfulness",
    "correctness",
)


class Judge(Protocol):
    """Interfaz mínima del juez que consume el bucle (la cumple `LlmJudge` y los fakes)."""

    model_label: str

    def judge_faithfulness(self, *, answer: str, evidences_block: str): ...

    def judge_correctness(self, *, question: str, answer: str, reference: str): ...


def _evidences_block_for(
    generator: AnswerGenerator, question: str, query_profile_id: str | None
) -> str:
    """Reconstruye el bloque de evidencias que vio el generador (mismo pipeline determinista)."""
    cfg = generator.config
    hits = generator.retriever.retrieve(
        question, query_profile_id=query_profile_id, top_k=cfg.top_k
    )
    selection = build_evidences(
        hits,
        parents_by_id=generator.retriever.corpus.get("parents_by_id", {}),
        max_evidences=cfg.max_evidences,
        context_strategy=cfg.context_strategy,
        context_budget_chars=cfg.context_budget_chars,
        max_total_context_chars=cfg.max_total_context_chars,
    )
    return build_evidences_block(selection.evidences)


def _scalar_row(q: EvalQuestion, metrics: dict, *, latency_s: float | None) -> dict:
    row = {
        "query_id": q.query_id,
        "split": q.split,
        "query_style": q.query_style,
        "failure_mode": q.failure_mode or "",
        "difficulty": q.difficulty,
        "answered": metrics["answered"],
        "answerable": metrics["answerable"],
        "abstention_outcome": metrics["abstention_outcome"],
        "abstention_point": metrics["abstention_point"],
        "hallucinated_forbidden": metrics.get("hallucinated_forbidden", False),
        "latency_s": latency_s if isinstance(latency_s, int | float) else "",
    }
    for key in _SCALAR_KEYS:
        value = metrics.get(key)
        row[key] = value if isinstance(value, int | float) else ""
    return row


def evaluate_generation(
    *,
    questions: list[dict],
    answer_keys: list[dict],
    generator: AnswerGenerator,
    judge: Judge | None = None,
    query_profile_id: str | None = None,
    limit: int | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """Evalúa la generación de un split. Devuelve (per_query, metrics_rows, aggregate)."""
    q_models = [EvalQuestion.model_validate(q) for q in questions]
    ak_by_qid = {a["query_id"]: EvalAnswerKey.model_validate(a) for a in answer_keys}
    selected = q_models if limit is None else q_models[:limit]

    per_query: list[dict] = []
    metrics_rows: list[dict] = []
    for q in selected:
        answer: RagAnswerV1 = generator.answer(q.query, query_profile_id=query_profile_id)
        cited_parents = [c.parent_id for c in answer.citations]
        ak = ak_by_qid.get(q.query_id)
        answerable = ak.answerable if ak is not None else (q.split != "out_of_corpus")
        key_facts = ak.key_facts if ak else []
        forbidden_facts = ak.forbidden_facts if ak else []
        expected_parents = ak.expected_citation_parents if ak else []
        reference = ak.reference_answer if ak else ""

        faithfulness_claims: list[bool] | None = None
        correctness_label: str | None = None
        if judge is not None and answer.answered and answerable:
            block = _evidences_block_for(generator, q.query, query_profile_id)
            faith_verdict, _ = judge.judge_faithfulness(answer=answer.answer, evidences_block=block)
            faithfulness_claims = [c.supported for c in faith_verdict.claims]
            if reference.strip():
                corr_verdict, _ = judge.judge_correctness(
                    question=q.query, answer=answer.answer, reference=reference
                )
                correctness_label = corr_verdict.verdict

        gen_metrics = answer.generation_metrics
        latency_s = gen_metrics.total_duration_s if gen_metrics is not None else None
        eval_count = gen_metrics.eval_count if gen_metrics is not None else None

        # Traza de retrieval para atribuir fallos: ¿el parent correcto se recuperó? ¿se entregó al
        # LLM o se omitió (p. ej. por presupuesto de contexto)? Permite distinguir fallo de
        # recuperación vs de ensamblado vs de generación en el análisis de error.
        trace = answer.retrieval_trace
        retrieved_parents = [h.parent_id for h in trace.hits]
        delivered_parents = [h.parent_id for h in trace.hits if h.selected]
        omitted_evidences = [
            {"parent_id": o.parent_id, "retrieval_rank": o.retrieval_rank, "reason": o.reason}
            for o in trace.omitted_evidences
        ]

        metrics = compute_query_generation_metrics(
            answered=answer.answered,
            answer_text=answer.answer,
            has_generation_metrics=answer.generation_metrics is not None,
            cited_parents=cited_parents,
            answerable=answerable,
            key_facts=key_facts,
            forbidden_facts=forbidden_facts,
            expected_citation_parents=expected_parents,
            faithfulness_claims=faithfulness_claims,
            correctness_label=correctness_label,
        )
        per_query.append(
            {
                "query_id": q.query_id,
                "split": q.split,
                "query_style": q.query_style,
                "failure_mode": q.failure_mode,
                "difficulty": q.difficulty,
                **metrics,
                "latency_s": latency_s,
                "eval_count": eval_count,
                "cited_parents": cited_parents,
                "delivered_parents": delivered_parents,
                "retrieved_parents": retrieved_parents,
                "omitted_evidences": omitted_evidences,
                "expected_citation_parents": expected_parents,
                "abstention_reason": answer.abstention_reason,
            }
        )
        metrics_rows.append(_scalar_row(q, metrics, latency_s=latency_s))

    aggregate = aggregate_generation_metrics(per_query)
    return per_query, metrics_rows, aggregate
