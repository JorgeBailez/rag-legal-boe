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

from collections.abc import Callable
from typing import Protocol

from src.contracts.generation_models import RagAnswerV1
from src.core.exceptions import RagLegalBoeError
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
    on_progress: Callable[[dict], None] | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """Evalúa la generación de un split. Devuelve (per_query, metrics_rows, aggregate).

    `on_progress` (opcional) recibe eventos de avance por pregunta y sub-fase
    (`start` → `judging` → `done`) para pintar una barra de progreso o trazas; si es None, no hay
    salida (los tests corren mudos).
    """
    q_models = [EvalQuestion.model_validate(q) for q in questions]
    ak_by_qid = {a["query_id"]: EvalAnswerKey.model_validate(a) for a in answer_keys}
    selected = q_models if limit is None else q_models[:limit]
    total = len(selected)
    notify = on_progress or (lambda _info: None)

    per_query: list[dict] = []
    metrics_rows: list[dict] = []
    for idx, q in enumerate(selected, start=1):
        notify({"event": "start", "i": idx, "total": total, "query_id": q.query_id,
                "failure_mode": q.failure_mode, "query_style": q.query_style})
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
        judge_error: str | None = None
        if judge is not None and answer.answered and answerable:
            try:
                notify({"event": "judging", "phase": "fidelidad", "i": idx, "total": total,
                        "query_id": q.query_id})
                block = _evidences_block_for(generator, q.query, query_profile_id)
                faith_verdict, _ = judge.judge_faithfulness(
                    answer=answer.answer, evidences_block=block
                )
                faithfulness_claims = [c.supported for c in faith_verdict.claims]
                if reference.strip():
                    notify({"event": "judging", "phase": "corrección", "i": idx, "total": total,
                            "query_id": q.query_id})
                    corr_verdict, _ = judge.judge_correctness(
                        question=q.query, answer=answer.answer, reference=reference
                    )
                    correctness_label = corr_verdict.verdict
            except RagLegalBoeError as exc:
                # Un veredicto malformado/cortado del juez NO debe abortar la corrida (horas en
                # CPU): se marca la pregunta como no juzgada (L3/L5=None) y se continúa.
                judge_error = str(exc)
                faithfulness_claims = None
                correctness_label = None

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
                "answer_text": answer.answer,
                "abstention_reason": answer.abstention_reason,
                "judge_error": judge_error,
            }
        )
        metrics_rows.append(_scalar_row(q, metrics, latency_s=latency_s))
        notify({"event": "done", "i": idx, "total": total, "query_id": q.query_id,
                "answered": answer.answered, "answerable": answerable, "latency_s": latency_s,
                "abstention_outcome": metrics["abstention_outcome"], "judge_error": judge_error,
                "failure_mode": q.failure_mode, "query_style": q.query_style})

    aggregate = aggregate_generation_metrics(per_query)
    return per_query, metrics_rows, aggregate


def rejudge_correctness(
    *,
    prior_per_query: list[dict],
    answer_keys: list[dict],
    questions: list[dict],
    judge: Judge | None = None,
    limit: int | None = None,
    on_progress: Callable[[dict], None] | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """Recalcula las métricas reutilizando respuestas y fidelidad de un report previo.

    Pensado para iterar sobre el gold/el prompt del juez **sin regenerar**: solo re-ejecuta el juez
    de CORRECCIÓN (L5), que es lo único que depende de `reference_answer`. La fidelidad (L3) no mira
    la referencia → se **reutiliza** el valor guardado en `prior_per_query`; el resto (key-facts,
    citas, abstención, hechos prohibidos) se recalcula sobre la respuesta guardada y el answer_key
    actual (funciones puras, sin LLM). No recupera ni genera: no necesita bundle ni encoder.

    `prior_per_query` son las filas de un `per_query.jsonl` previo (deben traer al menos
    `query_id`, `answered`, `answer_text`, `faithfulness` y `eval_count`). Devuelve la
    misma tripleta que `evaluate_generation`.
    """
    q_by_qid = {q["query_id"]: EvalQuestion.model_validate(q) for q in questions}
    ak_by_qid = {a["query_id"]: EvalAnswerKey.model_validate(a) for a in answer_keys}
    selected = prior_per_query if limit is None else prior_per_query[:limit]

    missing = [
        r.get("query_id") for r in selected
        if r.get("query_id") not in q_by_qid or r.get("query_id") not in ak_by_qid
    ]
    if missing:
        raise RagLegalBoeError(
            "el report previo y el dataset no casan; faltan en questions/answer_keys: "
            + ", ".join(str(m) for m in missing[:10])
        )

    total = len(selected)
    notify = on_progress or (lambda _info: None)

    per_query: list[dict] = []
    metrics_rows: list[dict] = []
    for idx, pr in enumerate(selected, start=1):
        qid = pr["query_id"]
        q = q_by_qid[qid]
        ak = ak_by_qid[qid]
        answered = bool(pr.get("answered"))
        answer_text = pr.get("answer_text", "")
        cited_parents = pr.get("cited_parents", [])
        has_gen_metrics = pr.get("eval_count") is not None
        prior_faith = pr.get("faithfulness")
        notify({"event": "start", "i": idx, "total": total, "query_id": qid,
                "failure_mode": q.failure_mode, "query_style": q.query_style})

        correctness_label: str | None = None
        judge_error: str | None = None
        if judge is not None and answered and ak.answerable and ak.reference_answer.strip():
            try:
                notify({"event": "judging", "phase": "corrección", "i": idx, "total": total,
                        "query_id": qid})
                corr_verdict, _ = judge.judge_correctness(
                    question=q.query, answer=answer_text, reference=ak.reference_answer
                )
                correctness_label = corr_verdict.verdict
            except RagLegalBoeError as exc:
                judge_error = str(exc)
                correctness_label = None

        metrics = compute_query_generation_metrics(
            answered=answered,
            answer_text=answer_text,
            has_generation_metrics=has_gen_metrics,
            cited_parents=cited_parents,
            answerable=ak.answerable,
            key_facts=ak.key_facts,
            forbidden_facts=ak.forbidden_facts,
            expected_citation_parents=ak.expected_citation_parents,
            faithfulness_claims=None,  # L3 reutilizada del report previo (no re-juzgada)
            correctness_label=correctness_label,
        )
        if isinstance(prior_faith, int | float):
            metrics["faithfulness"] = prior_faith

        per_query.append(
            {
                "query_id": qid,
                "split": pr.get("split", q.split),
                "query_style": pr.get("query_style", q.query_style),
                "failure_mode": pr.get("failure_mode", q.failure_mode),
                "difficulty": pr.get("difficulty", q.difficulty),
                **metrics,
                "latency_s": pr.get("latency_s"),
                "eval_count": pr.get("eval_count"),
                "cited_parents": cited_parents,
                "delivered_parents": pr.get("delivered_parents", []),
                "retrieved_parents": pr.get("retrieved_parents", []),
                "omitted_evidences": pr.get("omitted_evidences", []),
                "expected_citation_parents": ak.expected_citation_parents,
                "answer_text": answer_text,
                "abstention_reason": pr.get("abstention_reason", ""),
                "judge_error": judge_error,
                "faithfulness_source": "reused" if isinstance(prior_faith, int | float) else "none",
                "rejudged": True,
            }
        )
        metrics_rows.append(_scalar_row(q, metrics, latency_s=pr.get("latency_s")))
        notify({"event": "done", "i": idx, "total": total, "query_id": qid,
                "answered": answered, "answerable": ak.answerable,
                "latency_s": pr.get("latency_s"),
                "abstention_outcome": metrics["abstention_outcome"], "judge_error": judge_error,
                "failure_mode": q.failure_mode, "query_style": q.query_style})

    aggregate = aggregate_generation_metrics(per_query)
    return per_query, metrics_rows, aggregate
