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
from types import SimpleNamespace
from typing import Protocol

from src.contracts.generation_models import RetrievalTraceV1
from src.core.exceptions import RagLegalBoeError
from src.evaluation.dataset import EvalAnswerKey, EvalQuestion
from src.evaluation.generation_metrics import (
    aggregate_generation_metrics,
    compute_query_generation_metrics,
    correctness_score,
    faithfulness_score,
)
from src.generation.answer_generator import AnswerGenerator
from src.generation.prompt import build_evidences_block
from src.retrieval.evidence_builder import build_evidences, build_oracle_evidences

EVAL_MODES = ("rag", "closed_book", "oracle")

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


def _oracle_gold_for(judgments_for_q: list[dict]) -> list[dict]:
    """Gold del oracle: parent_id + párrafos de evidencia, ordenado por relevancia descendente."""
    ordered = sorted(judgments_for_q, key=lambda j: j.get("relevance", 0), reverse=True)
    return [
        {
            "parent_id": j["parent_id"],
            "paragraph_orders": (j.get("evidence") or {}).get("paragraph_orders", []),
            "relevance": j.get("relevance", 0),
        }
        for j in ordered
    ]


def _empty_trace(generator: AnswerGenerator, resolved_profile: str) -> RetrievalTraceV1:
    """Traza vacía (closed-book no recupera nada; mantiene la forma del report)."""
    return RetrievalTraceV1(
        bundle_id=generator.retriever.bundle_id,
        model_alias=generator.retriever.model_alias,
        query_profile_id=resolved_profile,
        top_k=0,
        returned_hits=0,
        selected_evidences=0,
    )


def evaluate_generation(
    *,
    questions: list[dict],
    answer_keys: list[dict],
    generator: AnswerGenerator,
    judge: Judge | None = None,
    query_profile_id: str | None = None,
    mode: str = "rag",
    judgments: list[dict] | None = None,
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

    if mode not in EVAL_MODES:
        raise RagLegalBoeError(f"modo de evaluación desconocido: {mode!r} (esperado {EVAL_MODES})")
    judgments_by_qid: dict[str, list[dict]] = {}
    for j in judgments or []:
        if j.get("relevance", 0) >= 1:
            judgments_by_qid.setdefault(j["query_id"], []).append(j)
    resolved_profile = generator.retriever.resolved_query_profile_id(
        query_profile_id or generator.config.query_profile_id
    )

    per_query: list[dict] = []
    metrics_rows: list[dict] = []
    for idx, q in enumerate(selected, start=1):
        notify(
            {
                "event": "start",
                "i": idx,
                "total": total,
                "query_id": q.query_id,
                "failure_mode": q.failure_mode,
                "query_style": q.query_style,
            }
        )
        ak = ak_by_qid.get(q.query_id)
        answerable = ak.answerable if ak is not None else (q.split != "out_of_corpus")
        oracle_selection = None
        try:
            if mode == "closed_book":
                cb = generator.generate_closed_book(q.query)
                # Shim: el closed-book no cabe en RagAnswerV1 (no hay citas); se expone con la misma
                # forma de atributos que consume el resto del bucle (citas y traza vacías).
                answer = SimpleNamespace(
                    answered=cb.answered,
                    answer=cb.answer,
                    abstention_reason=cb.abstention_reason,
                    citations=[],
                    generation_metrics=cb.generation_metrics,
                    retrieval_trace=_empty_trace(generator, resolved_profile),
                )
            elif mode == "oracle":
                oracle_selection = build_oracle_evidences(
                    _oracle_gold_for(judgments_by_qid.get(q.query_id, [])),
                    parents_by_id=generator.retriever.corpus.get("parents_by_id", {}),
                    max_evidences=generator.config.max_evidences,
                    context_strategy=generator.config.context_strategy,
                    context_budget_chars=generator.config.context_budget_chars,
                    max_total_context_chars=generator.config.max_total_context_chars,
                )
                answer = generator.answer_with_evidences(
                    q.query, oracle_selection, query_profile_id=query_profile_id
                )
            else:
                answer = generator.answer(q.query, query_profile_id=query_profile_id)
        except RagLegalBoeError as exc:
            # Un fallo de contrato del generador (JSON inválido del LLM, ID inventado…) NO debe
            # abortar la corrida entera (horas): se registra como error técnico de la pregunta y se
            # continúa. Estas filas se EXCLUYEN de las métricas (no son una abstención deliberada).
            per_query.append(
                {
                    "query_id": q.query_id,
                    "split": q.split,
                    "query_style": q.query_style,
                    "failure_mode": q.failure_mode,
                    "difficulty": q.difficulty,
                    "mode": mode,
                    "answered": False,
                    "answerable": answerable,
                    "abstention_outcome": "generation_error",
                    "generation_error": str(exc),
                }
            )
            notify(
                {
                    "event": "done",
                    "i": idx,
                    "total": total,
                    "query_id": q.query_id,
                    "answered": False,
                    "answerable": answerable,
                    "latency_s": None,
                    "abstention_outcome": "generation_error",
                    "generation_error": str(exc),
                    "failure_mode": q.failure_mode,
                    "query_style": q.query_style,
                }
            )
            continue
        cited_parents = [c.parent_id for c in answer.citations]
        key_facts = ak.key_facts if ak else []
        forbidden_facts = ak.forbidden_facts if ak else []
        expected_parents = ak.expected_citation_parents if ak else []
        reference = ak.reference_answer if ak else ""

        faithfulness_claims: list[bool] | None = None
        correctness_label: str | None = None
        judge_error: str | None = None
        evidences_block: str | None = None
        # El bloque de evidencias que vio el generador se reconstruye y se guarda en el report
        # SIEMPRE que haya respuesta (haya juez o no): el anotador humano necesita exactamente esa
        # evidencia para validar la fidelidad (afirmación-contra-evidencia); sin ella ni el κ de L3
        # ni la anotación humana que lo sustituye son válidos.
        if answer.answered and answerable:
            if mode == "oracle" and oracle_selection is not None:
                evidences_block = build_evidences_block(oracle_selection.evidences)
            elif mode == "rag":
                try:
                    evidences_block = _evidences_block_for(generator, q.query, query_profile_id)
                except RagLegalBoeError as exc:
                    # Un fallo reconstruyendo la evidencia no debe abortar la corrida.
                    evidences_block = None
                    if judge is not None:
                        judge_error = str(exc)
            # closed_book: sin evidencia → no hay bloque que juzgar (fidelidad no aplica).

        if judge is not None and evidences_block is not None:
            try:
                notify(
                    {
                        "event": "judging",
                        "phase": "fidelidad",
                        "i": idx,
                        "total": total,
                        "query_id": q.query_id,
                    }
                )
                faith_verdict, _ = judge.judge_faithfulness(
                    answer=answer.answer, evidences_block=evidences_block
                )
                faithfulness_claims = [c.supported for c in faith_verdict.claims]
                if reference.strip():
                    notify(
                        {
                            "event": "judging",
                            "phase": "corrección",
                            "i": idx,
                            "total": total,
                            "query_id": q.query_id,
                        }
                    )
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
                "mode": mode,
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
                "evidences_block": evidences_block,
                "judge_error": judge_error,
            }
        )
        metrics_rows.append(_scalar_row(q, metrics, latency_s=latency_s))
        notify(
            {
                "event": "done",
                "i": idx,
                "total": total,
                "query_id": q.query_id,
                "answered": answer.answered,
                "answerable": answerable,
                "latency_s": latency_s,
                "abstention_outcome": metrics["abstention_outcome"],
                "judge_error": judge_error,
                "failure_mode": q.failure_mode,
                "query_style": q.query_style,
            }
        )

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
        r.get("query_id")
        for r in selected
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
        notify(
            {
                "event": "start",
                "i": idx,
                "total": total,
                "query_id": qid,
                "failure_mode": q.failure_mode,
                "query_style": q.query_style,
            }
        )

        correctness_label: str | None = None
        judge_error: str | None = None
        if judge is not None and answered and ak.answerable and ak.reference_answer.strip():
            try:
                notify(
                    {
                        "event": "judging",
                        "phase": "corrección",
                        "i": idx,
                        "total": total,
                        "query_id": qid,
                    }
                )
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
        notify(
            {
                "event": "done",
                "i": idx,
                "total": total,
                "query_id": qid,
                "answered": answered,
                "answerable": ak.answerable,
                "latency_s": pr.get("latency_s"),
                "abstention_outcome": metrics["abstention_outcome"],
                "judge_error": judge_error,
                "failure_mode": q.failure_mode,
                "query_style": q.query_style,
            }
        )

    aggregate = aggregate_generation_metrics(per_query)
    return per_query, metrics_rows, aggregate


def rejudge_report(
    *,
    prior_per_query: list[dict],
    answer_keys: list[dict],
    questions: list[dict],
    judge: Judge,
    limit: int | None = None,
    on_progress: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Re-juzga fidelidad (L3) y corrección (L5) con un prompt de juez nuevo, sin regenerar.

    Reutiliza `answer_text` y `evidences_block` ya guardados en un report previo (solo las
    respuestas; las abstenciones se omiten). Pensado para **calibrar el prompt del juez** y
    re-validar κ/AC1 contra la MISMA anotación humana, aislando el efecto del prompt. Devuelve filas
    `per_query` con `faithfulness`/`correctness` nuevos (las consume `validate_judge.py
    --annotations`). Si una fila no trae `evidences_block` (report antiguo) la fidelidad queda
    `None`; sin `reference_answer`, la corrección queda `None`. Un veredicto malformado del juez NO
    aborta: se marca `judge_error`.
    """
    q_by_qid = {q["query_id"]: EvalQuestion.model_validate(q) for q in questions}
    ak_by_qid = {a["query_id"]: EvalAnswerKey.model_validate(a) for a in answer_keys}
    answered = [r for r in prior_per_query if r.get("answered")]
    selected = answered if limit is None else answered[:limit]
    total = len(selected)
    notify = on_progress or (lambda _info: None)

    per_query: list[dict] = []
    for idx, pr in enumerate(selected, start=1):
        qid = pr["query_id"]
        q = q_by_qid.get(qid)
        ak = ak_by_qid.get(qid)
        answer_text = pr.get("answer_text", "")
        evidences_block = pr.get("evidences_block") or ""
        reference = ak.reference_answer if ak else ""
        notify({"event": "start", "i": idx, "total": total, "query_id": qid})

        faithfulness: float | None = None
        correctness: float | None = None
        judge_error: str | None = None
        try:
            if evidences_block.strip():
                notify({"event": "judging", "phase": "fidelidad", "query_id": qid})
                faith_verdict, _ = judge.judge_faithfulness(
                    answer=answer_text, evidences_block=evidences_block
                )
                faithfulness = faithfulness_score([c.supported for c in faith_verdict.claims])
            if reference.strip() and q is not None:
                notify({"event": "judging", "phase": "corrección", "query_id": qid})
                corr_verdict, _ = judge.judge_correctness(
                    question=q.query, answer=answer_text, reference=reference
                )
                correctness = correctness_score(corr_verdict.verdict)
        except RagLegalBoeError as exc:
            judge_error = str(exc)

        per_query.append(
            {
                "query_id": qid,
                "split": pr.get("split"),
                "query_style": pr.get("query_style"),
                "answered": True,
                "faithfulness": faithfulness,
                "correctness": correctness,
                "answer_text": answer_text,
                "evidences_block": evidences_block,
                "judge_error": judge_error,
                "rejudged": True,
            }
        )
        notify(
            {
                "event": "done",
                "i": idx,
                "total": total,
                "query_id": qid,
                "faithfulness": faithfulness,
                "correctness": correctness,
                "judge_error": judge_error,
            }
        )
    return per_query
