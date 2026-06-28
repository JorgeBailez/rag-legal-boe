"""CLI de evaluación de generación (L3–L6): corre el RAG sobre un split y escribe un report.

Mide fidelidad, citas, corrección y abstención de las respuestas generadas frente al dataset gold,
y escribe un report versionado y reproducible bajo `data/processed/reports/generation/<run_id>/`.

Requiere (solo en ejecución real, en el servidor): un bundle denso publicado, los pesos del modelo
de embeddings, y un Ollama local con el modelo generador y el modelo JUEZ (distinto, más fuerte).
El juez es opcional: sin `--judge-model`/`JUDGE_MODEL` se calculan las métricas que no lo necesitan
(abstención, key-fact recall, attribution), útiles para un primer barrido o un bake-off de modelos.

Uso:
    uv run python scripts/run_generation_eval.py \
      --bundle data/indexes/dense/<bundle_id> --split development --judge-model <modelo_grande>

Códigos de salida: 0 normal; ≠0 ante fallo técnico o (con --require-gate-c) Gate C no listo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm  # noqa: E402

from src.config.settings import get_settings  # noqa: E402
from src.core.exceptions import RagLegalBoeError  # noqa: E402
from src.embeddings.corpus_loader import load_processed_corpus  # noqa: E402
from src.embeddings.encoder import set_cpu_threads  # noqa: E402
from src.embeddings.fingerprints import fingerprint  # noqa: E402
from src.evaluation.dataset import (  # noqa: E402
    ANSWER_KEYS_FILE,
    DATASET_DIR,
    GATE_C_LEVELS,
    JUDGMENTS_FILE,
    QUESTIONS_FILE,
    load_and_validate,
    load_jsonl,
)
from src.evaluation.generation_eval import evaluate_generation  # noqa: E402
from src.evaluation.judge import LlmJudge  # noqa: E402
from src.evaluation.reports import new_run_id, write_generation_report  # noqa: E402
from src.generation.answer_generator import AnswerGenerator, GenerationConfig  # noqa: E402
from src.generation.ollama_client import OllamaClient  # noqa: E402
from src.generation.prompt import (  # noqa: E402
    RAG_PROMPT_FILE,
    SYSTEM_PROMPT_FILE,
    load_template,
)
from src.retrieval.dense_retriever import DenseRetriever  # noqa: E402


def _positive_int(value: str) -> int:
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"valor entero inválido: {value!r}") from None
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"debe ser un entero > 0 (recibido {ivalue}).")
    return ivalue


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluación de generación fundamentada (L3–L6).")
    parser.add_argument("--bundle", help="ruta al bundle; fallback a GENERATION_DENSE_BUNDLE.")
    parser.add_argument(
        "--split", default="development", choices=["development", "test", "out_of_corpus"]
    )
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--gate-c-level", default="checkpoint", choices=sorted(GATE_C_LEVELS))
    parser.add_argument(
        "--require-gate-c", action="store_true", help="exit≠0 si Gate C gen no listo."
    )
    parser.add_argument("--judge-model", default=None, help="modelo juez; fallback JUDGE_MODEL.")
    parser.add_argument(
        "--generator-model",
        default=None,
        help="modelo generador Ollama; fallback OLLAMA_MODEL/settings.ollama_model.",
    )
    parser.add_argument(
        "--no-judge", action="store_true", help="omite las métricas con juez (L3/L5)."
    )
    parser.add_argument(
        "--prompts-dir",
        default=None,
        help="directorio de prompts (system_prompt.txt + rag_prompt.txt); por defecto prompts/. "
        "Úsalo para A/B de prompts, p. ej. --prompts-dir prompts/v2.",
    )
    parser.add_argument("--query-profile-id", default=None)
    parser.add_argument("--top-k", type=_positive_int, default=None)
    parser.add_argument("--max-evidences", type=_positive_int, default=None)
    parser.add_argument(
        "--context-strategy",
        default=None,
        choices=["K_ONLY", "P_EXPAND_FULL", "P_EXPAND_BOUNDED"],
    )
    parser.add_argument("--context-budget-chars", type=_positive_int, default=None)
    parser.add_argument("--max-total-context-chars", type=_positive_int, default=None)
    parser.add_argument(
        "--num-predict",
        type=_positive_int,
        default=None,
        help="tokens máx. de salida del generador; fallback settings.ollama_num_predict. Subir si "
        "el LLM trunca el JSON (GenerationContractError).",
    )
    parser.add_argument(
        "--num-ctx",
        type=_positive_int,
        default=None,
        help="ventana de contexto del generador; fallback settings.ollama_num_ctx. Debe caber "
        "prompt + evidencia (max-total-context-chars) + salida.",
    )
    parser.add_argument(
        "--query-ids",
        default=None,
        help="comas; reevalúa solo esos query_id del split (p. ej. q0001,q0013,q0038).",
    )
    parser.add_argument("--threads", type=_positive_int, default=None)
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument(
        "--limit", type=_positive_int, default=None, help="evalúa solo N preguntas."
    )
    parser.add_argument("--output-root", default=None, help="raíz de reports de generación.")
    parser.add_argument(
        "--unload-model", action="store_true", help="descarga generador y juez al terminar."
    )
    return parser.parse_args()


def _override(value, default):  # noqa: ANN001, ANN202
    return value if value is not None else default


def _prompt_fingerprint(prompts_dir):  # noqa: ANN001, ANN202
    """Huella corta del contenido de los prompts (system+rag) para trazar el A/B."""
    try:
        text = load_template(SYSTEM_PROMPT_FILE, prompts_dir) + load_template(
            RAG_PROMPT_FILE, prompts_dir
        )
    except OSError:
        return "missing"
    return fingerprint(text)[:12]


def main() -> int:  # noqa: C901 - orquestación lineal del CLI
    args = _parse_args()
    settings = get_settings()

    bundle = args.bundle if args.bundle is not None else settings.generation_dense_bundle
    if not bundle:
        print("Falta el bundle denso. Indica --bundle o GENERATION_DENSE_BUNDLE.", file=sys.stderr)
        return 2

    threads = args.threads if args.threads is not None else settings.default_cpu_threads
    set_cpu_threads(threads)

    dataset_dir = Path(args.dataset_dir)
    corpus = load_processed_corpus()
    report = load_and_validate(dataset_dir, corpus=corpus, gate_c_level=args.gate_c_level)
    if report["errors"]:
        print(f"Dataset con {len(report['errors'])} errores estructurales; corrige antes.")
        for e in report["errors"][:10]:
            print(f"  - {e}", file=sys.stderr)
        return 1
    if args.require_gate_c and not report["gate_c"]["generation_ready"]:
        print("[--require-gate-c] Gate C de generación no listo → exit 1", file=sys.stderr)
        for r in report["gate_c"]["generation_reasons"]:
            print(f"  - {r}", file=sys.stderr)
        return 1
    if not report["gate_c"]["generation_ready"]:
        print("Aviso: Gate C de generación NO listo (dataset borrador). Resultados informativos.")

    all_questions = load_jsonl(dataset_dir / QUESTIONS_FILE)
    questions = [q for q in all_questions if q.get("split") == args.split]
    if args.query_ids:
        wanted = {x.strip() for x in args.query_ids.split(",") if x.strip()}
        questions = [q for q in questions if q.get("query_id") in wanted]
    answer_keys = load_jsonl(dataset_dir / ANSWER_KEYS_FILE)
    _ = load_jsonl(dataset_dir / JUDGMENTS_FILE)  # cargados/validados arriba; retrieval gold aparte
    if not questions:
        print(
            f"No hay preguntas en el split {args.split!r} (¿--query-ids correctos?).",
            file=sys.stderr,
        )
        return 1

    config = GenerationConfig(
        query_profile_id=_override(args.query_profile_id, settings.generation_query_profile_id),
        top_k=_override(args.top_k, settings.generation_top_k),
        max_evidences=_override(args.max_evidences, settings.generation_max_evidences),
        context_strategy=_override(args.context_strategy, settings.generation_context_strategy),
        context_budget_chars=_override(
            args.context_budget_chars, settings.generation_context_budget_chars
        ),
        max_total_context_chars=_override(
            args.max_total_context_chars, settings.generation_max_total_context_chars
        ),
        temperature=settings.ollama_temperature,
        seed=settings.ollama_seed,
        num_predict=_override(args.num_predict, settings.ollama_num_predict),
        num_ctx=_override(args.num_ctx, settings.ollama_num_ctx),
        keep_alive=settings.ollama_keep_alive,
    )

    try:
        retriever = DenseRetriever.from_bundle(bundle, corpus=corpus, batch_size=args.batch_size)
    except Exception as exc:  # noqa: BLE001 - fallo técnico de carga del bundle/modelo
        print(f"No se pudo cargar el bundle/modelo: {exc}", file=sys.stderr)
        return 1

    generator_model = (
        args.generator_model if args.generator_model is not None else settings.ollama_model
    )
    gen_client = OllamaClient(
        base_url=settings.ollama_base_url,
        model=generator_model,
        timeout=settings.ollama_timeout_seconds,
        think=settings.ollama_think,
        keep_alive=settings.ollama_keep_alive,
    )
    generator = AnswerGenerator(
        retriever=retriever, llm_client=gen_client, config=config, prompts_dir=args.prompts_dir
    )

    judge = None
    judge_client = None
    judge_model = args.judge_model if args.judge_model is not None else settings.judge_model
    if not args.no_judge and judge_model:
        judge_client = OllamaClient(
            base_url=settings.judge_base_url,
            model=judge_model,
            timeout=settings.judge_timeout_seconds,
            keep_alive=settings.judge_keep_alive,
        )
        judge = LlmJudge(
            client=judge_client,
            num_ctx=settings.judge_num_ctx,
            num_predict=settings.judge_num_predict,
            temperature=settings.judge_temperature,
            seed=settings.judge_seed,
            model_label=judge_model,
        )
    elif not args.no_judge:
        print("Aviso: sin JUDGE_MODEL/--judge-model; se omiten las métricas con juez (L3/L5).")

    total = len(questions) if args.limit is None else min(args.limit, len(questions))
    bar = tqdm(total=total, desc="generación", unit="q")

    def on_progress(ev: dict) -> None:
        event = ev.get("event")
        if event == "start":
            bar.set_description(str(ev["query_id"]))
            bar.set_postfix_str("generando…")
        elif event == "judging":
            bar.set_postfix_str(f"juez:{ev['phase']}…")
        elif event == "done":
            mark = "✓ resp" if ev["answered"] else "○ abst"
            lat = f" {ev['latency_s']:.0f}s" if ev.get("latency_s") else ""
            tag = ev.get("failure_mode") or ev.get("query_style") or ""
            jerr = " ⚠ juez falló" if ev.get("judge_error") else ""
            tqdm.write(
                f"  [{ev['i']}/{ev['total']}] {ev['query_id']} {tag}: "
                f"{mark} · {ev['abstention_outcome']}{lat}{jerr}"
            )
            bar.set_postfix_str("")
            bar.update(1)

    exit_code = 0
    try:
        try:
            per_query, metrics_rows, aggregate = evaluate_generation(
                questions=questions,
                answer_keys=answer_keys,
                generator=generator,
                judge=judge,
                query_profile_id=config.query_profile_id,
                limit=args.limit,
                on_progress=on_progress,
            )
        finally:
            bar.close()
        run_config = {
            "split": args.split,
            "dataset_dir": str(dataset_dir),
            "bundle_id": retriever.bundle_id,
            "model_alias": retriever.model_alias,
            "generator_model": generator_model,
            "judge_model": judge_model if judge is not None else None,
            "query_profile_id": retriever.resolved_query_profile_id(config.query_profile_id),
            "top_k": config.top_k,
            "max_evidences": config.max_evidences,
            "context_strategy": config.context_strategy,
            "context_budget_chars": config.context_budget_chars,
            "max_total_context_chars": config.max_total_context_chars,
            "temperature": config.temperature,
            "seed": config.seed,
            "num_predict": config.num_predict,
            "num_ctx": config.num_ctx,
            "judge_num_ctx": settings.judge_num_ctx if judge is not None else None,
            "judge_num_predict": settings.judge_num_predict if judge is not None else None,
            "prompts_dir": args.prompts_dir or "(default)",
            "prompt_fingerprint": _prompt_fingerprint(args.prompts_dir),
            "gate_c_generation_ready": report["gate_c"]["generation_ready"],
            "n_questions": len(per_query),
        }
        run_id = new_run_id("gen", fingerprint(run_config))
        output_root = Path(args.output_root) if args.output_root else None
        write_kwargs = {"reports_root": output_root} if output_root else {}
        summary_keys = ("split", "bundle_id", "generator_model", "judge_model", "n_questions")
        out_dir = write_generation_report(
            run_id,
            summary={k: run_config[k] for k in summary_keys},
            config=run_config,
            per_query=per_query,
            metrics_rows=metrics_rows,
            aggregate=aggregate,
            **write_kwargs,
        )
        _print_summary(out_dir, aggregate)
        n_gen_err = sum(1 for r in per_query if r.get("generation_error"))
        if n_gen_err:
            print(
                f"⚠ {n_gen_err} pregunta(s) con generación fallida por contrato (excluidas de las "
                "métricas; ver generation_error en per_query.jsonl). La corrida NO se abortó."
            )
        n_judge_err = sum(1 for r in per_query if r.get("judge_error"))
        if n_judge_err:
            print(
                f"⚠ {n_judge_err} pregunta(s) con veredicto del juez fallido (no juzgadas; "
                "ver judge_error en per_query.jsonl). La corrida NO se abortó."
            )
    except RagLegalBoeError as exc:
        print(f"Fallo de evaluación: {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        if args.unload_model:
            for c in (gen_client, judge_client):
                if c is not None:
                    try:
                        c.unload()
                    except RagLegalBoeError as exc:
                        print(f"Aviso: no se pudo descargar un modelo: {exc}", file=sys.stderr)
        gen_client.close()
        if judge_client is not None:
            judge_client.close()

    return exit_code


def _print_summary(out_dir: Path, aggregate: dict) -> None:
    ab = aggregate["abstention"]
    print(f"\nReport: {out_dir}")
    print(f"preguntas: {aggregate['n_queries']}")
    print(
        f"abstención: balanced_acc={_fmt(ab['balanced_accuracy'])} "
        f"answer_rate(answerable)={_fmt(ab['answer_rate_on_answerable'])} "
        f"abst_rate(unanswerable)={_fmt(ab['abstention_rate_on_unanswerable'])} "
        f"false_answer_rate={_fmt(ab['false_answer_rate'])}"
    )
    print(
        f"contenido: faithfulness={_fmt(aggregate['faithfulness_mean'])} "
        f"correctness={_fmt(aggregate['correctness_mean'])} "
        f"key_fact_recall={_fmt(aggregate['key_fact_recall_mean'])} "
        f"citation_f1={_fmt(aggregate['citation_f1_mean'])}"
    )
    if ab["hallucinated_forbidden_count"]:
        print(f"⚠ hechos prohibidos detectados en {ab['hallucinated_forbidden_count']} respuestas")


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if isinstance(value, int | float) else "n/a"


if __name__ == "__main__":
    raise SystemExit(main())
