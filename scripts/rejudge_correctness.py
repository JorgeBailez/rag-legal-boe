"""Re-juzga SOLO la corrección (L5) reutilizando las respuestas de un report previo.

Cuando solo cambian las referencias del gold (o el prompt del juez de corrección) y la generación
queda congelada, no hace falta regenerar ni re-juzgar la fidelidad: la respuesta del generador es la
misma y la fidelidad (L3) no mira la referencia. Este runner lee un `per_query.jsonl` previo,
reutiliza respuestas + veredictos de fidelidad y vuelve a pasar por el juez **solo** la corrección,
recalculando de paso las métricas puras (key-facts, citas, abstención). NO necesita bundle ni
encoder: solo el juez (Ollama).

Es la herramienta del bucle «afinar el juez hasta κ≥0.6»: cada iteración es ~1 llamada de juez por
pregunta respondida, en vez de regenerar + 2 llamadas de juez.

Uso:
    uv run python scripts/rejudge_correctness.py \
      --report data/processed/reports/generation/<run_id_previo> --judge-model <modelo_juez>

Después: validar el juez sobre el NUEVO report con la anotación humana (las respuestas no han
cambiado, así que la anotación se reutiliza):
    uv run python scripts/validate_judge.py --report <nuevo_report> --annotations anotacion_juez.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm  # noqa: E402

from src.config.settings import get_settings  # noqa: E402
from src.core.exceptions import RagLegalBoeError  # noqa: E402
from src.embeddings.fingerprints import fingerprint  # noqa: E402
from src.evaluation.dataset import (  # noqa: E402
    ANSWER_KEYS_FILE,
    DATASET_DIR,
    QUESTIONS_FILE,
    load_jsonl,
)
from src.evaluation.generation_eval import rejudge_correctness  # noqa: E402
from src.evaluation.judge import CORRECTNESS_PROMPT_FILE, LlmJudge  # noqa: E402
from src.evaluation.reports import new_run_id, write_generation_report  # noqa: E402
from src.generation.ollama_client import OllamaClient  # noqa: E402
from src.generation.prompt import load_template  # noqa: E402


def _positive_int(value: str) -> int:
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"valor entero inválido: {value!r}") from None
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"debe ser un entero > 0 (recibido {ivalue}).")
    return ivalue


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-juzga solo la corrección (L5) reutilizando un report previo."
    )
    parser.add_argument(
        "--report",
        required=True,
        help="report previo (debe tener per_query.jsonl con answer_text).",
    )
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--judge-model", default=None, help="modelo juez; fallback JUDGE_MODEL.")
    parser.add_argument(
        "--judge-prompts-dir",
        default=None,
        help="directorio de prompts del juez (para A/B del prompt de corrección); por defecto "
        "prompts/. P. ej. --judge-prompts-dir prompts/judge_v2.",
    )
    parser.add_argument(
        "--query-ids",
        default=None,
        help="comas; re-juzga solo esos query_id del report (p. ej. q0001,q0019).",
    )
    parser.add_argument(
        "--limit", type=_positive_int, default=None, help="re-juzga solo las primeras N filas."
    )
    parser.add_argument("--output-root", default=None, help="raíz de reports de generación.")
    parser.add_argument(
        "--unload-model", action="store_true", help="descarga el modelo juez al terminar."
    )
    return parser.parse_args()


def _print_summary(out_dir: Path, aggregate: dict) -> None:
    ab = aggregate["abstention"]
    print(f"\nReport: {out_dir}")
    print(f"preguntas: {aggregate['n_queries']}")
    print(
        f"contenido: faithfulness(reusada)={_fmt(aggregate['faithfulness_mean'])} "
        f"correctness={_fmt(aggregate['correctness_mean'])} "
        f"key_fact_recall={_fmt(aggregate['key_fact_recall_mean'])} "
        f"citation_f1={_fmt(aggregate['citation_f1_mean'])}"
    )
    print(
        f"abstención: balanced_acc={_fmt(ab['balanced_accuracy'])} "
        f"false_answer_rate={_fmt(ab['false_answer_rate'])}"
    )
    if ab["hallucinated_forbidden_count"]:
        print(f"⚠ hechos prohibidos detectados en {ab['hallucinated_forbidden_count']} respuestas")


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if isinstance(value, int | float) else "n/a"


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    args = _parse_args()
    settings = get_settings()

    report_dir = Path(args.report)
    prior_per_query = load_jsonl(report_dir / "per_query.jsonl")
    if not prior_per_query:
        print(f"No hay per_query.jsonl en {report_dir} (¿report de generación?).", file=sys.stderr)
        return 2
    if args.query_ids:
        wanted = {x.strip() for x in args.query_ids.split(",") if x.strip()}
        prior_per_query = [r for r in prior_per_query if r.get("query_id") in wanted]
        if not prior_per_query:
            print("Ningún query_id de --query-ids está en el report previo.", file=sys.stderr)
            return 1

    judge_model = args.judge_model if args.judge_model is not None else settings.judge_model
    if not judge_model:
        print(
            "El re-juzgado necesita un modelo juez. Indica --judge-model o JUDGE_MODEL.",
            file=sys.stderr,
        )
        return 2

    dataset_dir = Path(args.dataset_dir)
    questions = load_jsonl(dataset_dir / QUESTIONS_FILE)
    answer_keys = load_jsonl(dataset_dir / ANSWER_KEYS_FILE)

    judge_client = OllamaClient(
        base_url=settings.judge_base_url,
        model=judge_model,
        timeout=settings.judge_timeout_seconds,
        keep_alive=settings.judge_keep_alive,
    )
    judge = LlmJudge(
        client=judge_client,
        prompts_dir=args.judge_prompts_dir,
        num_ctx=settings.judge_num_ctx,
        num_predict=settings.judge_num_predict,
        temperature=settings.judge_temperature,
        seed=settings.judge_seed,
        model_label=judge_model,
    )

    bar = tqdm(total=len(prior_per_query), desc="re-juzgando", unit="q")

    def on_progress(ev: dict) -> None:
        event = ev.get("event")
        if event == "start":
            bar.set_description(str(ev["query_id"]))
            bar.set_postfix_str("")
        elif event == "judging":
            bar.set_postfix_str("juez:corrección…")
        elif event == "done":
            mark = "✓ resp" if ev["answered"] else "○ abst"
            tag = ev.get("failure_mode") or ev.get("query_style") or ""
            jerr = " ⚠ juez falló" if ev.get("judge_error") else ""
            tqdm.write(
                f"  [{ev['i']}/{ev['total']}] {ev['query_id']} {tag}: "
                f"{mark} · {ev['abstention_outcome']}{jerr}"
            )
            bar.set_postfix_str("")
            bar.update(1)

    exit_code = 0
    try:
        try:
            per_query, metrics_rows, aggregate = rejudge_correctness(
                prior_per_query=prior_per_query,
                answer_keys=answer_keys,
                questions=questions,
                judge=judge,
                limit=args.limit,
                on_progress=on_progress,
            )
        finally:
            bar.close()

        prior_config_path = report_dir / "config.json"
        source_config = (
            json.loads(prior_config_path.read_text(encoding="utf-8"))
            if prior_config_path.exists()
            else None
        )
        splits = sorted({r.get("split") for r in per_query if r.get("split")})
        prompt_fp = fingerprint(load_template(CORRECTNESS_PROMPT_FILE, args.judge_prompts_dir))[:12]
        run_config = {
            "mode": "rejudge_correctness",
            "rejudged_from": report_dir.name,
            "faithfulness_source": "reused_from_prior_report",
            "split": splits[0] if len(splits) == 1 else "mixed",
            "dataset_dir": str(dataset_dir),
            "judge_model": judge_model,
            "judge_prompts_dir": args.judge_prompts_dir or "(default)",
            "judge_correctness_prompt_fingerprint": prompt_fp,
            "n_questions": len(per_query),
            "source_config": source_config,
        }
        run_id = new_run_id("gen", fingerprint(run_config))
        output_root = Path(args.output_root) if args.output_root else None
        write_kwargs = {"reports_root": output_root} if output_root else {}
        summary_keys = ("mode", "rejudged_from", "split", "judge_model", "n_questions")
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
        n_judge_err = sum(1 for r in per_query if r.get("judge_error"))
        if n_judge_err:
            print(
                f"⚠ {n_judge_err} pregunta(s) con veredicto del juez fallido (no juzgadas; "
                "ver judge_error en per_query.jsonl)."
            )
        print(
            "\nSiguiente paso: validar κ sobre este report (reutiliza tu anotación humana):\n"
            f"  uv run python scripts/validate_judge.py --report {out_dir} "
            "--annotations anotacion_juez.jsonl"
        )
    except RagLegalBoeError as exc:
        print(f"Fallo de re-juzgado: {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        if args.unload_model:
            try:
                judge_client.unload()
            except RagLegalBoeError as exc:
                print(f"Aviso: no se pudo descargar el modelo juez: {exc}", file=sys.stderr)
        judge_client.close()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
