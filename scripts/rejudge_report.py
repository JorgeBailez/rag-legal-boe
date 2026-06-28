"""Re-juzga un report de generación con un juez/prompt NUEVO, sin regenerar (calibración del juez).

Reutiliza `answer_text` + `evidences_block` del report previo y re-ejecuta SOLO al juez (fidelidad
L3 + corrección L5) con el prompt que indique `--judge-prompts-dir`. Escribe un report nuevo (solo
`per_query.jsonl` + `config.json`) que `validate_judge.py --annotations` consume para recomputar
κ/AC1 contra la MISMA anotación humana. Aísla el efecto del prompt del juez (no recupera ni genera).

Necesita Ollama con el modelo juez (servidor/Colab; en local con `uv run`).

Uso:
    uv run python scripts/rejudge_report.py --report data/processed/reports/generation/<run_id> \
        --judge-model gemma3:12b --judge-prompts-dir prompts/judge_v3 \
        --out-report data/processed/reports/generation/<run_id>__judgev3
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
from src.evaluation.dataset import (  # noqa: E402
    ANSWER_KEYS_FILE,
    DATASET_DIR,
    QUESTIONS_FILE,
    load_jsonl,
)
from src.evaluation.generation_eval import rejudge_report  # noqa: E402
from src.evaluation.judge import LlmJudge  # noqa: E402
from src.generation.ollama_client import OllamaClient  # noqa: E402


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description="Re-juzga un report con un juez/prompt nuevo.")
    parser.add_argument("--report", required=True, help="report previo (lee per_query.jsonl).")
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--judge-model", default=None, help="modelo juez; fallback JUDGE_MODEL.")
    parser.add_argument(
        "--judge-prompts-dir",
        default=None,
        help="directorio de prompts del juez (judge_faithfulness.txt + judge_correctness.txt), "
        "p. ej. prompts/judge_v3. Por defecto, los de prompts/.",
    )
    parser.add_argument("--out-report", required=True, help="report nuevo a escribir.")
    parser.add_argument("--limit", type=int, default=None, help="re-juzga solo N respuestas.")
    args = parser.parse_args()

    settings = get_settings()
    judge_model = args.judge_model if args.judge_model is not None else settings.judge_model
    if not judge_model:
        print("Falta el modelo juez. Indica --judge-model o JUDGE_MODEL.", file=sys.stderr)
        return 2

    report_dir = Path(args.report)
    prior_per_query = load_jsonl(report_dir / "per_query.jsonl")
    if not prior_per_query:
        print(f"No hay per_query.jsonl en {report_dir}.", file=sys.stderr)
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

    answered = [r for r in prior_per_query if r.get("answered")]
    total = len(answered) if args.limit is None else min(args.limit, len(answered))
    bar = tqdm(total=total, desc="re-juez", unit="q")

    def on_progress(ev: dict) -> None:
        if ev.get("event") == "judging":
            bar.set_postfix_str(f"{ev.get('query_id')}:{ev.get('phase')}…")
        elif ev.get("event") == "done":
            bar.set_postfix_str("")
            bar.update(1)

    try:
        try:
            new_per_query = rejudge_report(
                prior_per_query=prior_per_query,
                answer_keys=answer_keys,
                questions=questions,
                judge=judge,
                limit=args.limit,
                on_progress=on_progress,
            )
        finally:
            bar.close()
    except RagLegalBoeError as exc:
        print(f"Fallo de re-juicio: {exc}", file=sys.stderr)
        judge_client.close()
        return 1
    judge_client.close()

    out_dir = Path(args.out_report)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_query.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in new_per_query) + "\n",
        encoding="utf-8",
    )
    config = {
        "kind": "rejudge",
        "source_report": report_dir.name,
        "dataset_dir": str(dataset_dir),
        "judge_model": judge_model,
        "judge_prompts_dir": args.judge_prompts_dir or "(default)",
        "judge_num_ctx": settings.judge_num_ctx,
        "judge_num_predict": settings.judge_num_predict,
        "n_rejudged": len(new_per_query),
    }
    (out_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    n_err = sum(1 for r in new_per_query if r.get("judge_error"))
    faith = _mean(
        [r["faithfulness"] for r in new_per_query if isinstance(r["faithfulness"], float)]
    )
    corr = _mean([r["correctness"] for r in new_per_query if isinstance(r["correctness"], float)])
    print(f"\nRe-juzgadas {len(new_per_query)} respuestas → {out_dir}")
    print(
        f"  faithfulness media={_fmt(faith)} · correctness media={_fmt(corr)} · juez_error={n_err}"
    )
    print(
        "Valida κ/AC1 contra tu anotación:\n"
        f"  python scripts/validate_judge.py --report {out_dir} "
        f"--dataset-dir {dataset_dir} --annotations <tu_anotacion.jsonl>"
    )
    return 0


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if isinstance(value, int | float) else "n/a"


if __name__ == "__main__":
    raise SystemExit(main())
