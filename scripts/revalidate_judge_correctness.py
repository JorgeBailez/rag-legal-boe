"""Re-valida la CORRECCIÓN del juez con el prompt ACTUAL sobre una anotación ya hecha.

Aísla el efecto del PROMPT del juez: re-ejecuta `judge_correctness` (prompt vigente) sobre los
mismos `(pregunta, respuesta, referencia)` ya anotados a mano y compara el NUEVO veredicto con la
etiqueta humana —sin regenerar respuestas ni tocar el gold—. Responde "¿el prompt nuevo mejora el
acuerdo juez↔humano?" sin gastar horas de anotación nuevas. Reporta también cuántos veredictos
cambian respecto al juez ANTERIOR guardado en la anotación (`judge_correctness`), para ver en qué
dirección movió el prompt (p. ej. si dejó de marcar `partial` lo que el humano da por `correct`).

Necesita Ollama con el modelo juez (dslab01). Determinista (temperatura 0, seed fija). La anotación
de entrada debe traer, por fila: `question`, `answer_text`, `reference_answer`, `human_correctness`
(correct|partial|incorrect) y, opcionalmente, `judge_correctness` (el veredicto del juez anterior).

Uso:
    uv run python scripts/revalidate_judge_correctness.py --annotations anotacion_juez.jsonl \
        --judge-model gemma3:12b --out anotacion_juez_rejuzgada.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import get_settings  # noqa: E402
from src.core.exceptions import RagLegalBoeError  # noqa: E402
from src.evaluation.judge import LlmJudge, judge_agreement  # noqa: E402
from src.generation.ollama_client import OllamaClient  # noqa: E402

CORRECTNESS_LABELS = ("correct", "partial", "incorrect")
# Orden ORDINAL (de menor a mayor) para el κ lineal-ponderado de corrección.
CORRECTNESS_ORDER = ("incorrect", "partial", "correct")


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if isinstance(value, int | float) else "n/a"


def _fmt_ci(ci: dict | None) -> str:
    if not ci:
        return ""
    return f" IC{int(ci['level'] * 100)}%=[{ci['lo']:.2f}, {ci['hi']:.2f}]"


def _print_agreement(title: str, pairs: list[tuple[str, str]]) -> None:
    if not pairs:
        print(f"{title}: sin pares.")
        return
    human = [h for h, _ in pairs]
    judge = [j for _, j in pairs]
    out = judge_agreement(human, judge, ordered_labels=CORRECTNESS_ORDER)
    print(f"{title}: n={out['n']} · acuerdo={out['percent_agreement']:.0%}")
    print(f"    κ nominal={_fmt(out['cohens_kappa'])}{_fmt_ci(out.get('cohens_kappa_ci'))}")
    print(
        f"    κ lineal-ponderado={_fmt(out['weighted_kappa'])}"
        f"{_fmt_ci(out.get('weighted_kappa_ci'))}"
    )
    print(f"    AC1 (Gwet)={_fmt(out['gwet_ac1'])}{_fmt_ci(out.get('gwet_ac1_ci'))}")
    labels = out["labels"]
    print(f"    confusión (filas=humano, columnas=juez) · orden [{', '.join(labels)}]:")
    width = max(len(lab) for lab in labels)
    for i, row in enumerate(out["confusion_matrix"]):
        print(f"        {labels[i]:>{width}} | {row}")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(
        description="Re-valida la corrección del juez (prompt actual) contra una anotación."
    )
    parser.add_argument("--annotations", required=True, help="jsonl con human_correctness, etc.")
    parser.add_argument("--judge-model", default=None, help="modelo juez; fallback JUDGE_MODEL.")
    parser.add_argument("--out", default=None, help="vuelca las filas con el nuevo veredicto.")
    parser.add_argument("--limit", type=int, default=None, help="re-juzga solo N filas.")
    args = parser.parse_args()

    settings = get_settings()
    judge_model = args.judge_model if args.judge_model is not None else settings.judge_model
    if not judge_model:
        print("Falta el modelo juez. Indica --judge-model o JUDGE_MODEL.", file=sys.stderr)
        return 2

    rows = [
        json.loads(line)
        for line in Path(args.annotations).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    annotated = [r for r in rows if r.get("human_correctness") in CORRECTNESS_LABELS]
    if args.limit is not None:
        annotated = annotated[: args.limit]
    if not annotated:
        print("No hay filas con human_correctness anotado.", file=sys.stderr)
        return 1

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

    new_pairs: list[tuple[str, str]] = []
    old_pairs: list[tuple[str, str]] = []
    flips: list[dict] = []
    errors = 0
    out_rows: list[dict] = []
    try:
        for i, r in enumerate(annotated, start=1):
            qid = r.get("query_id", f"#{i}")
            human = r["human_correctness"]
            question = r.get("question", "")
            answer = r.get("answer_text", "")
            reference = r.get("reference_answer", "")
            if not answer or not reference:
                print(f"  [{i}/{len(annotated)}] {qid}: sin answer/reference, se omite.")
                continue
            try:
                verdict, _ = judge.judge_correctness(
                    question=question, answer=answer, reference=reference
                )
                new = verdict.verdict
            except RagLegalBoeError as exc:
                errors += 1
                print(f"  [{i}/{len(annotated)}] {qid}: ⚠ juez falló ({exc}); se omite.")
                continue
            new_pairs.append((human, new))
            old = r.get("judge_correctness")
            if old in CORRECTNESS_LABELS:
                old_pairs.append((human, old))
            mark = "=" if old == new else "→"
            print(f"  [{i}/{len(annotated)}] {qid}: humano={human} · juez {old}{mark}{new}")
            if old != new:
                flips.append({"query_id": qid, "old": old, "new": new, "human": human})
            out_rows.append({**r, "judge_correctness_new": new})
    finally:
        judge_client.close()

    print(f"\nRe-juzgadas {len(new_pairs)} filas (errores del juez: {errors}).")
    print("\n=== CORRECCIÓN — prompt ACTUAL (re-juzgado) vs humano ===")
    _print_agreement("  nuevo", new_pairs)
    if old_pairs:
        print("\n=== CORRECCIÓN — juez ANTERIOR (guardado) vs humano (mismos casos) ===")
        _print_agreement("  anterior", old_pairs)
    if flips:
        print(f"\nVeredictos que CAMBIARON con el prompt nuevo ({len(flips)}):")
        for f in flips:
            print(f"  {f['query_id']}: {f['old']} → {f['new']}  (humano={f['human']})")

    if args.out:
        Path(args.out).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n",
            encoding="utf-8",
        )
        print(f"\nVolcado con el nuevo veredicto: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
