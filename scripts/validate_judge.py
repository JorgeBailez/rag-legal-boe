"""Validación del LLM-juez contra anotación humana: % de acuerdo + Cohen's κ.

Las métricas con juez (fidelidad L3, corrección L5) NO son fiables hasta validarlas contra una
muestra anotada a mano (lección de ALCE). Flujo en dos pasos:

1) `--scaffold <out.jsonl>` — genera una plantilla de anotación a partir de un report CON juez: una
   fila por respuesta con la pregunta, la respuesta generada, la referencia del gold, el **bloque de
   evidencias** que vio el generador (para anotar fidelidad sin adivinar) y el veredicto del juez,
   más campos `human_*` vacíos para que tú rellenes a mano (~30–50 casos).

2) (validar) `--annotations <plantilla_rellenada.jsonl>` — compara tu anotación con el veredicto del
   juez y calcula κ por dimensión (corrección categórica correct/partial/incorrect; fidelidad
   binaria faithful/unfaithful), lista los desacuerdos y escribe `judge_agreement.json` en el report
   (lo consume la §4 del notebook 06). κ < ~0.6 ⇒ tratar L3/L5 como provisionales o cambiar de juez.

Uso:
    uv run python scripts/validate_judge.py --report data/processed/reports/generation/<run_id> \
        --scaffold anotacion_juez.jsonl
    # ... rellenas human_correctness (correct|partial|incorrect) y human_faithful (true|false) ...
    uv run python scripts/validate_judge.py --report data/processed/reports/generation/<run_id> \
        --annotations anotacion_juez.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.dataset import (  # noqa: E402
    ANSWER_KEYS_FILE,
    DATASET_DIR,
    QUESTIONS_FILE,
    load_jsonl,
)
from src.evaluation.judge import judge_agreement  # noqa: E402

CORRECTNESS_LABELS = ("correct", "partial", "incorrect")
# Orden ORDINAL (de menor a mayor) para el κ lineal-ponderado de corrección.
CORRECTNESS_ORDER = ("incorrect", "partial", "correct")
FAITHFULNESS_ORDER = ("unfaithful", "faithful")


def correctness_label_from_score(score: float | None) -> str | None:
    """Mapea el score de corrección del juez (1.0/0.5/0.0) a su etiqueta categórica."""
    if not isinstance(score, int | float):
        return None
    if score >= 0.75:
        return "correct"
    if score >= 0.25:
        return "partial"
    return "incorrect"


def is_faithful(faithfulness: float | None, threshold: float = 1.0) -> bool | None:
    """True si la respuesta es plenamente fiel (sin afirmaciones no soportadas); None si N/A."""
    if not isinstance(faithfulness, int | float):
        return None
    return faithfulness >= threshold


def _agreement(
    pairs: list[tuple[str, str]], *, ordered_labels: tuple[str, ...] | None = None
) -> dict:
    human = [h for h, _ in pairs]
    judge = [j for _, j in pairs]
    out = judge_agreement(human, judge, ordered_labels=ordered_labels)
    out["disagreements"] = [{"human": h, "judge": j} for h, j in pairs if h != j]
    return out


def compute_agreement(
    human_rows: list[dict],
    per_query: list[dict],
    *,
    faithfulness_threshold: float = 1.0,
) -> dict:
    """Empareja anotación humana y veredicto del juez por query_id y calcula κ por dimensión."""
    judge_by_qid = {r["query_id"]: r for r in per_query}

    corr_pairs: list[tuple[str, str]] = []
    corr_disagree: list[dict] = []
    faith_pairs: list[tuple[str, str]] = []
    faith_disagree: list[dict] = []

    for r in human_rows:
        qid = r.get("query_id")
        jrec = judge_by_qid.get(qid)
        if jrec is None:
            continue
        hc = r.get("human_correctness")
        if hc in CORRECTNESS_LABELS:
            jc = correctness_label_from_score(jrec.get("correctness"))
            if jc is not None:
                corr_pairs.append((hc, jc))
                if hc != jc:
                    corr_disagree.append({"query_id": qid, "human": hc, "judge": jc})
        hf = r.get("human_faithful")
        jf_bool = is_faithful(jrec.get("faithfulness"), faithfulness_threshold)
        if isinstance(hf, bool) and jf_bool is not None:
            hl = "faithful" if hf else "unfaithful"
            jl = "faithful" if jf_bool else "unfaithful"
            faith_pairs.append((hl, jl))
            if hl != jl:
                faith_disagree.append({"query_id": qid, "human": hl, "judge": jl})

    correctness = {
        **_agreement(corr_pairs, ordered_labels=CORRECTNESS_ORDER),
        "disagreements": corr_disagree,
    }
    faithfulness = {
        **_agreement(faith_pairs, ordered_labels=FAITHFULNESS_ORDER),
        "disagreements": faith_disagree,
    }
    return {"correctness": correctness, "faithfulness": faithfulness}


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """IC de Wilson para una proporción k/n (más honesto que el normal con n pequeño)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def human_summary(human_rows: list[dict]) -> dict:
    """Tasa humana de fidelidad y corrección con IC de Wilson (medida PRIMARIA sin juez).

    No es acuerdo con nadie: es la proporción observada en la anotación (p. ej. el held-out de
    test), reportada con su incertidumbre. Anotador único ⇒ sin κ inter-humano; solo tasa + IC.
    """
    faith = [r["human_faithful"] for r in human_rows if isinstance(r.get("human_faithful"), bool)]
    corr = [
        r["human_correctness"]
        for r in human_rows
        if r.get("human_correctness") in CORRECTNESS_LABELS
    ]
    n_f, k_f = len(faith), sum(1 for x in faith if x)
    n_c = len(corr)
    dist = {lab: corr.count(lab) for lab in CORRECTNESS_LABELS}
    return {
        "faithfulness": {
            "n": n_f,
            "faithful": k_f,
            "rate": (k_f / n_f if n_f else None),
            "ci95": _wilson_ci(k_f, n_f),
        },
        "correctness": {
            "n": n_c,
            "distribution": dist,
            "correct_rate": (dist["correct"] / n_c if n_c else None),
            "ci95_correct": _wilson_ci(dist["correct"], n_c),
        },
    }


def _run_summary(annotations_path: Path) -> int:
    """Imprime y persiste el resumen de la anotación humana (tasa + IC), sin juez ni report."""
    human_rows = load_jsonl(annotations_path)
    if not human_rows:
        print(f"No hay anotaciones en {annotations_path}.", file=sys.stderr)
        return 2
    summary = human_summary(human_rows)
    f, c = summary["faithfulness"], summary["correctness"]
    print(f"Resumen de anotación humana ({annotations_path.name})")
    if f["n"]:
        lo, hi = f["ci95"]
        print(
            f"  Fidelidad (L3): {f['faithful']}/{f['n']} fieles = {f['rate']:.0%} "
            f"· IC95 Wilson=[{lo:.0%}, {hi:.0%}]"
        )
    else:
        print("  Fidelidad (L3): sin anotar.")
    if c["n"]:
        lo, hi = c["ci95_correct"]
        d = c["distribution"]
        print(
            f"  Corrección (L5): {d['correct']} correct / {d['partial']} partial / "
            f"{d['incorrect']} incorrect (n={c['n']}) · correct-rate {c['correct_rate']:.0%} "
            f"· IC95=[{lo:.0%}, {hi:.0%}]"
        )
    else:
        print("  Corrección (L5): sin anotar.")
    out = annotations_path.with_name(annotations_path.stem + "_summary.json")
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Escrito {out}")
    return 0


def scaffold_rows(
    per_query: list[dict],
    q_text: dict[str, str],
    ref_by_qid: dict[str, str],
    *,
    faithfulness_threshold: float = 1.0,
) -> list[dict]:
    """Plantilla de anotación (solo respondidas): veredicto del juez + campos human_* vacíos."""
    rows = []
    for r in per_query:
        if not r.get("answered"):
            continue
        qid = r["query_id"]
        rows.append(
            {
                "query_id": qid,
                "question": q_text.get(qid, ""),
                "answer_text": r.get("answer_text", ""),
                "reference_answer": ref_by_qid.get(qid, ""),
                # Evidencia que vio el generador: imprescindible para anotar fidelidad (L3) sin
                # adivinar (afirmación-contra-evidencia). La rellena el runner con --judge-model.
                "evidences_block": r.get("evidences_block") or "",
                "judge_correctness": correctness_label_from_score(r.get("correctness")),
                "judge_faithful": is_faithful(r.get("faithfulness"), faithfulness_threshold),
                "human_correctness": "",  # rellenar: correct | partial | incorrect
                "human_faithful": None,  # rellenar: true | false
                "notes": "",
            }
        )
    return rows


def _fmt_kappa(value: float | None) -> str:
    return f"{value:.3f}" if isinstance(value, int | float) else "n/a"


def _fmt_ci(ci: dict | None) -> str:
    if not ci:
        return ""
    return f" IC{int(ci['level'] * 100)}%=[{ci['lo']:.2f}, {ci['hi']:.2f}]"


def _print_dim(name: str, dim: dict) -> None:
    n = dim.get("n", 0)
    if not n:
        print(f"  {name}: sin pares anotados (rellena human_* en la plantilla).")
        return
    pa = dim.get("percent_agreement")
    k = dim.get("cohens_kappa")
    wk = dim.get("weighted_kappa")
    print(f"  {name}: n={n} · acuerdo={pa:.0%}")
    print(f"      κ nominal={_fmt_kappa(k)}{_fmt_ci(dim.get('cohens_kappa_ci'))}")
    if wk is not None:
        print(f"      κ lineal-ponderado={_fmt_kappa(wk)}{_fmt_ci(dim.get('weighted_kappa_ci'))}")
    ac1 = dim.get("gwet_ac1")  # robusto a la prevalencia (ver judge.py)
    print(f"      AC1 (Gwet)={_fmt_kappa(ac1)}{_fmt_ci(dim.get('gwet_ac1_ci'))}")
    labels = dim.get("labels")
    matrix = dim.get("confusion_matrix")
    if labels and matrix:
        width = max(len(lab) for lab in labels)
        print(f"      confusión (filas=humano, columnas=juez) · orden [{', '.join(labels)}]:")
        for i, row in enumerate(matrix):
            print(f"        {labels[i]:>{width}} | {row}")
    for d in dim.get("disagreements", []):
        print(f"      [DIFF] {d['query_id']}: humano={d['human']} vs juez={d['judge']}")


def main() -> int:
    # Algunas consolas de Windows no imprimen bien κ; forzamos UTF-8 si se puede.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description="Validación del LLM-juez (acuerdo + Cohen's κ).")
    parser.add_argument("--report", help="directorio del report (para --scaffold / --annotations).")
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--scaffold", help="genera la plantilla de anotación en esta ruta y sale.")
    parser.add_argument("--annotations", help="plantilla rellenada para validar contra el juez.")
    parser.add_argument(
        "--summary",
        help="anotación humana (JSONL): resume tasa de fidelidad/corrección + IC, sin juez.",
    )
    parser.add_argument("--faithfulness-threshold", type=float, default=1.0)
    args = parser.parse_args()

    # --summary NO necesita report ni juez: mide solo la anotación humana (medida primaria de
    # L3/L5 cuando el juez no se usa, p. ej. en el held-out de test).
    if args.summary:
        return _run_summary(Path(args.summary))

    if not args.report:
        print("Indica --report (para --scaffold/--annotations) o --summary.", file=sys.stderr)
        return 2
    report_dir = Path(args.report)
    per_query = load_jsonl(report_dir / "per_query.jsonl")
    if not per_query:
        print(f"No hay per_query.jsonl en {report_dir}.", file=sys.stderr)
        return 2

    if args.scaffold:
        dataset_dir = Path(args.dataset_dir)
        q_text = {q["query_id"]: q["query"] for q in load_jsonl(dataset_dir / QUESTIONS_FILE)}
        ref_by_qid = {
            a["query_id"]: a.get("reference_answer", "")
            for a in load_jsonl(dataset_dir / ANSWER_KEYS_FILE)
        }
        rows = scaffold_rows(
            per_query, q_text, ref_by_qid, faithfulness_threshold=args.faithfulness_threshold
        )
        out_path = Path(args.scaffold)
        out_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
        )
        missing_answer = sum(1 for r in rows if not r["answer_text"])
        missing_evidence = sum(1 for r in rows if not r["evidences_block"])
        print(f"Plantilla escrita: {out_path} ({len(rows)} respuestas a anotar).")
        if missing_answer:
            print(
                f"[WARN] {missing_answer} filas sin answer_text (report antiguo): "
                "vuelve a generar el report con el runner actual para tener "
                "la respuesta a la vista."
            )
        if missing_evidence:
            print(
                f"[WARN] {missing_evidence} filas sin evidences_block: L3 (fidelidad) no anotable; "
                "regenera el report con el runner actual (guarda la evidencia aunque no haya juez)."
            )
        print("Rellena los campos human_correctness y human_faithful de cada fila.")
        return 0

    if not args.annotations:
        print(
            "Indica --scaffold para crear la plantilla, --annotations para validar contra el "
            "juez, o --summary para resumir la anotación humana.",
            file=sys.stderr,
        )
        return 2

    # La validación contra el juez SÍ requiere veredictos en el report.
    judged = [
        r
        for r in per_query
        if r.get("faithfulness") is not None or r.get("correctness") is not None
    ]
    if not judged:
        print(
            "El report no tiene veredictos del juez (¿se corrió con --judge-model?).",
            file=sys.stderr,
        )
        return 2

    human_rows = load_jsonl(Path(args.annotations))
    result = compute_agreement(
        human_rows, per_query, faithfulness_threshold=args.faithfulness_threshold
    )
    print(f"Validación del juez (report: {report_dir.name})")
    _print_dim("Corrección (L5)", result["correctness"])
    _print_dim("Fidelidad  (L3)", result["faithfulness"])

    corr = result["correctness"]
    payload = {
        "n": corr.get("n", 0),
        "percent_agreement": corr.get("percent_agreement"),
        "cohens_kappa": corr.get("cohens_kappa"),
        "weighted_kappa": corr.get("weighted_kappa"),
        "gwet_ac1": corr.get("gwet_ac1"),
        "cohens_kappa_ci": corr.get("cohens_kappa_ci"),
        "weighted_kappa_ci": corr.get("weighted_kappa_ci"),
        "gwet_ac1_ci": corr.get("gwet_ac1_ci"),
        "confusion_matrix": corr.get("confusion_matrix"),
        "labels": corr.get("labels"),
        "primary_dimension": "correctness",
        "primary_kappa": "weighted_kappa",
        "primary_metric_imbalanced": "gwet_ac1",
        "by_dimension": result,
    }
    (report_dir / "judge_agreement.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nEscrito {report_dir / 'judge_agreement.json'} (lo lee el notebook 06 §4).")
    # En escala ordinal el κ de referencia es el ponderado; cae al nominal si no aplica.
    k = corr.get("weighted_kappa")
    if k is None:
        k = corr.get("cohens_kappa")
    ac1 = corr.get("gwet_ac1")
    if k is not None and k < 0.6:
        if ac1 is not None and ac1 >= 0.6:
            print(
                f"ℹ κ < 0.6 pero AC1={ac1:.2f} ≥ 0.6: probable PARADOJA DE PREVALENCIA "
                "(clases desbalanceadas). Reporta AC1 como métrica primaria y el % de acuerdo."
            )
        else:
            print(
                "[WARN] κ < 0.6 y AC1 < 0.6: trata L3/L5 como PROVISIONALES "
                "o cambia de modelo juez."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
