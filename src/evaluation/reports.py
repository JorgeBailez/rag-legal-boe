"""Escritura de reportes densos regenerables (fuera de Git): smoke tests y benchmarks.

Layout:

    data/processed/reports/dense/
    ├── smoke_tests/<smoke_test_id>/{report.json, models.csv}
    └── benchmarks/<benchmark_id>/{report.json, metrics.csv, query_results.jsonl,
                                   context_results.jsonl}

`report.json` = resumen humano · `metrics.csv` = tabla analítica · `query_results.jsonl` = detalle
por consulta · `context_results.jsonl` = solo finalistas/ablaciones. No se duplican chunks ni
parents completos dentro de los reportes.
"""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

REPORTS_ROOT = Path("data/processed/reports/dense")
GENERATION_REPORTS_ROOT = Path("data/processed/reports/generation")


def new_run_id(prefix: str, fingerprint: str = "") -> str:
    """Id de ejecución estable y legible: `<prefix>_<timestamp>[_<fp8>]` (no alias mutable)."""
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"_{fingerprint[:8]}" if fingerprint else ""
    return f"{prefix}_{ts}{suffix}"


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for r in rows for k in r})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    lines = [json.dumps(r, ensure_ascii=False) for r in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_smoke_report(
    smoke_test_id: str,
    *,
    meta: dict,
    model_rows: list[dict],
    reports_root: Path = REPORTS_ROOT,
) -> Path:
    """Escribe `report.json` (resumen + meta) y `models.csv` (tabla por modelo)."""
    out_dir = Path(reports_root) / "smoke_tests" / smoke_test_id
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"smoke_test_id": smoke_test_id, "kind": "smoke_test", **meta, "models": model_rows}
    _write_json(out_dir / "report.json", report)
    _write_csv(out_dir / "models.csv", model_rows)
    return out_dir


def write_benchmark_report(
    benchmark_id: str,
    *,
    summary: dict,
    metrics_rows: list[dict],
    query_results: list[dict],
    context_results: list[dict],
    reports_root: Path = REPORTS_ROOT,
) -> Path:
    """Escribe report.json (resumen), metrics.csv, query_results.jsonl y context_results.jsonl."""
    out_dir = Path(reports_root) / "benchmarks" / benchmark_id
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        out_dir / "report.json", {"benchmark_id": benchmark_id, "kind": "benchmark", **summary}
    )
    _write_csv(out_dir / "metrics.csv", metrics_rows)
    _write_jsonl(out_dir / "query_results.jsonl", query_results)
    _write_jsonl(out_dir / "context_results.jsonl", context_results)
    return out_dir


def write_generation_report(
    run_id: str,
    *,
    summary: dict,
    config: dict,
    per_query: list[dict],
    metrics_rows: list[dict],
    aggregate: dict,
    judge_agreement: dict | None = None,
    reports_root: Path = GENERATION_REPORTS_ROOT,
) -> Path:
    """Escribe el report de evaluación de generación bajo `<reports_root>/<run_id>/`.

    Ficheros: report.json (resumen+agregado), config.json (config canónica + fingerprint para
    auditoría), per_query.jsonl (detalle), metrics.csv (escalares por query), aggregate.json y
    judge_agreement.json (si se validó el juez contra humano).
    """
    out_dir = Path(reports_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        out_dir / "report.json",
        {"generation_run_id": run_id, "kind": "generation", **summary, "aggregate": aggregate},
    )
    _write_json(out_dir / "config.json", config)
    _write_json(out_dir / "aggregate.json", aggregate)
    _write_jsonl(out_dir / "per_query.jsonl", per_query)
    _write_csv(out_dir / "metrics.csv", metrics_rows)
    if judge_agreement is not None:
        _write_json(out_dir / "judge_agreement.json", judge_agreement)
    return out_dir
