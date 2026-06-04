"""Tests de los escritores de reportes densos (smoke / benchmark)."""

import csv
import json

from src.evaluation.reports import (
    new_run_id,
    write_benchmark_report,
    write_smoke_report,
)


def test_new_run_id_format() -> None:
    rid = new_run_id("smoke", fingerprint="abcdef1234")
    assert rid.startswith("smoke_")
    assert rid.endswith("_abcdef12")
    # sin aliases mutables
    assert "latest" not in rid and "current" not in rid


def test_write_smoke_report(tmp_path) -> None:
    rows = [
        {"model_alias": "bge-m3", "embedding_dimension": 1024, "doc_throughput_per_s": 7.9},
        {"model_alias": "e5-large", "embedding_dimension": 1024, "warning": "x"},
    ]
    out = write_smoke_report(
        "smoke_X", meta={"device": "cpu", "threads": 8}, model_rows=rows, reports_root=tmp_path
    )
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["smoke_test_id"] == "smoke_X"
    assert report["kind"] == "smoke_test"
    assert len(report["models"]) == 2
    with (out / "models.csv").open(encoding="utf-8") as fh:
        csv_rows = list(csv.DictReader(fh))
    assert len(csv_rows) == 2


def test_write_benchmark_report(tmp_path) -> None:
    out = write_benchmark_report(
        "bench_X",
        summary={"split": "development", "primary_metric": "ParentnDCG@10"},
        metrics_rows=[{"bundle_id": "b1", "ParentnDCG@10": 0.7}],
        query_results=[{"bundle_id": "b1", "query_id": "q1", "metrics": {"ParentHit@1": 1.0}}],
        context_results=[],
        reports_root=tmp_path,
    )
    for fname in ("report.json", "metrics.csv", "query_results.jsonl", "context_results.jsonl"):
        assert (out / fname).is_file()
    qr = (out / "query_results.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(qr[0])["query_id"] == "q1"
    assert (out / "context_results.jsonl").read_text(encoding="utf-8") == ""
