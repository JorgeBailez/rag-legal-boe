"""Tests de corpus_loader: agregación de chunks/parents/documents procesados (offline, sin red)."""

import json
from pathlib import Path

from src.embeddings.corpus_loader import load_processed_corpus, load_readiness


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_load_processed_corpus_aggregates(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "chunks" / "norm_a.json", {"chunks": [{"chunk_id": "a1"}, {"chunk_id": "a2"}]}
    )
    _write_json(tmp_path / "chunks" / "norm_b.json", {"chunks": [{"chunk_id": "b1"}]})
    _write_json(tmp_path / "parents" / "norm_a.json", {"parents": [{"parent_id": "pa"}]})
    _write_json(tmp_path / "parents" / "norm_b.json", {"parents": [{"parent_id": "pb"}]})
    _write_json(tmp_path / "documents" / "norm_a.json", {"document_id": "A"})
    _write_json(tmp_path / "documents" / "norm_b.json", {"document_id": "B"})

    corpus = load_processed_corpus(tmp_path)

    assert len(corpus["chunks"]) == 3  # se concatenan los chunks de todas las normas
    assert set(corpus["parents_by_id"]) == {"pa", "pb"}  # indexado por parent_id
    assert set(corpus["documents_by_id"]) == {"A", "B"}  # indexado por document_id
    assert corpus["n_norms"] == 2


def test_load_processed_corpus_empty_dir(tmp_path: Path) -> None:
    corpus = load_processed_corpus(tmp_path)
    assert corpus == {"chunks": [], "parents_by_id": {}, "documents_by_id": {}, "n_norms": 0}


def test_load_readiness_missing_returns_none(tmp_path: Path) -> None:
    assert load_readiness(tmp_path / "no_existe.json") is None


def test_load_readiness_reads_field(tmp_path: Path) -> None:
    report = tmp_path / "audit.json"
    _write_json(report, {"pre_embedding_readiness": {"ready": True}})
    assert load_readiness(report) == {"ready": True}
