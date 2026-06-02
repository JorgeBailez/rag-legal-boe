"""Invariantes del pipeline procesado v2 (raw → bundle → chunks), regenerado desde raw.

Comprueba, sobre la representación vigente (sin comparar con arquitecturas retiradas), que el
texto vive solo en parents, que `retrieval_text` antepone el contexto de la norma, que los
bloques `(Sin contenido)` se indexan y que la cuarentena no produce parent ni chunks. Si el raw
no está disponible, se omite.
"""

import json
from pathlib import Path

import pytest

from src.boe.parser import build_processed_bundle
from src.preprocessing.chunker import create_chunks

RAW_DIR = Path("data/raw/boe")
MANIFEST_DIR = Path("data/manifests")
SEED = Path("data/corpus/seed_corpus.json")


def _norm_ids() -> list[str]:
    if not SEED.is_file():
        return []
    return [n["norm_id"] for n in json.loads(SEED.read_text(encoding="utf-8"))["norms"]]


@pytest.fixture(params=_norm_ids() or ["__skip__"])
def processed(request):
    nid = request.param
    if nid == "__skip__" or not (RAW_DIR / nid / "texto.xml").is_file():
        pytest.skip("raw no disponible")
    bundle = build_processed_bundle(nid, RAW_DIR, MANIFEST_DIR / f"{nid}.json")
    chunks = create_chunks(bundle.document, bundle.parents)
    return nid, bundle, chunks


def test_text_lives_only_in_parents(processed) -> None:
    _, bundle, chunks = processed
    # Ningún descriptor lleva texto vigente; ningún chunk lleva parent_text.
    assert all("text" not in b and "latest_version" not in b for b in bundle.document["blocks"])
    assert all("parent_text" not in c for c in chunks["chunks"])


def test_modification_notes_only_in_history(processed) -> None:
    _, bundle, _ = processed
    assert all("modification_notes" not in p for p in bundle.parents["parents"])
    assert all("modification_notes" in h for h in bundle.history["blocks"])


def test_retrieval_text_prefixes_norm_context(processed) -> None:
    # El texto del chunk vive en parents y retrieval_text antepone el contexto de la norma.
    _, bundle, chunks = processed
    parents_by_id = {p["parent_id"]: p for p in bundle.parents["parents"]}
    short_title = bundle.document["metadata"]["short_title"]
    for c in chunks["chunks"]:
        assert c["text"] in parents_by_id[c["parent_id"]]["text"]
        assert c["retrieval_text"].startswith(short_title)


def test_without_content_block_indexable(processed) -> None:
    _, bundle, chunks = processed
    for b in bundle.document["blocks"]:
        if b["is_without_content"]:
            assert b["indexable"] is True
            assert any(c["block_id"] == b["block_id"] for c in chunks["chunks"])


def test_quarantine_has_no_parent_no_chunks(processed) -> None:
    _, bundle, chunks = processed
    parent_blocks = {p["block_id"] for p in bundle.parents["parents"]}
    chunk_blocks = {c["block_id"] for c in chunks["chunks"]}
    for b in bundle.document["blocks"]:
        if b["temporal_status"] != "resolved":
            assert b["block_id"] not in parent_blocks
            assert b["block_id"] not in chunk_blocks
