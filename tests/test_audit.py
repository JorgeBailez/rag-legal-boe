"""Tests del módulo de auditoría v2 (joins, checks de contrato, integridad temporal/raw)."""

import copy
import hashlib
import json
from pathlib import Path

from src.quality.corpus_audit import (
    analyze_overlap,
    check_chunks,
    check_document,
    check_history,
    check_parents,
    check_relational,
    compute_readiness,
    efficiency_metrics,
    join_norm,
    oversized_rows,
    raw_integrity,
    temporal_integrity,
    verify_manifest,
)

DOC_ID = "BOE-A-2015-10565"
HTML = f"https://www.boe.es/buscar/act.php?id={DOC_ID}"
GEN = {"generated_at": "2026-06-02T00:00:00Z", "generator": "test"}


def _doc() -> dict:
    return {
        "schema_version": "boe_legal_document_v2",
        "document_id": DOC_ID,
        "source": {"name": "BOE", "manifest_ref": f"data/manifests/{DOC_ID}.json"},
        "metadata": {
            "title": "Ley X",
            "short_title": "Ley X",
            "identifier": DOC_ID,
            "html_url": HTML,
            "rank": {"code": "1300", "label": "Ley"},
            "scope": {"code": "1", "label": "Estatal"},
            "publication_date": "2015-10-02",
        },
        "analysis": {
            "subjects": [{"code": "85", "label": "Régimen local"}],
            "notes": [],
            "references": {"previous": [], "next": []},
        },
        "blocks": [
            {
                "block_id": "ti",
                "parent_id": f"{DOC_ID}__ti",
                "order": 0,
                "block_type": "encabezado",
                "block_title": "TÍTULO I",
                "full_title": "TÍTULO I",
                "semantic_role": "structural_heading",
                "has_retrievable_body": False,
                "is_annex": False,
                "contains_table": False,
                "table_text_available": False,
                "contains_image": False,
                "content_status": "present",
                "is_without_content": False,
                "temporal_status": "resolved",
                "hierarchy": {
                    "book": None,
                    "title": "TÍTULO I",
                    "chapter": None,
                    "section": None,
                    "subsection": None,
                    "annex": None,
                },
                "indexable": False,
                "excluded_reason": "no_retrievable_body",
                "citation": {"label": "Ley X, tÍTULO I", "url": f"{HTML}#ti"},
            },
            {
                "block_id": "a1",
                "parent_id": f"{DOC_ID}__a1",
                "order": 1,
                "block_type": "precepto",
                "block_title": "Artículo 1",
                "full_title": "Artículo 1. Objeto.",
                "semantic_role": "precept",
                "has_retrievable_body": True,
                "is_annex": False,
                "contains_table": False,
                "table_text_available": False,
                "contains_image": False,
                "content_status": "present",
                "is_without_content": False,
                "temporal_status": "resolved",
                "hierarchy": {
                    "book": None,
                    "title": "TÍTULO I",
                    "chapter": None,
                    "section": None,
                    "subsection": None,
                    "annex": None,
                },
                "indexable": True,
                "excluded_reason": None,
                "citation": {"label": "Ley X, artículo 1", "url": f"{HTML}#a1"},
            },
        ],
        "generation_meta": GEN,
    }


A1_TEXT = "Artículo 1. Objeto.\n1. Primer apartado.\n2. Segundo apartado."
A1_PARAS = [
    {"order": 1, "class": "articulo", "text": "Artículo 1. Objeto."},
    {"order": 2, "class": "parrafo", "text": "1. Primer apartado."},
    {"order": 3, "class": "parrafo", "text": "2. Segundo apartado."},
]


def _history() -> dict:
    def rec(bid: str) -> dict:
        return {
            "block_id": bid,
            "versions": [
                {
                    "source_norm_id": DOC_ID,
                    "publication_date": "2015-10-02",
                    "validity_date": "2016-10-02",
                    "is_current": True,
                }
            ],
            "modification_notes": [],
            "temporal_resolution": {
                "status": "resolved",
                "selection_method": "index_date_exact_unique_match",
                "index_last_update_date": "2015-10-02",
                "selected_version_index": 0,
                "selected_publication_date": "2015-10-02",
                "selected_source_norm_id": DOC_ID,
                "candidate_versions": [],
                "max_publication_date": "2015-10-02",
                "warnings": [],
            },
            "temporal_quarantined": False,
            "index_title": None,
            "index_url": None,
            "index_last_update_date": "2015-10-02",
            "index_last_update_date_raw": "20151002",
            "warnings": [],
        }

    return {
        "schema_version": "boe_legal_history_v2",
        "document_id": DOC_ID,
        "blocks": [rec("ti"), rec("a1")],
        "generation_meta": GEN,
    }


def _parents() -> dict:
    return {
        "schema_version": "boe_legal_parents_v2",
        "document_id": DOC_ID,
        "parents": [
            {
                "parent_id": f"{DOC_ID}__ti",
                "document_id": DOC_ID,
                "block_id": "ti",
                "order": 0,
                "block_type": "encabezado",
                "title": "TÍTULO I",
                "full_title": "TÍTULO I",
                "semantic_role": "structural_heading",
                "text": "TÍTULO I",
                "paragraphs": [{"order": 1, "class": "titulo_num", "text": "TÍTULO I"}],
                "hierarchy": {
                    "book": None,
                    "title": "TÍTULO I",
                    "chapter": None,
                    "section": None,
                    "subsection": None,
                    "annex": None,
                },
                "citation": {"label": "Ley X, tÍTULO I", "url": f"{HTML}#ti"},
                "current_version": {
                    "source_norm_id": DOC_ID,
                    "publication_date": "2015-10-02",
                    "validity_date": "2016-10-02",
                },
                "is_annex": False,
                "contains_table": False,
                "table_text_available": False,
                "contains_image": False,
                "content_status": "present",
                "is_without_content": False,
            },
            {
                "parent_id": f"{DOC_ID}__a1",
                "document_id": DOC_ID,
                "block_id": "a1",
                "order": 1,
                "block_type": "precepto",
                "title": "Artículo 1",
                "full_title": "Artículo 1. Objeto.",
                "semantic_role": "precept",
                "text": A1_TEXT,
                "paragraphs": A1_PARAS,
                "hierarchy": {
                    "book": None,
                    "title": "TÍTULO I",
                    "chapter": None,
                    "section": None,
                    "subsection": None,
                    "annex": None,
                },
                "citation": {"label": "Ley X, artículo 1", "url": f"{HTML}#a1"},
                "current_version": {
                    "source_norm_id": DOC_ID,
                    "publication_date": "2015-10-02",
                    "validity_date": "2016-10-02",
                },
                "is_annex": False,
                "contains_table": False,
                "table_text_available": False,
                "contains_image": False,
                "content_status": "present",
                "is_without_content": False,
            },
        ],
        "generation_meta": GEN,
    }


def _chunk(text: str, index: int = 1, count: int = 1) -> dict:
    return {
        "chunk_id": f"{DOC_ID}__a1__c{index:03d}",
        "parent_id": f"{DOC_ID}__a1",
        "document_id": DOC_ID,
        "block_id": "a1",
        "position": {"index": index, "count_for_parent": count},
        "text": text,
        "retrieval_text": f"Ley X. Artículo 1. Objeto. {text}",
        "citation": {"label": "Ley X, artículo 1", "url": f"{HTML}#a1"},
        "filters": {
            "rank_code": "1300",
            "scope_code": "1",
            "subject_codes": ["85"],
            "semantic_role": "precept",
            "without_content": False,
            "annex": False,
            "table": False,
            "image": False,
        },
    }


def _chunks() -> dict:
    return {
        "schema_version": "boe_legal_chunks_v2",
        "document_id": DOC_ID,
        "source_refs": {
            "document": f"data/processed/documents/{DOC_ID}.json",
            "parents": f"data/processed/parents/{DOC_ID}.json",
        },
        "chunking_strategy": {
            "name": "legal_parent_child_paragraphs",
            "max_chars": 1800,
            "overlap_paragraphs": 1,
            "split_unit": "paragraphs",
            "parent_unit": "boe_block",
        },
        "chunks": [_chunk(A1_TEXT)],
        "generation_meta": GEN,
    }


def errors(findings: list[dict]) -> list[str]:
    return [f["check"] for f in findings if f["severity"] == "ERROR"]


# --- documento/joins limpios -------------------------------------------------


def test_clean_document_and_chunks_no_errors() -> None:
    document, history, parents, chunks = _doc(), _history(), _parents(), _chunks()
    joined = join_norm(document, history, parents)
    findings = (
        check_document(document, history, parents, processing_date="2026-05-31")
        + check_history(document, history)
        + check_parents(document, parents)
        + check_chunks(chunks, joined)
        + check_relational(document, history, parents, chunks)
    )
    assert errors(findings) == []


def test_join_norm_reconstructs_latest_version() -> None:
    joined = join_norm(_doc(), _history(), _parents())
    a1 = next(b for b in joined["blocks"] if b["block_id"] == "a1")
    assert a1["latest_version"]["text"] == A1_TEXT
    assert a1["retrieval"]["indexable"] is True


# --- problemas inyectados ----------------------------------------------------


def test_chunk_with_parent_text_flagged() -> None:
    document, history, parents = _doc(), _history(), _parents()
    chunks = _chunks()
    chunks["chunks"][0]["parent_text"] = "no debería estar aquí"
    checks = errors(check_chunks(chunks, join_norm(document, history, parents)))
    assert "chunk.parent_text_present" in checks


def test_orphan_chunk_flagged() -> None:
    document, history, parents = _doc(), _history(), _parents()
    chunks = _chunks()
    chunks["chunks"][0]["block_id"] = "zzz"
    chunks["chunks"][0]["chunk_id"] = f"{DOC_ID}__zzz__c001"
    checks = errors(check_chunks(chunks, join_norm(document, history, parents)))
    assert "chunk.orphan" in checks


def test_missing_manifest_ref_relative_flagged() -> None:
    document = _doc()
    document["source"]["manifest_ref"] = "C:/abs/path/manifest.json"
    checks = errors(check_document(document, _history(), _parents()))
    assert "doc.manifest_ref" in checks


def test_note_leak_in_chunk_flagged() -> None:
    document, parents = _doc(), _parents()
    history = _history()
    # La nota es propiedad de history; se hidrata por join. Inyectada también en el texto del chunk.
    h_a1 = next(h for h in history["blocks"] if h["block_id"] == "a1")
    h_a1["modification_notes"] = [{"text": "Se modifica por Z.", "target_norm_id": "BOE-A-2019-1"}]
    chunks = _chunks()
    chunks["chunks"][0]["text"] = A1_TEXT + "\nSe modifica por Z."
    joined = join_norm(document, history, parents)
    checks = errors(check_chunks(chunks, joined))
    assert "chunk.note_leak" in checks


# --- overlap y oversized -----------------------------------------------------


def test_overlap_reconstructs_parent_in_order() -> None:
    document, history, parents = _doc(), _history(), _parents()
    paras = [p["text"] for p in A1_PARAS]
    chunks = _chunks()
    chunks["chunks"] = [
        _chunk("\n".join(paras[:2]), 1, 2),
        _chunk("\n".join(paras[1:]), 2, 2),
    ]
    result = analyze_overlap(chunks, join_norm(document, history, parents))
    assert result["parents_split"] == 1
    assert result["overlap_boundary_ok"] == 1
    assert result["order_preserved"] == 1


def test_oversized_rows_detects_long_chunk() -> None:
    document, history, parents = _doc(), _history(), _parents()
    chunks = _chunks()
    chunks["chunks"][0]["text"] = "X" * 50
    rows = oversized_rows(chunks, join_norm(document, history, parents), max_chars=30)
    assert len(rows) == 1
    assert rows[0]["max_chars_excess"] == 20
    assert copy.deepcopy(rows[0])["block_type"] == "precepto"


def test_efficiency_no_parent_text_redundancy() -> None:
    chunks = _chunks()
    eff = efficiency_metrics(chunks, {"duplicated_chars": 0}, _parents())
    assert eff["parent_text_in_chunks_chars"] == 0
    assert eff["subjects_repeated_chars"] == 0
    assert eff["parents_store_unique_text_chars"] > 0


# --- integridad temporal (sobre joined) -------------------------------------


def test_temporal_integrity_clean() -> None:
    joined = {DOC_ID: join_norm(_doc(), _history(), _parents())}
    ti = temporal_integrity(joined, {DOC_ID: _chunks()}, processing_date="2026-05-31")
    assert ti["ready"] is True
    assert ti["mismatches"] == []


# --- readiness ---------------------------------------------------------------


def test_readiness_ready_when_clean() -> None:
    r = compute_readiness([], {}, [], temporal={"ready": True}, raw={"ready": True})
    assert r["ready"] is True
    assert r["deferred_findings"] == ["H3_oversized_token_measurement"]


def test_readiness_blocks_on_temporal_or_raw() -> None:
    assert (
        compute_readiness(
            [], {}, [], temporal={"ready": False, "mismatches": ["x"]}, raw={"ready": True}
        )["ready"]
        is False
    )
    assert (
        compute_readiness([], {}, [], temporal={"ready": True}, raw={"ready": False})["ready"]
        is False
    )


# --- raw integrity -----------------------------------------------------------


def _write_manifest(tmp_path: Path, content: bytes) -> tuple[str, Path]:
    norm_id = DOC_ID
    raw_dir = tmp_path / "raw" / norm_id
    raw_dir.mkdir(parents=True)
    f = raw_dir / "texto.xml"
    f.write_bytes(content)
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    manifest = {
        "norm_id": norm_id,
        "files": [
            {
                "endpoint_name": "texto",
                "path": f.as_posix(),
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            }
        ],
    }
    (manifest_dir / f"{norm_id}.json").write_text(json.dumps(manifest), encoding="utf-8")
    return norm_id, manifest_dir


def test_raw_integrity_ok_and_tamper(tmp_path: Path) -> None:
    norm_id, manifest_dir = _write_manifest(tmp_path, b"<xml>c</xml>")
    assert raw_integrity([norm_id], manifest_dir)["ready"] is True
    f = next((tmp_path / "raw" / norm_id).glob("texto.xml"))
    f.write_bytes(b"<xml>MOD</xml>")
    r = verify_manifest(norm_id, manifest_dir)
    assert r["sha256_mismatches"] or r["size_mismatches"]
    assert raw_integrity([norm_id], manifest_dir)["ready"] is False
