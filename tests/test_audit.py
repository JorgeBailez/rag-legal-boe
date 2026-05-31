"""Tests del módulo de auditoría (sin red, dicts mínimos en memoria)."""

import copy
import hashlib
import json
from pathlib import Path

from src.quality.corpus_audit import (
    analyze_overlap,
    check_chunks,
    check_document,
    check_retrieval_text,
    compute_readiness,
    oversized_rows,
    raw_integrity,
    temporal_integrity,
    verify_manifest,
)

DOC_ID = "BOE-A-2015-10565"
HTML = "https://www.boe.es/buscar/act.php?id=BOE-A-2015-10565"


def make_doc() -> dict:
    a1_text = "Artículo 1. Objeto.\n1. Primer apartado.\n2. Segundo apartado."
    return {
        "schema_version": "boe_legal_document_v1",
        "document_id": DOC_ID,
        "source": {"name": "BOE legislación consolidada"},
        "metadata": {
            "title": "Ley X",
            "short_title": "Ley X",
            "identifier": DOC_ID,
            "html_url": HTML,
            "rank": {"code": "1300", "label": "Ley"},
            "scope": {"code": "1", "label": "Estatal"},
            "publication_date": "2015-10-02",
            "last_update_datetime": "2026-05-20T07:06:02Z",
        },
        "analysis": {"subjects": [], "notes": [], "references": {"previous": [], "next": []}},
        "blocks": [
            {
                "block_id": "ti",
                "parent_id": f"{DOC_ID}__ti",
                "block_type": "encabezado",
                "block_title": "TÍTULO I",
                "full_title": "TÍTULO I",
                "hierarchy": {"title": "TÍTULO I", "chapter": None, "section": None},
                "index_last_update_date": "2015-10-02",
                "versions": [
                    {
                        "source_norm_id": DOC_ID,
                        "publication_date": "2015-10-02",
                        "validity_date": "2016-10-02",
                        "is_latest": True,
                    }
                ],
                "latest_version": {
                    "source_norm_id": DOC_ID,
                    "publication_date": "2015-10-02",
                    "validity_date": "2016-10-02",
                    "text": "TÍTULO I",
                    "paragraphs": [],
                    "modification_notes": [],
                },
                "retrieval": {
                    "indexable": False,
                    "retrieval_text": "Ley X. TÍTULO I",
                    "citation_label": "Ley X, tÍTULO I",
                    "source_url": f"{HTML}#ti",
                },
            },
            {
                "block_id": "a1",
                "parent_id": f"{DOC_ID}__a1",
                "block_type": "precepto",
                "block_title": "Artículo 1",
                "full_title": "Artículo 1. Objeto.",
                "hierarchy": {"title": "TÍTULO I", "chapter": None, "section": None},
                "index_last_update_date": "2015-10-02",
                "versions": [
                    {
                        "source_norm_id": DOC_ID,
                        "publication_date": "2015-10-02",
                        "validity_date": "2016-10-02",
                        "is_latest": True,
                    }
                ],
                "latest_version": {
                    "source_norm_id": DOC_ID,
                    "publication_date": "2015-10-02",
                    "validity_date": "2016-10-02",
                    "text": a1_text,
                    "paragraphs": [
                        {"order": 1, "class": "articulo", "text": "Artículo 1. Objeto."},
                        {"order": 2, "class": "parrafo", "text": "1. Primer apartado."},
                        {"order": 3, "class": "parrafo", "text": "2. Segundo apartado."},
                    ],
                    "modification_notes": [
                        {"text": "Se modifica por Z.", "target_norm_id": "BOE-A-2019-1"}
                    ],
                },
                "retrieval": {
                    "indexable": True,
                    "retrieval_text": f"Ley X. Artículo 1. Objeto. {a1_text}",
                    "citation_label": "Ley X, artículo 1",
                    "source_url": f"{HTML}#a1",
                },
            },
        ],
        "quality_checks": {
            "index_blocks_count": 2,
            "text_blocks_count": 2,
            "unmatched_index_blocks": [],
            "unmatched_text_blocks": [],
            "warnings": [],
        },
    }


def make_chunk_meta() -> dict:
    return {
        "schema_version": "boe_legal_document_v1",
        "source": "BOE legislación consolidada",
        "legal_status_notice": "...",
        "norm_title": "Ley X",
        "short_title": "Ley X",
        "document_id": DOC_ID,
        "block_id": "a1",
        "block_type": "precepto",
        "block_title": "Artículo 1",
        "full_title": "Artículo 1. Objeto.",
        "citation_label": "Ley X, artículo 1",
        "source_url": f"{HTML}#a1",
        "hierarchy": {"title": "TÍTULO I", "chapter": None, "section": None},
        "rank": {"code": "1300", "label": "Ley"},
        "scope": {"code": "1", "label": "Estatal"},
        "subjects": [],
        "is_preamble": False,
    }


def make_chunks() -> dict:
    a1_text = "Artículo 1. Objeto.\n1. Primer apartado.\n2. Segundo apartado."
    return {
        "schema_version": "boe_legal_chunks_v1",
        "document_id": DOC_ID,
        "chunking_strategy": {
            "name": "legal_parent_child_v1",
            "max_chars": 1800,
            "overlap_paragraphs": 1,
            "split_unit": "paragraphs",
            "parent_unit": "boe_block",
        },
        "chunks": [
            {
                "chunk_id": f"{DOC_ID}__a1__c001",
                "parent_id": f"{DOC_ID}__a1",
                "document_id": DOC_ID,
                "block_id": "a1",
                "chunk_index": 1,
                "chunk_count_for_parent": 1,
                "chunking_strategy": "legal_parent_child_v1",
                "text": a1_text,
                "retrieval_text": f"Ley X. Artículo 1. Objeto. {a1_text}",
                "parent_text": a1_text,
                "metadata": make_chunk_meta(),
            }
        ],
        "quality_checks": {},
    }


def errors(findings: list[dict]) -> list[dict]:
    return [f for f in findings if f["severity"] == "ERROR"]


# --- documento limpio --------------------------------------------------------


def test_clean_document_has_no_errors() -> None:
    assert errors(check_document(make_doc())) == []


def test_clean_chunks_have_no_errors() -> None:
    doc = make_doc()
    assert errors(check_chunks(make_chunks(), doc)) == []


# --- problemas inyectados ----------------------------------------------------


def test_duplicate_is_latest_flagged() -> None:
    doc = make_doc()
    doc["blocks"][1]["versions"].append({"source_norm_id": DOC_ID, "is_latest": True})
    checks = [f["check"] for f in check_document(doc)]
    assert "block.is_latest" in checks


def test_indexable_encabezado_flagged() -> None:
    doc = make_doc()
    doc["blocks"][0]["retrieval"]["indexable"] = True
    checks = [f["check"] for f in check_document(doc)]
    assert "block.indexable" in checks


def test_note_leak_in_text_flagged() -> None:
    doc = make_doc()
    lv = doc["blocks"][1]["latest_version"]
    lv["text"] = lv["text"] + "\nSe modifica por Z."  # nota dentro del texto normativo
    checks = [f["check"] for f in check_document(doc)]
    assert "block.note_leak" in checks


def test_chunk_sequence_broken_flagged() -> None:
    doc = make_doc()
    cd = make_chunks()
    cd["chunks"][0]["chunk_index"] = 2  # rompe la secuencia c001..cN
    checks = [f["check"] for f in check_chunks(cd, doc)]
    assert "chunk.sequence" in checks


def test_parent_text_mismatch_flagged() -> None:
    doc = make_doc()
    cd = make_chunks()
    cd["chunks"][0]["parent_text"] = "otro texto"
    checks = [f["check"] for f in check_chunks(cd, doc)]
    assert "chunk.parent_text" in checks


def test_retrieval_text_double_period_flagged() -> None:
    chunk = make_chunks()["chunks"][0]
    chunk["retrieval_text"] = "Ley X.. Artículo 1."
    checks = [f["check"] for f in check_retrieval_text(chunk)]
    assert "rt.double_period" in checks


# --- overlap y oversized -----------------------------------------------------


def test_overlap_reconstructs_parent_in_order() -> None:
    doc = make_doc()
    paras = [p["text"] for p in doc["blocks"][1]["latest_version"]["paragraphs"]]
    cd = make_chunks()
    # Dos chunks con overlap de 1 párrafo (el 2.º se repite).
    cd["chunks"] = [
        {
            "chunk_id": f"{DOC_ID}__a1__c001",
            "parent_id": f"{DOC_ID}__a1",
            "document_id": DOC_ID,
            "block_id": "a1",
            "chunk_index": 1,
            "chunk_count_for_parent": 2,
            "chunking_strategy": "legal_parent_child_v1",
            "text": "\n".join(paras[:2]),
            "retrieval_text": "Ley X. " + " ".join(paras[:2]),
            "parent_text": "\n".join(paras),
            "metadata": make_chunk_meta(),
        },
        {
            "chunk_id": f"{DOC_ID}__a1__c002",
            "parent_id": f"{DOC_ID}__a1",
            "document_id": DOC_ID,
            "block_id": "a1",
            "chunk_index": 2,
            "chunk_count_for_parent": 2,
            "chunking_strategy": "legal_parent_child_v1",
            "text": "\n".join(paras[1:]),
            "retrieval_text": "Ley X. " + " ".join(paras[1:]),
            "parent_text": "\n".join(paras),
            "metadata": make_chunk_meta(),
        },
    ]
    result = analyze_overlap(cd, doc)
    assert result["parents_split"] == 1
    assert result["overlap_boundary_ok"] == 1
    assert result["order_preserved"] == 1
    assert result["duplicated_paragraphs"] == 1


def test_oversized_rows_detects_long_chunk() -> None:
    doc = make_doc()
    cd = make_chunks()
    cd["chunks"][0]["text"] = "X" * 50
    rows = oversized_rows(cd, doc, max_chars=30)
    assert len(rows) == 1
    assert rows[0]["single_paragraph_oversized"] is True
    assert rows[0]["max_chars_excess"] == 20
    assert copy.deepcopy(rows[0])["block_type"] == "precepto"


# --- pre_embedding_readiness + encabezado sustantivo no indexado -------------


def test_readiness_ready_when_clean() -> None:
    r = compute_readiness(findings=[], unhandled_hierarchy={}, editorial_indexable=[])
    assert r["ready"] is True
    assert r["blocking_findings"] == []
    assert r["deferred_findings"] == ["H3_oversized_token_measurement"]


def test_readiness_blocks_on_editorial_indexable() -> None:
    r = compute_readiness(findings=[], unhandled_hierarchy={}, editorial_indexable=[("n", "ni")])
    assert r["ready"] is False
    assert "H1_nota_inicial_indexable" in r["blocking_findings"]


def test_readiness_h3_stays_deferred_not_blocking() -> None:
    r = compute_readiness(findings=[], unhandled_hierarchy={}, editorial_indexable=[])
    assert "H3_oversized_token_measurement" in r["deferred_findings"]
    assert r["ready"] is True  # H3 no bloquea


def test_substantive_heading_not_indexed_flagged() -> None:
    doc = make_doc()
    # Encabezado con cuerpo sustantivo (anexo) pero marcado como NO indexable.
    doc["blocks"].append(
        {
            "block_id": "ai",
            "parent_id": f"{DOC_ID}__ai",
            "block_type": "encabezado",
            "block_title": "ANEXO I",
            "full_title": "ANEXO I. Tablas",
            "hierarchy": {"annex": "ANEXO I"},
            "versions": [{"source_norm_id": DOC_ID, "is_latest": True}],
            "latest_version": {
                "source_norm_id": DOC_ID,
                "text": "ANEXO I\nContenido.",
                "paragraphs": [
                    {"order": 1, "class": "anexo_num", "text": "ANEXO I"},
                    {"order": 2, "class": "parrafo", "text": "Contenido."},
                ],
                "modification_notes": [],
            },
            "retrieval": {
                "indexable": False,
                "retrieval_text": "x",
                "citation_label": "Ley X",
                "source_url": f"{HTML}#ai",
            },
        }
    )
    checks = [f["check"] for f in check_document(doc)]
    assert "block.heading_body_not_indexed" in checks


# --- integridad temporal -----------------------------------------------------


def _multiversion_doc(index_date: str, latest_pub: str) -> dict:
    """Doc con un bloque de 2 versiones; `index_date` = vigente; `latest_pub` = lo persistido."""
    doc = make_doc()
    src = "BOE-A-2020-1" if latest_pub == "2020-01-01" else "BOE-A-2013-1"
    doc["blocks"] = [
        {
            "block_id": "a2",
            "parent_id": f"{DOC_ID}__a2",
            "block_type": "precepto",
            "block_title": "Artículo 2",
            "full_title": "Artículo 2.",
            "hierarchy": {"title": None, "chapter": None, "section": None},
            "index_last_update_date": index_date,
            "versions": [
                {"source_norm_id": "BOE-A-2013-1", "publication_date": "2013-12-30"},
                {"source_norm_id": "BOE-A-2020-1", "publication_date": "2020-01-01"},
            ],
            "latest_version": {
                "source_norm_id": src,
                "publication_date": latest_pub,
                "validity_date": "2021-01-01",
                "text": "Artículo 2. Texto.",
                "paragraphs": [{"order": 1, "class": "articulo", "text": "Artículo 2. Texto."}],
                "modification_notes": [],
            },
            "retrieval": {
                "indexable": True,
                "retrieval_text": "Ley X. Artículo 2. Texto.",
                "citation_label": "Ley X, artículo 2",
                "source_url": f"{HTML}#a2",
            },
        }
    ]
    return doc


def test_temporal_integrity_clean_corpus_is_ready() -> None:
    ti = temporal_integrity({DOC_ID: make_doc()}, processing_date="2026-05-31")
    assert ti["ready"] is True
    assert ti["mismatches"] == []
    assert ti["latest_matches_index"] == 2


def test_temporal_integrity_detects_injected_mismatch() -> None:
    # Índice dice 2020 pero latest_version persiste la versión histórica de 2013.
    doc = _multiversion_doc(index_date="2020-01-01", latest_pub="2013-12-30")
    ti = temporal_integrity({DOC_ID: doc}, processing_date="2026-05-31")
    assert ti["ready"] is False
    assert f"{DOC_ID}/a2" in ti["mismatches"]
    # Y la auditoría estructural lo marca como ERROR.
    checks = [f["check"] for f in errors(check_document(doc, processing_date="2026-05-31"))]
    assert "block.temporal_mismatch" in checks


def test_temporal_integrity_reports_future_effective() -> None:
    doc = _multiversion_doc(index_date="2020-01-01", latest_pub="2020-01-01")
    doc["blocks"][0]["latest_version"]["validity_date"] = "2099-01-01"
    ti = temporal_integrity({DOC_ID: doc}, processing_date="2026-05-31")
    # Vigencia futura se reporta pero NO bloquea (política explícita del MVP).
    assert f"{DOC_ID}/a2" in ti["future_effective_selected_versions"]
    assert ti["ready"] is True


def test_temporal_integrity_flags_non_chronological_order() -> None:
    doc = make_doc()
    # Reordena las versiones del bloque a1 para que no sean cronológicas.
    doc["blocks"][1]["versions"] = [
        {"source_norm_id": DOC_ID, "publication_date": "2020-01-01"},
        {"source_norm_id": DOC_ID, "publication_date": "2015-10-02"},
    ]
    doc["blocks"][1]["index_last_update_date"] = "2020-01-01"
    doc["blocks"][1]["latest_version"]["publication_date"] = "2020-01-01"
    ti = temporal_integrity({DOC_ID: doc}, processing_date="2026-05-31")
    assert f"{DOC_ID}/a1" in ti["non_chronological_xml_order_blocks"]


# --- gate combinado: readiness con temporal y raw ----------------------------


def test_readiness_blocks_on_temporal_not_ready() -> None:
    temporal = {
        "ready": False,
        "mismatches": ["x/a2"],
        "ambiguous_blocks": [],
        "missing_index_date": [],
        "invalid_dates": [],
        "index_not_max": [],
        "chunks_built_from_non_current_version": [],
    }
    r = compute_readiness([], {}, [], temporal=temporal, raw={"ready": True})
    assert r["ready"] is False
    assert "temporal_mismatches" in r["blocking_findings"]


def test_readiness_blocks_on_raw_not_ready() -> None:
    r = compute_readiness([], {}, [], temporal={"ready": True}, raw={"ready": False})
    assert r["ready"] is False
    assert "raw_integrity" in r["blocking_findings"]


def test_readiness_ready_when_temporal_and_raw_ok() -> None:
    r = compute_readiness([], {}, [], temporal={"ready": True}, raw={"ready": True})
    assert r["ready"] is True
    assert r["blocking_findings"] == []


# --- integridad raw (manifests) ----------------------------------------------


def _write_manifest(tmp_path: Path, content: bytes) -> tuple[str, Path]:
    norm_id = "BOE-A-2015-10565"
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


def test_raw_integrity_ok_when_hashes_match(tmp_path: Path) -> None:
    norm_id, manifest_dir = _write_manifest(tmp_path, b"<xml>contenido</xml>")
    agg = raw_integrity([norm_id], manifest_dir)
    assert agg["ready"] is True
    assert agg["files_checked"] == 1


def test_raw_integrity_detects_tampered_file(tmp_path: Path) -> None:
    norm_id, manifest_dir = _write_manifest(tmp_path, b"<xml>contenido</xml>")
    # Manipula el fichero tras escribir el manifest.
    f = next((tmp_path / "raw" / norm_id).glob("texto.xml"))
    f.write_bytes(b"<xml>contenido MODIFICADO</xml>")
    r = verify_manifest(norm_id, manifest_dir)
    assert r["sha256_mismatches"] or r["size_mismatches"]
    agg = raw_integrity([norm_id], manifest_dir)
    assert agg["ready"] is False
