"""Tests del chunking jurídico v2 (sin red; document descriptor + parents en memoria).

El chunker v2 consume el descriptor (`boe_legal_document_v2`) y el parent store
(`boe_legal_parents_v2`) y produce child chunks vector-ready (`boe_legal_chunks_v2`): sin
`parent_text`, con `filters` y `citation`, y `retrieval_text` con el mismo contexto que la v1.
"""

import json
from pathlib import Path

from src.contracts.models import ChunksV2
from src.preprocessing.chunker import (
    CHUNKS_SCHEMA_VERSION,
    build_chunk_filters,
    build_chunk_retrieval_text,
    chunk_block,
    chunking_diagnostics,
    create_chunks,
    save_chunks,
)

DOC_ID = "BOE-A-2015-10565"
URL = f"https://www.boe.es/buscar/act.php?id={DOC_ID}"


def _descriptor(
    block_id: str,
    *,
    indexable: bool,
    block_title: str | None = None,
    full_title: str | None = None,
    semantic_role: str = "precept",
    is_annex: bool = False,
    is_without_content: bool = False,
) -> dict:
    return {
        "block_id": block_id,
        "parent_id": f"{DOC_ID}__{block_id}",
        "order": 0,
        "block_type": "precepto",
        "block_title": block_title,
        "full_title": full_title or (f"{block_title}. Rúbrica." if block_title else None),
        "semantic_role": semantic_role,
        "has_retrievable_body": True,
        "is_annex": is_annex,
        "contains_table": False,
        "table_text_available": False,
        "contains_image": False,
        "content_status": "without_content" if is_without_content else "present",
        "is_without_content": is_without_content,
        "temporal_status": "resolved",
        "hierarchy": {
            "book": None,
            "title": "TÍTULO I",
            "chapter": "CAPÍTULO I",
            "section": None,
            "subsection": None,
            "annex": None,
        },
        "indexable": indexable,
        "excluded_reason": None,
        "citation": {
            "label": f"Ley 39/2015, {block_title.lower()}" if block_title else "Ley 39/2015",
            "url": f"{URL}#{block_id}",
        },
    }


def _parent(block_id: str, paragraphs: list[str], block_title: str | None = None) -> dict:
    return {
        "parent_id": f"{DOC_ID}__{block_id}",
        "document_id": DOC_ID,
        "block_id": block_id,
        "order": 0,
        "block_type": "precepto",
        "title": block_title,
        "full_title": f"{block_title}. Rúbrica." if block_title else None,
        "semantic_role": "precept",
        "text": "\n".join(paragraphs),
        "paragraphs": [
            {"order": i + 1, "class": "parrafo", "text": t} for i, t in enumerate(paragraphs)
        ],
        "hierarchy": {
            "book": None,
            "title": "TÍTULO I",
            "chapter": "CAPÍTULO I",
            "section": None,
            "subsection": None,
            "annex": None,
        },
        "citation": {"label": "Ley 39/2015", "url": f"{URL}#{block_id}"},
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
    }


def _document(descriptors: list[dict]) -> dict:
    return {
        "schema_version": "boe_legal_document_v2",
        "document_id": DOC_ID,
        "source": {
            "name": "BOE legislación consolidada",
            "manifest_ref": f"data/manifests/{DOC_ID}.json",
        },
        "metadata": {
            "title": "Ley 39/2015...",
            "short_title": "Ley 39/2015",
            "rank": {"code": "1300", "label": "Ley"},
            "scope": {"code": "1", "label": "Estatal"},
        },
        "analysis": {
            "subjects": [{"code": "5703", "label": "Procedimiento administrativo"}],
            "notes": [],
            "references": {"previous": [], "next": []},
        },
        "blocks": descriptors,
        "generation_meta": {"generated_at": "2026-06-02T00:00:00Z", "generator": "test"},
    }


def _parents_doc(records: list[dict]) -> dict:
    return {
        "schema_version": "boe_legal_parents_v2",
        "document_id": DOC_ID,
        "parents": records,
        "generation_meta": {"generated_at": "2026-06-02T00:00:00Z", "generator": "test"},
    }


# --- contrato y básicos ------------------------------------------------------


def test_short_block_single_chunk() -> None:
    d = _descriptor("a1", indexable=True, block_title="Artículo 1")
    p = _parent("a1", ["Artículo 1.", "Texto breve."], "Artículo 1")
    chunks = chunk_block(_document([d]), d, p)
    assert len(chunks) == 1
    assert chunks[0]["position"] == {"index": 1, "count_for_parent": 1}
    assert "parent_text" not in chunks[0]
    assert chunks[0]["citation"]["url"].endswith("#a1")


def test_non_indexable_descriptor_yields_no_chunks() -> None:
    d = _descriptor(
        "ti", indexable=False, block_title="TÍTULO I", semantic_role="structural_heading"
    )
    p = _parent("ti", ["TÍTULO I"], "TÍTULO I")
    assert chunk_block(_document([d]), d, p) == []


def test_long_block_splits_with_paragraph_overlap() -> None:
    paras = ["P-uno " * 5, "P-dos " * 5, "P-tres " * 5, "P-cuatro " * 5]
    d = _descriptor("a14", indexable=True, block_title="Artículo 14")
    p = _parent("a14", paras, "Artículo 14")
    chunks = chunk_block(_document([d]), d, p, max_chars=40, overlap_paragraphs=1)
    assert len(chunks) > 1
    assert all(c["position"]["count_for_parent"] == len(chunks) for c in chunks)
    assert chunks[0]["text"].split("\n")[-1] == chunks[1]["text"].split("\n")[0]


def test_chunk_id_and_citation_preserved() -> None:
    d = _descriptor("a14", indexable=True, block_title="Artículo 14")
    p = _parent("a14", ["Artículo 14.", "Cuerpo."], "Artículo 14")
    chunk = chunk_block(_document([d]), d, p)[0]
    assert chunk["chunk_id"] == f"{DOC_ID}__a14__c001"
    assert chunk["parent_id"] == f"{DOC_ID}__a14"
    assert chunk["citation"]["label"] == "Ley 39/2015, artículo 14"
    assert chunk["citation"]["url"].endswith("#a14")


def test_modification_notes_excluded_from_chunk() -> None:
    d = _descriptor("a9", indexable=True, block_title="Artículo 9")
    p = _parent("a9", ["Artículo 9.", "Cuerpo vigente."], "Artículo 9")
    chunk = chunk_block(_document([d]), d, p)[0]
    assert "Se modifica" not in chunk["text"]
    assert "Se modifica" not in chunk["retrieval_text"]
    assert "Ley 39/2015" in chunk["retrieval_text"]
    assert "TÍTULO I" in chunk["retrieval_text"]


# --- filtros y subjects ------------------------------------------------------


def test_filters_carry_codes_not_full_subjects() -> None:
    d = _descriptor("a1", indexable=True, block_title="Artículo 1")
    filters = build_chunk_filters(_document([d]), d)
    assert filters["subject_codes"] == ["5703"]
    assert filters["rank_code"] == "1300"
    assert filters["scope_code"] == "1"
    assert filters["semantic_role"] == "precept"


# --- retrieval_text (mismo comportamiento que v1) ----------------------------


def test_retrieval_text_has_no_double_period_and_dedups_heading() -> None:
    text = "Artículo 14. Derecho y obligación.\n1. Las Administraciones."
    rt = build_chunk_retrieval_text(
        text,
        "Ley 39/2015",
        {"title": "TÍTULO II", "chapter": "CAPÍTULO I"},
        "Artículo 14. Derecho y obligación.",
    )
    assert ".." not in rt
    assert rt.startswith("Ley 39/2015")
    assert rt.count("Artículo 14. Derecho y obligación.") == 1
    assert "TÍTULO II" in rt


def test_retrieval_text_keeps_heading_when_not_at_start() -> None:
    rt = build_chunk_retrieval_text(
        "2. Apartado intermedio.", "Ley 39/2015", {"title": "TÍTULO II"}, "Artículo 14. Derecho."
    )
    assert "Artículo 14. Derecho." in rt
    assert "2. Apartado intermedio." in rt


def test_legal_text_unchanged_in_chunk() -> None:
    legal = "1. La «Administración» actúa con sometimiento pleno a la Ley."
    rt = build_chunk_retrieval_text(legal, "Ley 39/2015", {}, "Artículo 1. Objeto.")
    assert legal in rt


# --- create_chunks + diagnostics + persistencia ------------------------------


def test_create_chunks_v2_schema_and_no_diagnostics_in_payload() -> None:
    d_idx = _descriptor("a1", indexable=True, block_title="Artículo 1")
    d_hdr = _descriptor(
        "ti", indexable=False, block_title="TÍTULO I", semantic_role="structural_heading"
    )
    document = _document([d_idx, d_hdr])
    parents = _parents_doc(
        [_parent("a1", ["Corto."], "Artículo 1"), _parent("ti", ["TÍTULO I"], "TÍTULO I")]
    )
    result = create_chunks(document, parents)
    ChunksV2.model_validate(result)  # valida contra el contrato (extra=forbid)
    assert result["schema_version"] == CHUNKS_SCHEMA_VERSION
    assert "quality_checks" not in result  # diagnóstico va a reports/, no al payload
    assert result["source_refs"] == {
        "document": f"data/processed/documents/{DOC_ID}.json",
        "parents": f"data/processed/parents/{DOC_ID}.json",
    }
    diag = chunking_diagnostics(document, parents, result)
    assert diag["chunks_count"] == 1
    assert diag["indexable_blocks_count"] == 1
    assert diag["missing_parents"] == []


def test_without_content_block_is_chunked_with_neutral_filter() -> None:
    d = _descriptor("a45", indexable=True, block_title="Artículo 45", is_without_content=True)
    p = _parent("a45", ["Artículo 45.", "(Sin contenido)"], "Artículo 45")
    chunk = chunk_block(_document([d]), d, p)[0]
    assert chunk["filters"]["without_content"] is True
    assert chunk["citation"]["url"].endswith("#a45")


def test_save_chunks_writes_valid_json(tmp_path: Path) -> None:
    d = _descriptor("a1", indexable=True, block_title="Artículo 1")
    parents = _parents_doc([_parent("a1", ["Artículo 1.", "Cuerpo."], "Artículo 1")])
    result = create_chunks(_document([d]), parents)
    out_path = save_chunks(result, tmp_path / "chunks")
    assert out_path.name == f"{DOC_ID}.json"
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == CHUNKS_SCHEMA_VERSION
    assert loaded["chunks"][0]["chunk_id"] == f"{DOC_ID}__a1__c001"
