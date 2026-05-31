"""Tests del chunking jurídico v0 (sin red, documentos mínimos en memoria/tmp_path).

No se usa la Ley completa como fixture: cada test construye un documento mínimo
`boe_legal_document_v1` en memoria. Se usan `max_chars` pequeños para forzar la división
sin necesidad de textos enormes.
"""

import json
from pathlib import Path

from src.preprocessing.chunker import (
    CHUNKS_SCHEMA_VERSION,
    build_chunk_retrieval_text,
    chunk_block,
    create_chunks,
    load_processed_document,
    save_chunks,
)

DOC_ID = "BOE-A-2015-10565"


def _block(
    block_id: str,
    block_type: str,
    *,
    indexable: bool,
    paragraphs: list[str],
    block_title: str | None = None,
) -> dict:
    paras = [{"order": i + 1, "class": "parrafo", "text": t} for i, t in enumerate(paragraphs)]
    text = "\n".join(paragraphs)
    return {
        "block_id": block_id,
        "parent_id": f"{DOC_ID}__{block_id}",
        "order": 0,
        "block_type": block_type,
        "block_title": block_title,
        "full_title": f"{block_title}. Rúbrica." if block_title else None,
        "hierarchy": {"title": "TÍTULO I", "chapter": "CAPÍTULO I", "section": None},
        "versions": [{"source_norm_id": DOC_ID, "is_latest": True}],
        "latest_version": {
            "source_norm_id": DOC_ID,
            "publication_date": "2015-10-02",
            "validity_date": "2016-10-02",
            "text": text,
            "paragraphs": paras,
            "modification_notes": [
                {"text": "Se modifica por X.", "target_norm_id": "BOE-A-2019-1"}
            ],
        },
        "retrieval": {
            "indexable": indexable,
            "retrieval_text": text,
            "citation_label": f"Ley 39/2015, {block_title.lower()}"
            if block_title
            else "Ley 39/2015",
            "source_url": f"https://www.boe.es/buscar/act.php?id={DOC_ID}#{block_id}",
        },
    }


def _document(blocks: list[dict]) -> dict:
    return {
        "schema_version": "boe_legal_document_v1",
        "document_id": DOC_ID,
        "source": {"name": "BOE legislación consolidada"},
        "metadata": {
            "title": "Ley 39/2015, de 1 de octubre...",
            "short_title": "Ley 39/2015",
            "legal_status_notice": "Texto consolidado de carácter informativo.",
            "rank": {"code": "1300", "label": "Ley"},
            "scope": {"code": "1", "label": "Estatal"},
        },
        "analysis": {"subjects": [{"code": "5703", "label": "Procedimiento administrativo"}]},
        "blocks": blocks,
    }


# --- 1. carga ----------------------------------------------------------------


def test_load_processed_document(tmp_path: Path) -> None:
    doc = _document([])
    path = tmp_path / "doc.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    loaded = load_processed_document(path)
    assert loaded["document_id"] == DOC_ID


# --- 2. no indexable ---------------------------------------------------------


def test_non_indexable_block_yields_no_chunks() -> None:
    block = _block(
        "ti", "encabezado", indexable=False, paragraphs=["TÍTULO I"], block_title="TÍTULO I"
    )
    assert chunk_block(block, _document([block])) == []


# --- 3. bloque corto = 1 chunk ----------------------------------------------


def test_short_block_single_chunk() -> None:
    block = _block(
        "a1",
        "precepto",
        indexable=True,
        paragraphs=["Artículo 1.", "Texto breve."],
        block_title="Artículo 1",
    )
    chunks = chunk_block(block, _document([block]))
    assert len(chunks) == 1
    assert chunks[0]["chunk_count_for_parent"] == 1
    assert chunks[0]["chunk_index"] == 1


# --- 4-5. bloque largo: varios chunks por párrafos + overlap ----------------


def test_long_block_splits_with_paragraph_overlap() -> None:
    paragraphs = ["P-uno " * 5, "P-dos " * 5, "P-tres " * 5, "P-cuatro " * 5]
    block = _block(
        "a14", "precepto", indexable=True, paragraphs=paragraphs, block_title="Artículo 14"
    )
    chunks = chunk_block(block, _document([block]), max_chars=40, overlap_paragraphs=1)

    assert len(chunks) > 1
    assert all(c["chunk_count_for_parent"] == len(chunks) for c in chunks)
    # overlap de 1 párrafo: el último párrafo de un chunk reaparece al inicio del siguiente.
    first_last_para = chunks[0]["text"].split("\n")[-1]
    second_first_para = chunks[1]["text"].split("\n")[0]
    assert first_last_para == second_first_para


# --- 6-7. formato de id y preservación de citas -----------------------------


def test_chunk_id_format_and_citation_preserved() -> None:
    block = _block(
        "a14",
        "precepto",
        indexable=True,
        paragraphs=["Artículo 14.", "Cuerpo."],
        block_title="Artículo 14",
    )
    chunk = chunk_block(block, _document([block]))[0]
    assert chunk["chunk_id"] == f"{DOC_ID}__a14__c001"
    assert chunk["parent_id"] == f"{DOC_ID}__a14"
    assert chunk["block_id"] == "a14"
    assert chunk["metadata"]["citation_label"] == "Ley 39/2015, artículo 14"
    assert chunk["metadata"]["source_url"].endswith("#a14")


# --- 8. modification_notes fuera del chunk ----------------------------------


def test_modification_notes_excluded_from_chunk() -> None:
    block = _block(
        "a9",
        "precepto",
        indexable=True,
        paragraphs=["Artículo 9.", "Cuerpo vigente."],
        block_title="Artículo 9",
    )
    chunk = chunk_block(block, _document([block]))[0]
    assert "Se modifica" not in chunk["text"]
    assert "Se modifica" not in chunk["retrieval_text"]
    # El contexto jurídico sí está en retrieval_text.
    assert "Ley 39/2015" in chunk["retrieval_text"]
    assert "TÍTULO I" in chunk["retrieval_text"]


# --- 9. parent_text completo -------------------------------------------------


def test_parent_text_is_full_block_text() -> None:
    paragraphs = ["Artículo 5.", "Apartado 1.", "Apartado 2."]
    block = _block(
        "a5", "precepto", indexable=True, paragraphs=paragraphs, block_title="Artículo 5"
    )
    for chunk in chunk_block(block, _document([block]), max_chars=20, overlap_paragraphs=1):
        assert chunk["parent_text"] == "\n".join(paragraphs)


# --- 10. quality_checks ------------------------------------------------------


def test_quality_checks_counts_and_oversized() -> None:
    short = _block(
        "a1", "precepto", indexable=True, paragraphs=["Corto."], block_title="Artículo 1"
    )
    header = _block(
        "ti", "encabezado", indexable=False, paragraphs=["TÍTULO I"], block_title="TÍTULO I"
    )
    oversized = _block(
        "a2", "precepto", indexable=True, paragraphs=["X" * 50], block_title="Artículo 2"
    )
    doc = _document([short, header, oversized])

    result = create_chunks(doc, max_chars=30, overlap_paragraphs=1)
    checks = result["quality_checks"]
    assert checks["source_blocks_count"] == 3
    assert checks["indexable_blocks_count"] == 2  # encabezado excluido
    assert checks["chunks_count"] == 2
    assert checks["oversized_chunks"] == [f"{DOC_ID}__a2__c001"]
    assert result["schema_version"] == CHUNKS_SCHEMA_VERSION


def test_preamble_is_flagged() -> None:
    pre = _block("preambulo", "preambulo", indexable=True, paragraphs=["Exposición de motivos."])
    chunk = chunk_block(pre, _document([pre]))[0]
    assert chunk["metadata"]["is_preamble"] is True


# --- 11. persistencia / integración -----------------------------------------


def test_save_chunks_writes_valid_json(tmp_path: Path) -> None:
    block = _block(
        "a1",
        "precepto",
        indexable=True,
        paragraphs=["Artículo 1.", "Cuerpo."],
        block_title="Artículo 1",
    )
    result = create_chunks(_document([block]))
    out_path = save_chunks(result, tmp_path / "chunks")

    assert out_path.name == f"{DOC_ID}.json"
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == CHUNKS_SCHEMA_VERSION
    assert loaded["chunks"][0]["chunk_id"] == f"{DOC_ID}__a1__c001"


# --- correcciones pre-embeddings: retrieval_text sin '..' y dedup -------------


def _meta(full_title="Artículo 14. Derecho y obligación.", hierarchy=None):
    return {
        "short_title": "Ley 39/2015",
        "full_title": full_title,
        "block_title": "Artículo 14",
        "hierarchy": hierarchy or {"title": "TÍTULO II", "chapter": "CAPÍTULO I", "section": None},
    }


def test_retrieval_text_has_no_double_period() -> None:
    # full_title acaba en '.', el prefijo NO debe generar '..'.
    text = "Artículo 14. Derecho y obligación.\n1. Las Administraciones."
    rt = build_chunk_retrieval_text(text, _meta())
    assert ".." not in rt
    assert rt.startswith("Ley 39/2015")


def test_retrieval_text_dedups_heading_in_first_chunk() -> None:
    # El texto empieza por full_title -> no se repite la cabecera en el prefijo.
    text = "Artículo 14. Derecho y obligación.\n1. Cuerpo."
    rt = build_chunk_retrieval_text(text, _meta())
    assert rt.count("Artículo 14. Derecho y obligación.") == 1
    assert "TÍTULO II" in rt  # mantiene contexto jerárquico


def test_retrieval_text_keeps_heading_when_not_at_start() -> None:
    # Chunk posterior (c002) NO empieza por la cabecera -> se mantiene como contexto.
    text = "2. Apartado intermedio del artículo."
    rt = build_chunk_retrieval_text(text, _meta())
    assert "Artículo 14. Derecho y obligación." in rt
    assert text in rt


def test_retrieval_text_includes_book_and_annex_levels() -> None:
    meta = _meta(hierarchy={"book": "LIBRO I", "annex": None, "subsection": "Subsección 1"})
    rt = build_chunk_retrieval_text("Cuerpo del artículo.", meta)
    assert "LIBRO I" in rt and "Subsección 1" in rt


def test_legal_text_unchanged_in_chunk() -> None:
    # build_chunk_retrieval_text no altera el texto del chunk (solo antepone contexto).
    legal = "1. La «Administración» actúa con sometimiento pleno a la Ley."
    rt = build_chunk_retrieval_text(legal, _meta(full_title="Artículo 1. Objeto."))
    assert legal in rt


def test_quarantined_block_yields_no_chunks() -> None:
    # Bloque en cuarentena temporal: latest_version=null, indexable=false -> sin chunks.
    block = {
        "block_id": "a45",
        "parent_id": f"{DOC_ID}__a45",
        "block_type": "precepto",
        "block_title": "Artículo 45",
        "temporal_quarantined": True,
        "versions": [
            {"source_norm_id": "N1", "publication_date": "2015-01-01", "is_latest": False},
            {"source_norm_id": "N2", "publication_date": "2020-01-01", "is_latest": False},
        ],
        "latest_version": None,
        "retrieval": {
            "indexable": False,
            "retrieval_text": "Ley 39/2015",
            "citation_label": "Ley 39/2015, artículo 45",
            "source_url": f"https://www.boe.es/buscar/act.php?id={DOC_ID}#a45",
            "excluded_reason": "temporal_quarantine:unresolved",
        },
    }
    assert chunk_block(block, _document([block])) == []
    result = create_chunks(_document([block]))
    assert result["quality_checks"]["chunks_count"] == 0


def test_without_content_chunk_has_neutral_metadata_and_citation() -> None:
    block = _block(
        "a45",
        "precepto",
        indexable=True,
        paragraphs=["Artículo 45.", "(Sin contenido)"],
        block_title="Artículo 45",
    )
    block["content_status"] = "without_content"
    block["is_without_content"] = True
    chunk = chunk_block(block, _document([block]))[0]
    # Sigue siendo recuperable: conserva cita y URL oficial.
    assert chunk["metadata"]["citation_label"] == "Ley 39/2015, artículo 45"
    assert chunk["metadata"]["source_url"].endswith("#a45")
    # Metadata neutral, sin inferir causa/derogación.
    assert chunk["metadata"]["content_status"] == "without_content"
    assert chunk["metadata"]["is_without_content"] is True


def test_annex_block_with_body_is_chunked() -> None:
    # Un encabezado de anexo con cuerpo (indexable) debe producir chunks.
    block = {
        "block_id": "ai",
        "parent_id": f"{DOC_ID}__ai",
        "block_type": "encabezado",
        "block_title": "ANEXO I",
        "full_title": "ANEXO I. Tablas",
        "is_annex": True,
        "semantic_role": "annex",
        "has_retrievable_body": True,
        "hierarchy": {
            "book": None,
            "title": None,
            "chapter": None,
            "section": None,
            "subsection": None,
            "annex": "ANEXO I",
        },
        "latest_version": {
            "source_norm_id": DOC_ID,
            "text": "ANEXO I\nTablas\nContenido del anexo.",
            "paragraphs": [
                {"order": 1, "class": "anexo_num", "text": "ANEXO I"},
                {"order": 2, "class": "anexo_tit", "text": "Tablas"},
                {"order": 3, "class": "parrafo", "text": "Contenido del anexo."},
            ],
            "modification_notes": [],
        },
        "retrieval": {
            "indexable": True,
            "citation_label": "Ley 39/2015, aNEXO I",
            "source_url": f"https://www.boe.es/buscar/act.php?id={DOC_ID}#ai",
        },
    }
    doc = _document([block])
    chunks = chunk_block(block, doc)
    assert len(chunks) >= 1
    assert chunks[0]["metadata"]["is_annex"] is True
    assert chunks[0]["metadata"]["source_url"].endswith("#ai")
    assert chunks[0]["parent_text"] == block["latest_version"]["text"]
