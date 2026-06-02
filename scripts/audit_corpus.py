"""Auditoría de calidad del corpus MVP (parser + chunker), de solo lectura.

Uso:
    uv run python scripts/audit_corpus.py

Lee los artefactos generados (documents/chunks) y el raw (texto.xml) — sin red, sin
modificar nada — y escribe:
    data/processed/reports/mvp_chunking_audit.json
    data/processed/reports/mvp_chunking_audit.csv   (tabla de chunks sobredimensionados)
Imprime un resumen por severidad/clasificación.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.quality import corpus_audit as ca  # noqa: E402

DOCS_DIR = Path("data/processed/documents")
HISTORIES_DIR = Path("data/processed/histories")
PARENTS_DIR = Path("data/processed/parents")
CHUNKS_DIR = Path("data/processed/chunks")
RAW_DIR = Path("data/raw/boe")
MANIFEST_DIR = Path("data/manifests")
REPORTS_DIR = Path("data/processed/reports")
TABLE_CLASSES = ("cuerpo_tabla", "cabeza_tabla")


def _load_all() -> tuple[dict, dict, dict, dict, dict]:
    """Carga los 4 artefactos v2 + la vista compuesta (joined) por norma.

    Devuelve (joined_docs, raw_documents, histories, parents, chunks). `joined_docs` es la vista
    autoritativa (document+history+parents) con `latest_version`/`versions`/`retrieval` por bloque.
    """
    documents, histories, parents, chunks = {}, {}, {}, {}
    for f in sorted(glob.glob(str(DOCS_DIR / "*.json"))):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        documents[d["document_id"]] = d
    for f in sorted(glob.glob(str(HISTORIES_DIR / "*.json"))):
        h = json.loads(Path(f).read_text(encoding="utf-8"))
        histories[h["document_id"]] = h
    for f in sorted(glob.glob(str(PARENTS_DIR / "*.json"))):
        p = json.loads(Path(f).read_text(encoding="utf-8"))
        parents[p["document_id"]] = p
    for f in sorted(glob.glob(str(CHUNKS_DIR / "*.json"))):
        c = json.loads(Path(f).read_text(encoding="utf-8"))
        chunks[c["document_id"]] = c
    joined = {
        did: ca.join_norm(
            documents[did], histories.get(did, {"blocks": []}), parents.get(did, {"parents": []})
        )
        for did in documents
    }
    return joined, documents, histories, parents, chunks


def _has_table(block: dict) -> bool:
    classes = {p["class"] for p in (block.get("latest_version") or {}).get("paragraphs", [])}
    return any(c.startswith(TABLE_CLASSES) for c in classes)


def _select_examples(docs: dict, chunks: dict) -> list[tuple[str, str]]:
    """Selecciona (norma, block_id) representativos para trazar (uno por categoría)."""
    picked: dict[str, tuple[str, str]] = {}
    oversized_blocks = set()
    for nid, cd in chunks.items():
        mc = cd["chunking_strategy"]["max_chars"]
        for ch in cd["chunks"]:
            if len(ch["text"]) > mc:
                oversized_blocks.add((nid, ch["block_id"]))

    for nid, doc in docs.items():
        cd = chunks.get(nid, {})
        counts = {}
        for ch in cd.get("chunks", []):
            counts[ch["block_id"]] = (ch.get("position") or {}).get("count_for_parent")
        for b in doc["blocks"]:
            bid = b["block_id"]
            bt = b["block_type"]
            lv = b.get("latest_version") or {}
            if "preambulo" not in picked and bt == "preambulo":
                picked["preambulo"] = (nid, bid)
            if "nota_inicial" not in picked and bt == "nota_inicial":
                picked["nota_inicial"] = (nid, bid)
            if "multiversion" not in picked and len(b.get("versions") or []) > 1:
                picked["multiversion"] = (nid, bid)
            if "nota_editorial" not in picked and lv.get("modification_notes"):
                picked["nota_editorial"] = (nid, bid)
            if "tabla" not in picked and _has_table(b):
                picked["tabla"] = (nid, bid)
            if "anexo" not in picked and b.get("is_annex") and b.get("has_retrievable_body"):
                picked["anexo"] = (nid, bid)
            if (
                "encabezado_contenido" not in picked
                and bt == "encabezado"
                and b.get("has_retrievable_body")
            ):
                picked["encabezado_contenido"] = (nid, bid)
            if (
                "corto" not in picked
                and bt == "precepto"
                and counts.get(bid) == 1
                and len(lv.get("text", "")) < 400
            ):
                picked["corto"] = (nid, bid)
            if "dividido" not in picked and counts.get(bid, 0) > 1:
                picked["dividido"] = (nid, bid)
            if (nid, bid) in oversized_blocks and "oversized" not in picked:
                picked["oversized"] = (nid, bid)
    return list(picked.items())


def main(strict: bool = False) -> int:
    docs, documents, histories, parents, chunks = _load_all()
    if not docs:
        print("No hay documentos procesados en data/processed/documents/.", file=sys.stderr)
        return 1

    all_findings: list[dict] = []
    per_norm: dict[str, dict] = {}
    oversized_all: list[dict] = []

    for nid, doc in docs.items():
        cd = chunks.get(nid, {})
        pj = parents.get(nid, {"parents": []})
        max_chars = cd.get("chunking_strategy", {}).get("max_chars", 1800)
        findings = (
            ca.check_document(documents[nid], histories.get(nid), parents.get(nid))
            + ca.check_history(documents[nid], histories.get(nid, {"blocks": []}))
            + ca.check_parents(documents[nid], pj)
            + ca.check_chunks(cd, doc)
            + ca.check_relational(documents[nid], histories.get(nid, {"blocks": []}), pj, cd)
        )
        all_findings.extend(findings)
        overlap = ca.analyze_overlap(cd, doc)
        oversized = ca.oversized_rows(cd, doc, max_chars)
        oversized_all.extend(oversized)
        per_norm[nid] = {
            "blocks": len(doc["blocks"]),
            "chunks": len(cd.get("chunks", [])),
            "findings": len(findings),
            "overlap": overlap,
            "oversized": len(oversized),
            "efficiency": ca.efficiency_metrics(cd, overlap, pj),
        }

    # Hipótesis H1–H5
    editorial_indexable = [
        (nid, b["block_id"])
        for nid, doc in docs.items()
        for b in doc["blocks"]
        if b["block_type"] in ca.EDITORIAL_TYPES and (b.get("retrieval") or {}).get("indexable")
    ]
    table_blocks = [
        (nid, b["block_id"]) for nid, doc in docs.items() for b in doc["blocks"] if _has_table(b)
    ]
    note_leaks = [f for f in all_findings if f["check"] in ("block.note_leak", "chunk.note_leak")]
    hier = ca.hierarchy_stats(docs)
    hypotheses = {
        "H1_nota_inicial_indexable": {
            "verdict": "CONFIRMADO" if editorial_indexable else "DESCARTADO",
            "affected": editorial_indexable,
            "severity": "WARN",
            "classification": "Revisar antes de embeddings",
        },
        "H2_hierarchy_incompleta": {
            "verdict": "CONFIRMADO" if hier["norms_with_unhandled_hierarchy"] else "DESCARTADO",
            "norms_bloqueantes": hier["norms_with_unhandled_hierarchy"],
            "norms_singular_label": hier.get("norms_with_singular_labels", {}),
            "headings_without_full_title": hier["headings_without_full_title"],
            "severity": "WARN",
            "classification": "Revisar antes de embeddings",
        },
        "H3_oversized": {
            "verdict": "CONFIRMADO" if oversized_all else "DESCARTADO",
            "total": len(oversized_all),
            "single_paragraph": sum(1 for r in oversized_all if r["single_paragraph_oversized"]),
            "max_text_chars": max((r["text_chars"] for r in oversized_all), default=0),
            "severity": "WARN",
            "classification": "Revisar antes de embeddings (decisión por tokens en indexación)",
        },
        "H4_note_leak": {
            "verdict": "DESCARTADO (falso positivo)" if not note_leaks else "CONFIRMADO",
            "exact_leaks": len(note_leaks),
            "severity": "INFO" if not note_leaks else "ERROR",
            "classification": "Correcto" if not note_leaks else "Revisar antes de embeddings",
        },
        "H5_tablas_linealizadas": {
            "verdict": "CONFIRMADO" if table_blocks else "DESCARTADO",
            "n_blocks": len(table_blocks),
            "severity": "INFO",
            "classification": "Aceptable MVP",
        },
    }

    examples = [
        {"categoria": cat, **ca.trace_block(RAW_DIR, nid, bid, docs[nid], chunks.get(nid, {}))}
        for cat, (nid, bid) in _select_examples(docs, chunks)
    ]

    temporal = ca.temporal_integrity(docs, chunks)
    raw = ca.raw_integrity(list(docs.keys()), MANIFEST_DIR)
    readiness = _readiness(all_findings, editorial_indexable, hier, temporal, raw)

    report = {
        "corpus": {
            "n_norms": len(docs),
            "total_blocks": sum(len(d["blocks"]) for d in docs.values()),
            "total_chunks": sum(len(c.get("chunks", [])) for c in chunks.values()),
        },
        "summary": ca.summarize(all_findings),
        "block_type_stats": ca.block_type_stats(docs),
        "hierarchy_stats": hier,
        "hypotheses": hypotheses,
        "metadata_classification": ca.classify_metadata(),
        "oversized_summary": {
            "total": len(oversized_all),
            "by_norm": _count_by(oversized_all, "document_id"),
            "by_block_type": _count_by(oversized_all, "block_type"),
        },
        "per_norm": per_norm,
        "representative_examples": examples,
        "raw_integrity": raw,
        "temporal_integrity": temporal,
        "pre_embedding_readiness": readiness,
        "findings": all_findings,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "mvp_chunking_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(oversized_all, REPORTS_DIR / "mvp_chunking_audit.csv")

    _print_summary(report)
    if strict and not report["pre_embedding_readiness"]["ready"]:
        print("\n[--strict] pre_embedding_readiness.ready=false → exit 1", file=sys.stderr)
        return 1
    return 0


def _corpus_catalogs() -> list[str]:
    """Catálogos de corpus presentes (debe haber exactamente uno: el canónico)."""
    candidates = [
        "data/corpus/seed_corpus.json",
        "config/corpus_mvp.json",
        "data/corpus/mvp_corpus.json",
        "data/corpus/corpus_mvp.json",
    ]
    return [p for p in candidates if Path(p).is_file()]


def _readiness(
    findings: list[dict],
    editorial_indexable: list,
    hier: dict,
    temporal: dict,
    raw: dict,
) -> dict:
    """Calcula `pre_embedding_readiness` (bloqueantes vs diferidos) + catálogos."""
    catalogs = _corpus_catalogs()
    readiness = ca.compute_readiness(
        findings,
        hier["norms_with_unhandled_hierarchy"],
        editorial_indexable,
        duplicate_catalog=len(catalogs) > 1,
        temporal=temporal,
        raw=raw,
    )
    readiness["corpus_catalogs"] = catalogs
    return readiness


def _count_by(rows: list[dict], key: str) -> dict:
    from collections import Counter

    c: Counter = Counter(r[key] for r in rows)
    return dict(c)


def _write_csv(rows: list[dict], path: Path) -> None:
    cols = [
        "document_id",
        "block_id",
        "block_type",
        "chunk_id",
        "text_chars",
        "retrieval_text_chars",
        "words_count",
        "paragraphs_count",
        "single_paragraph_oversized",
        "max_chars_excess",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _print_summary(report: dict) -> None:
    s = report["summary"]
    print("=== Auditoría del corpus MVP ===")
    print(
        f"normas: {report['corpus']['n_norms']} | bloques: {report['corpus']['total_blocks']} | "
        f"chunks: {report['corpus']['total_chunks']}"
    )
    print(f"hallazgos: {s['total']} | por severidad: {s['by_severity']}")
    print(f"por clasificación: {s['by_classification']}")
    print("\nHipótesis:")
    for h, v in report["hypotheses"].items():
        print(f"  {h}: {v['verdict']}")
    print(
        f"\noversized: {report['oversized_summary']['total']} "
        f"(por tipo: {report['oversized_summary']['by_block_type']})"
    )
    raw = report["raw_integrity"]
    temp = report["temporal_integrity"]
    quarantine = (
        len(temp["ambiguous_blocks"]) + len(temp["missing_index_date"]) + len(temp["index_not_max"])
    )
    print(
        f"\nraw_integrity.ready: {raw['ready']} (files_checked={raw['files_checked']}, "
        f"sha256_mismatches={len(raw['sha256_mismatches'])})"
    )
    print(
        f"temporal_integrity.ready: {temp['ready']} "
        f"(versioned={temp['versioned_blocks']}, "
        f"no_cronologicos={len(temp['non_chronological_xml_order_blocks'])}, "
        f"mismatches={len(temp['mismatches'])}, cuarentena_irresoluble={quarantine}, "
        f"chunks_no_vigentes={len(temp['chunks_built_from_non_current_version'])}, "
        f"vigencia_futura={len(temp['future_effective_selected_versions'])})"
    )
    r = report["pre_embedding_readiness"]
    print(f"\npre_embedding_readiness.ready: {r['ready']}")
    print(f"  blocking_findings: {r['blocking_findings']}")
    print(f"  deferred_findings: {r['deferred_findings']}")
    print("Reportes: data/processed/reports/mvp_chunking_audit.{json,csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auditoría del corpus MVP (solo lectura).")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit != 0 si pre_embedding_readiness.ready es false (cierre/CI local).",
    )
    args = parser.parse_args()
    raise SystemExit(main(strict=args.strict))
