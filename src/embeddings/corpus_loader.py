"""Carga del corpus procesado de Fase 1 para la indexación densa (solo lectura, sin red).

Reúne chunks v2 + parents v2 + descriptors v2 de `data/processed/` en estructuras planas que
consumen la preparación de inputs, los filtros del índice y los gates.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
AUDIT_REPORT = PROCESSED_DIR / "reports" / "mvp_chunking_audit.json"


def load_processed_corpus(processed_dir: Path = PROCESSED_DIR) -> dict:
    """Devuelve {chunks, parents_by_id, documents_by_id, n_norms} desde data/processed/."""
    processed_dir = Path(processed_dir)
    chunks: list[dict] = []
    for f in sorted(glob.glob(str(processed_dir / "chunks" / "*.json"))):
        chunks.extend(json.loads(Path(f).read_text(encoding="utf-8")).get("chunks", []))
    parents_by_id: dict[str, dict] = {}
    for f in sorted(glob.glob(str(processed_dir / "parents" / "*.json"))):
        for p in json.loads(Path(f).read_text(encoding="utf-8")).get("parents", []):
            parents_by_id[p["parent_id"]] = p
    documents_by_id: dict[str, dict] = {}
    for f in sorted(glob.glob(str(processed_dir / "documents" / "*.json"))):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        documents_by_id[d["document_id"]] = d
    return {
        "chunks": chunks,
        "parents_by_id": parents_by_id,
        "documents_by_id": documents_by_id,
        "n_norms": len(documents_by_id),
    }


def load_readiness(audit_report: Path = AUDIT_REPORT) -> dict | None:
    """Lee `pre_embedding_readiness` del reporte de auditoría, o None si no existe."""
    audit_report = Path(audit_report)
    if not audit_report.is_file():
        return None
    data = json.loads(audit_report.read_text(encoding="utf-8"))
    return data.get("pre_embedding_readiness")
