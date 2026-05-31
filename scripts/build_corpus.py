"""Construye el corpus MVP: descarga, verifica y procesa las normas del corpus semilla.

Uso:
    uv run python scripts/build_corpus.py

Llama a la API externa del BOE. Para cada norma del corpus semilla:
  1. descarga el raw (endpoints opcionales tolerados) + manifest;
  2. la verifica contra los criterios (vigente + estado_consolidacion "Finalizado");
  3. si los cumple, genera el documento procesado y los chunks.

Las normas que NO cumplen criterios se EXCLUYEN del procesado y se reportan (no se
sustituyen automáticamente). Escribe `data/corpus/verification_report.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Permite `import src...` al ejecutar el script directamente.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.boe.client import BoeClient  # noqa: E402
from src.boe.corpus import load_seed_corpus, verify_norm  # noqa: E402
from src.boe.parser import parse_boe_document, save_processed_document  # noqa: E402
from src.config.settings import get_settings  # noqa: E402
from src.core.exceptions import BoeApiError, ParsingError  # noqa: E402
from src.preprocessing.chunker import create_chunks, save_chunks  # noqa: E402

RAW_DIR = Path("data/raw/boe")
MANIFEST_DIR = Path("data/manifests")
DOCUMENTS_DIR = Path("data/processed/documents")
CHUNKS_DIR = Path("data/processed/chunks")
REPORT_PATH = Path("data/corpus/verification_report.json")

# Endpoints opcionales: su ausencia no invalida la norma ni aborta la descarga.
OPTIONAL_DOWNLOAD = frozenset({"analisis", "metadata_eli", "full"})


def _acquire_and_verify(client: BoeClient, norm_id: str) -> dict:
    """Descarga el raw + manifest y devuelve la fila de verificación de una norma."""
    try:
        downloaded = client.download_norm_raw(norm_id, optional_endpoints=OPTIONAL_DOWNLOAD)
        client.write_manifest(norm_id, downloaded)
    except BoeApiError as exc:
        return {
            "norm_id": norm_id,
            "exists": False,
            "availability": {},
            "meets_criteria": False,
            "reasons": [f"error de descarga: {exc}"],
        }
    return verify_norm(norm_id, RAW_DIR, downloaded)


def _process(norm_id: str) -> tuple[int, int]:
    """Parsea y chunkea una norma ya descargada. Devuelve (n_bloques, n_chunks)."""
    document = parse_boe_document(norm_id, RAW_DIR, MANIFEST_DIR / f"{norm_id}.json")
    save_processed_document(document, DOCUMENTS_DIR)
    chunks_document = create_chunks(document)
    save_chunks(chunks_document, CHUNKS_DIR)
    return len(document["blocks"]), chunks_document["quality_checks"]["chunks_count"]


def _format_row(row: dict) -> str:
    rank = (row.get("rank") or {}).get("label", "—")
    status = (row.get("consolidation_status") or {}).get("label", "—")
    flag = "OK " if row["meets_criteria"] else "REV"
    title = (row.get("title") or "—")[:48]
    return f"  [{flag}] {row['norm_id']:18} {rank:24} {status:12} {title}"


def main(strict: bool = False) -> int:
    settings = get_settings()
    norms = load_seed_corpus()

    report: list[dict] = []
    with BoeClient(base_url=settings.boe_api_base) as client:
        for norm in norms:
            norm_id = norm["norm_id"]
            print(f"Descargando y verificando {norm_id} ...")
            row = _acquire_and_verify(client, norm_id)
            if row["meets_criteria"]:
                try:
                    blocks, chunks = _process(norm_id)
                    row["blocks_count"] = blocks
                    row["chunks_count"] = chunks
                except ParsingError as exc:
                    row["processing_error"] = str(exc)
                    print(f"  ⚠ error al procesar {norm_id}: {exc}", file=sys.stderr)
            report.append(row)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    passed = [r for r in report if r["meets_criteria"]]
    failed = [r for r in report if not r["meets_criteria"]]

    print("\n=== Verificación del corpus ===")
    for row in report:
        print(_format_row(row))
    print(f"\nCumplen criterios y procesadas: {len(passed)}/{len(report)}")
    print(f"Reporte: {REPORT_PATH}")

    if failed:
        print("\n⚠ Normas a REVISAR (no procesadas, no sustituidas):")
        for row in failed:
            print(f"  - {row['norm_id']}: {', '.join(row['reasons']) or 'ver reporte'}")

    processing_errors = [r for r in report if r.get("processing_error")]
    if strict and (failed or processing_errors):
        print(
            f"\n[--strict] {len(failed)} norma(s) a revisar, "
            f"{len(processing_errors)} con error de proceso → exit 1",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construye el corpus MVP (descarga + proceso).")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit != 0 si alguna norma no cumple criterios o falla el proceso.",
    )
    args = parser.parse_args()
    raise SystemExit(main(strict=args.strict))
