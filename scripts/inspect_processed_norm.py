"""Vista humana consolidada de una norma procesada (join de los 4 artefactos v2).

Uso:
    uv run python scripts/inspect_processed_norm.py BOE-A-1985-5392
    uv run python scripts/inspect_processed_norm.py BOE-A-1985-5392 --block a45

Reconstruye por **joins** (sin persistir duplicados) `document + history + parents + chunks` y
muestra una vista legible por norma: metadatos, bloques (incl. los **excluidos** con su texto
desde parents) y, por bloque, su estado temporal, indexabilidad, cita y nº de chunks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DOCS_DIR = Path("data/processed/documents")
HISTORIES_DIR = Path("data/processed/histories")
PARENTS_DIR = Path("data/processed/parents")
CHUNKS_DIR = Path("data/processed/chunks")


def _load(norm_id: str) -> tuple[dict, dict, dict, dict]:
    def rd(d: Path) -> dict:
        path = d / f"{norm_id}.json"
        if not path.is_file():
            raise SystemExit(f"Falta artefacto: {path} (ejecuta process_mvp_corpus.py)")
        return json.loads(path.read_text(encoding="utf-8"))

    return rd(DOCS_DIR), rd(HISTORIES_DIR), rd(PARENTS_DIR), rd(CHUNKS_DIR)


def main() -> int:
    parser = argparse.ArgumentParser(description="Vista humana de una norma procesada (v2).")
    parser.add_argument("norm_id")
    parser.add_argument("--block", help="muestra el detalle de un block_id concreto.")
    args = parser.parse_args()

    document, history, parents, chunks = _load(args.norm_id)
    parents_by_block = {p["block_id"]: p for p in parents.get("parents", [])}
    hist_by_block = {h["block_id"]: h for h in history.get("blocks", [])}
    chunks_by_block: dict[str, list] = {}
    for c in chunks.get("chunks", []):
        chunks_by_block.setdefault(c["block_id"], []).append(c)

    meta = document.get("metadata", {})
    print(f"# {meta.get('title')}")
    print(
        f"  id={document['document_id']} · rango={(meta.get('rank') or {}).get('label')} "
        f"· estado={(meta.get('consolidation_status') or {}).get('label')}"
    )
    print(
        f"  bloques={len(document['blocks'])} · history={len(history['blocks'])} "
        f"· parents={len(parents['parents'])} · chunks={len(chunks.get('chunks', []))}"
    )
    print(f"  manifest_ref={document['source'].get('manifest_ref')}")

    if args.block:
        b = next((x for x in document["blocks"] if x["block_id"] == args.block), None)
        if not b:
            raise SystemExit(f"block_id {args.block!r} no existe en {args.norm_id}")
        p = parents_by_block.get(args.block)
        h = hist_by_block.get(args.block)
        print(f"\n== bloque {args.block} ({b.get('block_type')}) ==")
        print(f"  título: {b.get('full_title') or b.get('block_title')}")
        print(
            f"  indexable={b.get('indexable')} excluded_reason={b.get('excluded_reason')} "
            f"temporal_status={b.get('temporal_status')}"
        )
        print(f"  cita: {b.get('citation', {}).get('label')} · {b.get('citation', {}).get('url')}")
        if h:
            print(
                f"  versiones: {len(h.get('versions', []))} · "
                f"notas: {len(h.get('modification_notes', []))}"
            )
        if p:
            print(f"  texto vigente ({len(p.get('text', ''))} car.):")
            print("    " + (p.get("text", "")[:400].replace("\n", "\n    ")))
        print(f"  chunks: {len(chunks_by_block.get(args.block, []))}")
        return 0

    # Resumen por bloque (incluye excluidos con texto desde parents).
    print("\nBloques (orden documental):")
    for b in document["blocks"]:
        bid = b["block_id"]
        p = parents_by_block.get(bid)
        nchunks = len(chunks_by_block.get(bid, []))
        tag = "IDX" if b.get("indexable") else ("TXT" if p else "—")
        text_chars = len(p.get("text", "")) if p else 0
        print(
            f"  [{tag}] {bid:10} {b.get('block_type'):12} "
            f"chunks={nchunks:<3} texto={text_chars:<6} {b.get('block_title') or ''}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
