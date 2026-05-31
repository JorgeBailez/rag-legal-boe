"""Perfila los tokenizadores de los modelos candidatos sobre los chunks del corpus.

Uso:
    uv run python scripts/profile_tokenizers.py                  # todos los candidatos
    uv run python scripts/profile_tokenizers.py --models BAAI/bge-m3 intfloat/multilingual-e5-large

Mide, por contrato de cada modelo, los tokens del input que se embeberá
(`document_formatter(retrieval_text)`) y los compara con el límite efectivo del modelo, para
resolver H3_oversized_token_measurement. Escribe:
    data/processed/reports/tokenizer_profile.json
    data/processed/reports/tokenizer_profile.csv

REQUIERE red la primera vez (descarga los **tokenizers** —no los modelos— de Hugging Face) y la
dependencia `transformers`. La lógica de perfilado vive en `src/embeddings/` y es testeable sin
red con un fake tokenizer (ver `tests/test_tokenizer_profiler.py`).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings import model_registry as reg  # noqa: E402
from src.embeddings.tokenizer_profiler import profile_model  # noqa: E402

CHUNKS_DIR = Path("data/processed/chunks")
REPORTS_DIR = Path("data/processed/reports")


def _load_chunks(chunks_dir: Path) -> list[dict]:
    chunks: list[dict] = []
    for f in sorted(glob.glob(str(chunks_dir / "*.json"))):
        chunks.extend(json.loads(Path(f).read_text(encoding="utf-8")).get("chunks", []))
    return chunks


def _load_tokenizer(model_id: str, revision: str | None):
    """Carga el tokenizer real (import perezoso de transformers; puede requerir red)."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise SystemExit(
            "Falta 'transformers'. Autoriza e instala con: uv add transformers"
        ) from exc
    return AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)


def _csv_rows(report: dict) -> list[dict]:
    rows = []
    for m in report["models"]:
        emb = m["embedding_input_profile"]["overall"]
        rows.append(
            {
                "model_id": m["model_id"],
                "declared_max_tokens": m["declared_max_tokens"],
                "tokenizer_model_max_length": m["tokenizer_model_max_length"],
                "effective_max_tokens": m["effective_max_tokens"],
                "source_of_effective_limit": m["source_of_effective_limit"],
                "embedding_dim": m["expected_embedding_dimension"],
                "n_chunks": emb["n_items"],
                "n_truncated": emb["n_truncated"],
                "pct_truncated": emb["pct_truncated"],
                "max_tokens": emb["max_tokens"],
                "p95_tokens": emb["p95_tokens"],
                "p99_tokens": emb["p99_tokens"],
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Perfilado de tokenizadores (resuelve H3).")
    parser.add_argument("--models", nargs="*", default=reg.all_model_ids())
    parser.add_argument("--input", default=str(CHUNKS_DIR))
    parser.add_argument("--out", default=str(REPORTS_DIR))
    parser.add_argument(
        "--keep-per-chunk", action="store_true", help="incluye el detalle por chunk en el JSON."
    )
    args = parser.parse_args()

    chunks = _load_chunks(Path(args.input))
    if not chunks:
        print(f"No hay chunks en {args.input} (ejecuta process_mvp_corpus.py).", file=sys.stderr)
        return 1
    print(f"Chunks cargados: {len(chunks)}")

    models = []
    for model_id in args.models:
        contract = reg.get_contract(model_id)
        print(f"  perfilando {model_id} …")
        tokenizer = _load_tokenizer(model_id, contract.revision)
        models.append(
            profile_model(contract, tokenizer, chunks, keep_per_chunk=args.keep_per_chunk)
        )

    report = {"n_chunks": len(chunks), "n_models": len(models), "models": models}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tokenizer_profile.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows = _csv_rows(report)
    with (out_dir / "tokenizer_profile.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== Resumen (input de embedding) ===")
    for r in rows:
        print(
            f"  {r['model_id']:42} eff_max={r['effective_max_tokens']:>6} "
            f"({r['source_of_effective_limit']}) truncados={r['n_truncated']:>4} "
            f"({r['pct_truncated']}%) max={r['max_tokens']} p99={r['p99_tokens']}"
        )
    print(f"\nReportes: {out_dir}/tokenizer_profile.{{json,csv}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
