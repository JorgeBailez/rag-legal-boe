"""Genera un bundle de índice denso para un modelo y una vista (CPU, reproducible).

Uso habitual (limpio):
    uv run python scripts/generate_dense_index.py --model e5-large-instruct

Implica por defecto: view J1, device cpu, threads 8, barra de progreso, overflow_policy=repair,
salida en data/indexes/dense.

Flujo: resolver alias → Gate A → preparar inputs → codificar (con progreso) → bundle en staging →
Gate B → publicar (rename atómico) → imprimir próximos comandos. El preflight valida sin cargar los
pesos del encoder:
    uv run python scripts/generate_dense_index.py --model e5-large-instruct --preflight-only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings import model_registry as reg  # noqa: E402
from src.embeddings.bundle import (  # noqa: E402
    BundleExistsError,
    ExecutionMeta,
    publish_bundle,
)
from src.embeddings.corpus_loader import load_processed_corpus, load_readiness  # noqa: E402
from src.embeddings.encoder import DenseEncoder, load_tokenizer, set_cpu_threads  # noqa: E402
from src.embeddings.fingerprints import source_corpus_fingerprint  # noqa: E402
from src.embeddings.input_preparation import prepare_inputs  # noqa: E402
from src.embeddings.validation import has_errors, run_gate_a, summarize_severity  # noqa: E402


def _print_models() -> int:
    print(f"{'alias':22} {'model_id':40} {'dim':>5} {'max_tok':>8} {'remote':>7}  notas")
    for c in reg.list_models():
        print(
            f"{c.alias:22} {c.model_id:40} {c.expected_embedding_dimension:>5} "
            f"{c.declared_max_tokens:>8} {str(c.trust_remote_code):>7}  {c.notes}"
        )
    return 0


def _print_findings(findings: list[dict]) -> None:
    for f in findings:
        if f["severity"] != "INFO":
            ev = f" [{f['evidence']}]" if f.get("evidence") else ""
            print(f"  {f['severity']:7} {f['gate']}.{f['check']}: {f['message']}{ev}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Genera un bundle de índice denso.")
    parser.add_argument("--model", help="alias corto o model_id (obligatorio salvo --list-models).")
    parser.add_argument("--view", default="J1", choices=["J1", "J2", "C1"])
    parser.add_argument(
        "--preflight-only", action="store_true", help="valida sin cargar los pesos del encoder."
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        default="cpu",
        help="dispositivo de torch para el encoder: 'cpu' (default) o 'cuda' (GPU).",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--output-root", default="data/indexes/dense")
    parser.add_argument(
        "--allow-unpinned-revision",
        action="store_true",
        help="permite preflight exploratorio sin commit hash fijado; nunca publica.",
    )
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()

    if args.list_models:
        return _print_models()
    if not args.model:
        parser.error("--model es obligatorio (o usa --list-models).")

    contract = reg.get_contract(args.model)
    print(f"modelo: {contract.alias} ({contract.model_id}) | view: {args.view}")

    # 1) Corpus + tokenizer (ligero) + preparación de inputs.
    corpus = load_processed_corpus()
    if not corpus["chunks"]:
        print("No hay corpus procesado en data/processed (ejecuta Fase 1).", file=sys.stderr)
        return 1
    tokenizer = load_tokenizer(contract, allow_unpinned_revision=args.allow_unpinned_revision)
    prepared = prepare_inputs(
        args.view,
        chunks=corpus["chunks"],
        parents_by_id=corpus["parents_by_id"],
        contract=contract,
        tokenizer=tokenizer,
    )
    print(
        f"inputs: {prepared.report['n_rows']} (derivados: {prepared.report['n_derived_rows']}, "
        f"reparados overflow: {prepared.report['n_overflow_repaired_inputs']}, "
        f"truncados: {prepared.report['n_truncated']}, "
        f"max_tokens: {prepared.report['max_token_count']}/{prepared.effective_max_tokens})"
    )

    # 2) Gate A (pre-encoding).
    readiness = load_readiness()
    gate_a = run_gate_a(
        readiness=readiness,
        contract=contract,
        allow_unpinned_revision=args.allow_unpinned_revision,
        prepared=prepared,
    )
    print(f"Gate A: {summarize_severity(gate_a)}")
    _print_findings(gate_a)
    if has_errors(gate_a):
        print("\nGate A con errores → no se codifica.", file=sys.stderr)
        return 1
    if args.preflight_only:
        print("\nPreflight OK (no se codifica; no se publica bundle).")
        return 0
    if args.allow_unpinned_revision:
        print(
            "\n--allow-unpinned-revision es solo exploratorio: no se codifica ni publica bundle.",
            file=sys.stderr,
        )
        return 1

    # 3) Codificación (con barra de progreso por defecto).
    set_cpu_threads(args.threads)
    encoder = DenseEncoder(
        contract,
        device=args.device,
        batch_size=args.batch_size,
        allow_unpinned_revision=args.allow_unpinned_revision,
    )
    print(
        f"\n[encoding] {contract.alias} {args.view}  ({len(prepared.texts)} inputs, "
        f"device={args.device}, threads={args.threads}, batch={args.batch_size})"
    )
    t0 = time.perf_counter()
    embeddings = encoder.encode_documents(prepared.texts, show_progress=not args.no_progress)
    duration = time.perf_counter() - t0

    # 4) Publicación (Gate B + rename atómico).
    execution = ExecutionMeta(
        device=args.device,
        threads=args.threads,
        batch_size=args.batch_size,
        duration_seconds=duration,
        encoder_backend=DenseEncoder.backend,
        allow_unpinned_revision=args.allow_unpinned_revision,
    )
    try:
        result = publish_bundle(
            contract=contract,
            view=args.view,
            prepared=prepared,
            embeddings=embeddings,
            source_corpus_fingerprint=source_corpus_fingerprint(
                corpus["chunks"], corpus["parents_by_id"]
            ),
            n_norms=corpus["n_norms"],
            execution=execution,
            output_root=Path(args.output_root),
            gate_a_findings=gate_a,
        )
    except BundleExistsError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

    man = result["manifest"]
    path = result["path"]
    print("\n=== bundle publicado ===")
    print(f"  ruta       : {path}")
    print(f"  modelo     : {man['bundle']['model_alias']} ({man['bundle']['model_id']})")
    print(f"  view       : {man['bundle']['view']}")
    print(f"  vectores   : {man['artifacts']['n_rows']}")
    print(f"  dimensión  : {man['artifacts']['embedding_dimension']}")
    print(f"  duración   : {man['execution']['duration_seconds']} s")
    print(f"  throughput : {man['execution']['throughput_inputs_per_second']} inputs/s")
    print(f"  warnings   : {man['validation']['n_warnings']}")
    print("\nPróximos comandos:")
    print(f"  uv run python scripts/validate_dense_index.py --bundle {path}")
    print(
        f"  uv run python scripts/query_dense_index.py --bundle {path} "
        f'--query "¿Cuánto tiempo tiene la Administración para responder a mi solicitud?"'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
