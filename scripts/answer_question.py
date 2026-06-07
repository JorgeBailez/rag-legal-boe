"""CLI del MVP de generación fundamentada (Fase 3): pregunta → respuesta con citas o abstención.

Capa DELGADA sobre los servicios desacoplados (retrieval denso + evidencias + prompt + Ollama +
orquestador). Requiere, para funcionar de verdad, un bundle denso publicado, los pesos del modelo
de embeddings y un Ollama local en marcha — por eso se ejecuta en el servidor, no en los tests.

Uso:
    uv run python scripts/answer_question.py \
      --bundle data/indexes/dense/<bundle_id> \
      --query "¿Qué plazo tengo para interponer un recurso de alzada?"

Códigos de salida: 0 si hay respuesta o abstención válida; ≠0 ante fallo técnico (bundle inválido,
Ollama caído, violación de contrato del LLM, etc.). Una abstención NO es un fallo técnico.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import get_settings  # noqa: E402
from src.contracts.generation_models import RagAnswerV1  # noqa: E402
from src.core.exceptions import RagLegalBoeError  # noqa: E402
from src.embeddings.encoder import set_cpu_threads  # noqa: E402
from src.generation.answer_generator import AnswerGenerator, GenerationConfig  # noqa: E402
from src.generation.ollama_client import OllamaClient  # noqa: E402
from src.retrieval.dense_retriever import DenseRetriever  # noqa: E402


def _positive_int(value: str) -> int:
    """Tipo argparse: entero estrictamente positivo (falla en el parseo del CLI, no más tarde)."""
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"valor entero inválido: {value!r}") from None
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"debe ser un entero > 0 (recibido {ivalue}).")
    return ivalue


def _non_blank(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError(
            "la pregunta no puede estar vacía ni contener solo espacios."
        )
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Responde una pregunta con RAG fundamentado.")
    parser.add_argument("--bundle", help="ruta al bundle; fallback a GENERATION_DENSE_BUNDLE.")
    parser.add_argument(
        "--query", required=True, type=_non_blank, help="pregunta en lenguaje natural."
    )
    parser.add_argument("--query-profile-id", default=None)
    parser.add_argument("--top-k", type=_positive_int, default=None)
    parser.add_argument("--max-evidences", type=_positive_int, default=None)
    parser.add_argument(
        "--context-strategy",
        default=None,
        choices=["K_ONLY", "P_EXPAND_FULL", "P_EXPAND_BOUNDED"],
    )
    parser.add_argument("--context-budget-chars", type=_positive_int, default=None)
    parser.add_argument("--max-total-context-chars", type=_positive_int, default=None)
    parser.add_argument("--threads", type=_positive_int, default=None)
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument("--json", action="store_true", help="imprime RagAnswerV1 completo en JSON.")
    parser.add_argument("--debug", action="store_true", help="añade scores, evidencias y métricas.")
    parser.add_argument(
        "--unload-model", action="store_true", help="descarga el modelo de Ollama al terminar."
    )
    return parser.parse_args()


def _print_human(ans: RagAnswerV1) -> None:
    if ans.answered:
        print("Respuesta:")
        print(ans.answer)
        print("\nFuentes:")
        for c in ans.citations:
            print(f"- {c.label}")
            if c.url:
                print(f"  {c.url}")
        print("\nAviso:")
        print(ans.disclaimer)
    else:
        print(
            "No dispongo de evidencia suficiente en el corpus indexado para responder con "
            "fiabilidad."
        )
        print(f"Motivo: {ans.abstention_reason}")
        print("\nAviso:")
        print(ans.disclaimer)


def _print_debug(ans: RagAnswerV1) -> None:
    t = ans.retrieval_trace
    print("\n[debug] retrieval", file=sys.stderr)
    print(
        f"  bundle={t.bundle_id} modelo={t.model_alias} perfil={t.query_profile_id} "
        f"top_k={t.top_k} hits={t.returned_hits} evidencias={t.selected_evidences}",
        file=sys.stderr,
    )
    for h in t.hits:
        mark = f" -> {h.evidence_id}" if h.selected else ""
        print(
            f"  #{h.rank} score={h.score:.4f} {h.document_id} {h.block_id}{mark}",
            file=sys.stderr,
        )
    print(
        f"[debug] evidencias: duplicados_eliminados={t.duplicate_parents_removed} "
        f"caracteres_contexto_total={t.total_context_chars}",
        file=sys.stderr,
    )
    for o in t.omitted_evidences:
        cc = f" chars={o.char_count}" if o.char_count is not None else ""
        print(
            f"  [omitida] #{o.retrieval_rank} {o.parent_id} reason={o.reason}{cc}",
            file=sys.stderr,
        )
    if ans.generation_metrics is not None:
        m = ans.generation_metrics
        print(
            f"[debug] ollama total={m.total_duration_s:.2f}s load={m.load_duration_s:.2f}s "
            f"eval_tokens={m.eval_count} tok/s={m.tokens_per_second:.2f}",
            file=sys.stderr,
        )


def main() -> int:
    args = _parse_args()
    settings = get_settings()

    bundle = args.bundle if args.bundle is not None else settings.generation_dense_bundle
    if not bundle:
        print(
            "Falta el bundle denso. Indica --bundle o define GENERATION_DENSE_BUNDLE en .env.",
            file=sys.stderr,
        )
        return 2

    threads = args.threads if args.threads is not None else settings.default_cpu_threads
    set_cpu_threads(threads)

    def _override(value, default):  # noqa: ANN001, ANN202 - helper local de override explícito
        return value if value is not None else default

    config = GenerationConfig(
        query_profile_id=_override(args.query_profile_id, settings.generation_query_profile_id),
        top_k=_override(args.top_k, settings.generation_top_k),
        max_evidences=_override(args.max_evidences, settings.generation_max_evidences),
        context_strategy=_override(args.context_strategy, settings.generation_context_strategy),
        context_budget_chars=_override(
            args.context_budget_chars, settings.generation_context_budget_chars
        ),
        max_total_context_chars=_override(
            args.max_total_context_chars, settings.generation_max_total_context_chars
        ),
        temperature=settings.ollama_temperature,
        seed=settings.ollama_seed,
        num_predict=settings.ollama_num_predict,
        num_ctx=settings.ollama_num_ctx,
        keep_alive=settings.ollama_keep_alive,
    )

    try:
        retriever = DenseRetriever.from_bundle(bundle, batch_size=args.batch_size)
    except Exception as exc:  # noqa: BLE001 - fallo técnico de carga del bundle/modelo
        print(f"No se pudo cargar el bundle/modelo: {exc}", file=sys.stderr)
        return 1

    client = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout=settings.ollama_timeout_seconds,
        think=settings.ollama_think,
        keep_alive=settings.ollama_keep_alive,
    )
    generator = AnswerGenerator(retriever=retriever, llm_client=client, config=config)

    exit_code = 0
    try:
        answer = generator.answer(args.query, query_profile_id=config.query_profile_id)
        if args.json:
            print(answer.model_dump_json(indent=2))
        else:
            _print_human(answer)
        if args.debug:
            _print_debug(answer)
    except RagLegalBoeError as exc:
        print(f"Fallo de generación: {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        if args.unload_model:
            try:
                client.unload()
            except RagLegalBoeError as exc:
                print(f"Aviso: no se pudo descargar el modelo: {exc}", file=sys.stderr)
        client.close()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
