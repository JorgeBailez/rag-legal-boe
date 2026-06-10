"""Verifica que el scaffold del proyecto existe y que la configuración importa.

No prueba lógica de negocio (aún no implementada): solo comprueba la base limpia
del repositorio para detectar regresiones de estructura.
"""

from pathlib import Path

import pytest

EXPECTED_DIRS = [
    "src/boe",
    "src/preprocessing",
    "src/indexing",
    "src/retrieval",
    "src/generation",
    "src/evaluation",
    "src/app",
    "src/core",
    "src/config",
    "src/contracts",
    "src/embeddings",
    "data/raw/boe",
    "data/processed/documents",
    "data/processed/histories",
    "data/processed/parents",
    "data/processed/chunks",
    "data/evaluation",
    "data/evaluation/dense_retrieval_v1",
    "data/manifests",
    "schemas",
    "docs",
    "notebooks",
    "tests/fixtures/boe",
    "prompts",
]

EXPECTED_FILES = [
    "pyproject.toml",
    ".python-version",
    ".gitignore",
    ".env.example",
    "README.md",
    "src/boe/__init__.py",
    "src/boe/client.py",
    "src/boe/parser.py",
    "src/preprocessing/chunker.py",
    "src/indexing/vector_index.py",
    "src/indexing/lexical_index.py",
    "src/retrieval/hybrid_retriever.py",
    "src/generation/prompt.py",
    "src/generation/answer_generator.py",
    "src/evaluation/metrics.py",
    "src/evaluation/generation_metrics.py",
    "src/evaluation/judge.py",
    "src/evaluation/generation_eval.py",
    "src/app/api.py",
    "src/core/exceptions.py",
    "src/config/settings.py",
    "src/contracts/models.py",
    "src/contracts/export_schemas.py",
    "src/contracts/embedding_models.py",
    "src/contracts/generation_models.py",
    "src/embeddings/tokenizer_profiler.py",
    "src/embeddings/model_registry.py",
    "src/embeddings/input_preparation.py",
    "src/embeddings/encoder.py",
    "src/embeddings/fingerprints.py",
    "src/embeddings/corpus_loader.py",
    "src/embeddings/bundle.py",
    "src/embeddings/validation.py",
    "src/retrieval/context_assembler.py",
    "src/retrieval/dense_retriever.py",
    "src/retrieval/evidence_builder.py",
    "src/generation/ollama_client.py",
    "src/evaluation/dataset.py",
    "src/evaluation/reports.py",
    "scripts/generate_dense_index.py",
    "scripts/validate_dense_index.py",
    "scripts/query_dense_index.py",
    "scripts/benchmark_dense_models.py",
    "scripts/validate_evaluation_dataset.py",
    "scripts/answer_question.py",
    "scripts/run_generation_eval.py",
    "scripts/audit_eval_dataset.py",
    "schemas/boe_legal_document_v2.schema.json",
    "schemas/boe_legal_chunks_v2.schema.json",
    "schemas/dense_embedding_bundle_v1.schema.json",
    "schemas/dense_embedding_row_v1.schema.json",
    "schemas/dense_embedding_validation_report_v1.schema.json",
    "schemas/rag_llm_answer_v1.schema.json",
    "schemas/rag_answer_v1.schema.json",
    "data/evaluation/dense_retrieval_v1/README.md",
    "data/evaluation/dense_retrieval_v1/questions.jsonl",
    "data/evaluation/dense_retrieval_v1/judgments.jsonl",
    "data/evaluation/dense_retrieval_v1/answer_keys.jsonl",
    "docs/decisiones_tecnicas.md",
    "docs/fuentes_y_licencias.md",
    "docs/evaluacion.md",
    "docs/known_issues.md",
    "docs/fase2_dense_baseline.md",
    "docs/run_dense_embeddings_server.md",
    "notebooks/README.md",
    "notebooks/02_perfilado_tokenizacion.ipynb",
    "notebooks/03_benchmark_modelos_densos.ipynb",
    "notebooks/04_ablaciones_chunking_y_contexto.ipynb",
    "notebooks/05_seleccion_baseline_dense.ipynb",
    "notebooks/06_evaluacion_generacion.ipynb",
    "prompts/system_prompt.txt",
    "prompts/rag_prompt.txt",
    "prompts/judge_faithfulness.txt",
    "prompts/judge_correctness.txt",
]


@pytest.mark.parametrize("rel_path", EXPECTED_DIRS)
def test_expected_directories_exist(project_root: Path, rel_path: str) -> None:
    assert (project_root / rel_path).is_dir(), f"Falta el directorio: {rel_path}"


@pytest.mark.parametrize("rel_path", EXPECTED_FILES)
def test_expected_files_exist(project_root: Path, rel_path: str) -> None:
    assert (project_root / rel_path).is_file(), f"Falta el fichero: {rel_path}"


def test_settings_import_and_defaults() -> None:
    from src.config.settings import Settings

    config = Settings()
    dumped = config.model_dump()

    assert dumped, "Settings.model_dump() no debe estar vacío"
    assert dumped["boe_api_base"].startswith("http")
    # Fase 2: defaults de índice denso (sin Chroma/Qdrant; el modelo se pasa con --model).
    assert dumped["dense_index_root"] == "data/indexes/dense"
    assert dumped["default_cpu_threads"] == 8
    assert dumped["max_cpu_threads"] == 16
    assert "vector_store_provider" not in dumped
    assert "embedding_model" not in dumped


def test_core_exceptions_hierarchy() -> None:
    from src.core.exceptions import (
        BoeApiError,
        ExternalServiceError,
        RagLegalBoeError,
    )

    assert issubclass(ExternalServiceError, RagLegalBoeError)
    assert issubclass(BoeApiError, ExternalServiceError)
