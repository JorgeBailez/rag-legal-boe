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
    "data/raw/boe",
    "data/processed/documents",
    "data/evaluation",
    "data/manifests",
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
    "src/app/api.py",
    "src/core/exceptions.py",
    "src/config/settings.py",
    "docs/decisiones_tecnicas.md",
    "docs/fuentes_y_licencias.md",
    "docs/evaluacion.md",
    "docs/known_issues.md",
    "prompts/system_prompt.txt",
    "prompts/rag_prompt.txt",
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
    assert dumped["vector_store_provider"] == "chroma"


def test_core_exceptions_hierarchy() -> None:
    from src.core.exceptions import (
        BoeApiError,
        ExternalServiceError,
        RagLegalBoeError,
    )

    assert issubclass(ExternalServiceError, RagLegalBoeError)
    assert issubclass(BoeApiError, ExternalServiceError)
