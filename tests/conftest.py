"""Fixtures compartidas de la suite de tests."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Ruta raíz del repositorio (carpeta que contiene `pyproject.toml`)."""
    return Path(__file__).resolve().parent.parent
