"""Configuración del proyecto cargada con pydantic-settings.

Todos los valores tienen un default seguro para desarrollo local, de modo que
`Settings()` se pueda instanciar sin `.env` ni secretos. Las variables se pueden
sobrescribir mediante variables de entorno o un fichero `.env` (ver `.env.example`).
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración tipada del sistema RAG Legal BOE."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API BOE
    boe_api_base: str = "https://www.boe.es/datosabiertos/api"

    # Logging
    log_level: str = "INFO"

    # Ollama (LLM local)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral"

    # Embeddings
    embedding_model: str = "intfloat/multilingual-e5-large"

    # Vector store
    vector_store_provider: str = "chroma"
    chroma_persist_dir: str = "data/indexes/chroma"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "boe_consolidado"


@lru_cache
def get_settings() -> Settings:
    """Devuelve una instancia cacheada de `Settings`."""
    return Settings()
