"""Configuración del proyecto cargada con pydantic-settings.

Todos los valores tienen un default seguro para desarrollo local, de modo que
`Settings()` se pueda instanciar sin `.env` ni secretos. Las variables se pueden
sobrescribir mediante variables de entorno o un fichero `.env` (ver `.env.example`).
"""

from functools import lru_cache
from urllib.parse import urlparse

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.core.exceptions import ConfigurationError
from src.retrieval.context_assembler import STRATEGIES

# Hosts considerados locales (loopback). Cualquier otro requiere opt-in explícito.
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


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

    # Ollama (LLM local de generación — Fase 3). Default loopback; no se expone a red.
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3:4b-instruct"
    ollama_timeout_seconds: float = 900.0
    ollama_keep_alive: str = "5m"
    ollama_num_ctx: int = 4096
    ollama_num_predict: int = 256
    ollama_temperature: float = 0.0
    ollama_seed: int = 42
    ollama_think: bool = False
    # Salvaguarda: solo se permite una URL de Ollama no-loopback con opt-in explícito.
    ollama_allow_remote: bool = False

    # Generación fundamentada (Fase 3). El bundle NO tiene default: se indica por entorno o CLI.
    generation_dense_bundle: str | None = None
    generation_query_profile_id: str = "I2_CITIZEN_LEGISLATION"
    generation_top_k: int = 5
    generation_max_evidences: int = 3
    generation_context_strategy: str = "P_EXPAND_BOUNDED"
    generation_context_budget_chars: int = 4000
    generation_max_total_context_chars: int = 8000

    # Índice denso (el modelo se elige por CLI, no por default global)
    dense_index_root: str = "data/indexes/dense"
    default_cpu_threads: int = 8
    max_cpu_threads: int = 16

    @model_validator(mode="after")
    def _validate_ollama_base_url(self) -> "Settings":
        """Valida la forma de OLLAMA_BASE_URL y exige loopback salvo opt-in explícito."""
        parsed = urlparse(self.ollama_base_url)
        if parsed.scheme not in {"http", "https"}:
            raise ConfigurationError(
                f"OLLAMA_BASE_URL debe usar esquema http/https (recibido {parsed.scheme!r})."
            )
        if not parsed.hostname:
            raise ConfigurationError("OLLAMA_BASE_URL no contiene un host válido.")
        try:
            port = parsed.port
        except ValueError as exc:
            raise ConfigurationError(
                f"OLLAMA_BASE_URL contiene un puerto inválido: {self.ollama_base_url!r}."
            ) from exc
        if port is not None and port <= 0:
            raise ConfigurationError(f"OLLAMA_BASE_URL contiene un puerto no utilizable: {port}.")
        if parsed.username or parsed.password:
            raise ConfigurationError("OLLAMA_BASE_URL no debe incluir credenciales (user:pass).")
        if parsed.query or parsed.fragment:
            raise ConfigurationError("OLLAMA_BASE_URL no debe incluir query ni fragment.")
        if parsed.path not in ("", "/"):
            raise ConfigurationError(
                f"OLLAMA_BASE_URL debe tener path vacío o '/' (recibido {parsed.path!r})."
            )
        if parsed.hostname.lower() not in _LOOPBACK_HOSTS and not self.ollama_allow_remote:
            raise ConfigurationError(
                f"OLLAMA_BASE_URL apunta a un host no local ({parsed.hostname!r}). El MVP usa "
                "loopback; para un host remoto define OLLAMA_ALLOW_REMOTE=true."
            )
        return self

    @model_validator(mode="after")
    def _validate_generation_params(self) -> "Settings":
        """Rechaza valores de generación inválidos (positivos + estrategia conocida)."""
        positive = {
            "generation_top_k": self.generation_top_k,
            "generation_max_evidences": self.generation_max_evidences,
            "generation_context_budget_chars": self.generation_context_budget_chars,
            "generation_max_total_context_chars": self.generation_max_total_context_chars,
            "ollama_timeout_seconds": self.ollama_timeout_seconds,
            "ollama_num_ctx": self.ollama_num_ctx,
            "ollama_num_predict": self.ollama_num_predict,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ConfigurationError(f"{name} debe ser > 0 (recibido {value}).")
        if self.generation_context_strategy not in STRATEGIES:
            raise ConfigurationError(
                f"GENERATION_CONTEXT_STRATEGY inválida: {self.generation_context_strategy!r} "
                f"(esperado uno de {STRATEGIES})."
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Devuelve una instancia cacheada de `Settings`."""
    return Settings()
