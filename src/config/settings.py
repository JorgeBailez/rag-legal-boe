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


def _validate_loopback_url(var_name: str, url: str, allow_remote: bool) -> None:
    """Valida la forma de una URL de servicio local y exige loopback salvo opt-in explícito."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ConfigurationError(
            f"{var_name} debe usar esquema http/https (recibido {parsed.scheme!r})."
        )
    if not parsed.hostname:
        raise ConfigurationError(f"{var_name} no contiene un host válido.")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ConfigurationError(f"{var_name} contiene un puerto inválido: {url!r}.") from exc
    if port is not None and port <= 0:
        raise ConfigurationError(f"{var_name} contiene un puerto no utilizable: {port}.")
    if parsed.username or parsed.password:
        raise ConfigurationError(f"{var_name} no debe incluir credenciales (user:pass).")
    if parsed.query or parsed.fragment:
        raise ConfigurationError(f"{var_name} no debe incluir query ni fragment.")
    if parsed.path not in ("", "/"):
        raise ConfigurationError(
            f"{var_name} debe tener path vacío o '/' (recibido {parsed.path!r})."
        )
    if parsed.hostname.lower() not in _LOOPBACK_HOSTS and not allow_remote:
        raise ConfigurationError(
            f"{var_name} apunta a un host no local ({parsed.hostname!r}). El MVP usa loopback; "
            "para un host remoto define OLLAMA_ALLOW_REMOTE=true."
        )


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

    # Ollama local para generación. Default loopback; no se expone a red.
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    ollama_timeout_seconds: float = 900.0
    ollama_keep_alive: str = "5m"
    ollama_num_ctx: int = 8192
    ollama_num_predict: int = 1536
    ollama_temperature: float = 0.0
    ollama_seed: int = 42
    ollama_think: bool = False
    # Salvaguarda: solo se permite una URL de Ollama no-loopback con opt-in explícito.
    ollama_allow_remote: bool = False

    # Generación fundamentada. El bundle se indica por entorno o CLI.
    generation_dense_bundle: str | None = None
    generation_query_profile_id: str = "I1_LEGAL"
    generation_top_k: int = 3
    generation_max_evidences: int = 3
    generation_context_strategy: str = "P_EXPAND_BOUNDED"
    generation_context_budget_chars: int = 4000
    generation_max_total_context_chars: int = 16000

    # Juez LLM de evaluación. Modelo configurable y sin default global.
    judge_base_url: str = "http://127.0.0.1:11434"
    judge_model: str = ""
    judge_timeout_seconds: float = 900.0
    judge_num_ctx: int = 8192
    # Tope alto: el veredicto de fidelidad enumera las afirmaciones de la respuesta; con respuestas
    # largas, 512 truncaba el JSON. Es un tope (no objetivo): no ralentiza los casos normales.
    judge_num_predict: int = 2048
    judge_temperature: float = 0.0
    judge_seed: int = 42
    judge_keep_alive: str = "5m"

    # Índice denso: el modelo se elige por CLI.
    dense_index_root: str = "data/indexes/dense"
    default_cpu_threads: int = 8
    max_cpu_threads: int = 16

    @model_validator(mode="after")
    def _validate_loopback_urls(self) -> "Settings":
        """Valida la forma de OLLAMA_BASE_URL y JUDGE_BASE_URL y exige loopback salvo opt-in."""
        _validate_loopback_url("OLLAMA_BASE_URL", self.ollama_base_url, self.ollama_allow_remote)
        _validate_loopback_url("JUDGE_BASE_URL", self.judge_base_url, self.ollama_allow_remote)
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
            "judge_timeout_seconds": self.judge_timeout_seconds,
            "judge_num_ctx": self.judge_num_ctx,
            "judge_num_predict": self.judge_num_predict,
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
