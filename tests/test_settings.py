"""Tests de validación de `Settings` (parámetros de generación + salvaguarda de URL remota)."""

import pytest

from src.config.settings import Settings
from src.core.exceptions import ConfigurationError


def test_defaults_are_valid() -> None:
    s = Settings()
    assert s.ollama_base_url.startswith("http://127.0.0.1")
    assert s.generation_context_strategy == "P_EXPAND_BOUNDED"
    assert s.generation_top_k > 0


@pytest.mark.parametrize(
    "field",
    [
        "generation_top_k",
        "generation_max_evidences",
        "generation_context_budget_chars",
        "generation_max_total_context_chars",
        "ollama_timeout_seconds",
        "ollama_num_ctx",
        "ollama_num_predict",
    ],
)
def test_non_positive_numeric_settings_rejected(field: str) -> None:
    with pytest.raises(ConfigurationError):
        Settings(**{field: 0})


def test_unknown_context_strategy_rejected() -> None:
    with pytest.raises(ConfigurationError):
        Settings(generation_context_strategy="NOPE")


def test_remote_ollama_url_rejected_without_optin() -> None:
    with pytest.raises(ConfigurationError):
        Settings(ollama_base_url="http://10.0.0.5:11434")


def test_remote_ollama_url_allowed_with_optin() -> None:
    s = Settings(ollama_base_url="http://10.0.0.5:11434", ollama_allow_remote=True)
    assert s.ollama_allow_remote is True


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:11434",
        "http://localhost:11434",
        "http://[::1]:11434",
        "https://localhost:11434/",
    ],
)
def test_valid_loopback_urls_accepted(url: str) -> None:
    assert Settings(ollama_base_url=url).ollama_base_url == url


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:abc",
        "http://127.0.0.1:99999",
        "http://127.0.0.1:0",
        "http://127.0.0.1:-1",
    ],
)
def test_invalid_ollama_ports_rejected(url: str) -> None:
    with pytest.raises(ConfigurationError):
        Settings(ollama_base_url=url)


@pytest.mark.parametrize("timeout", [0, -1])
def test_non_positive_ollama_timeout_rejected(timeout: float) -> None:
    with pytest.raises(ConfigurationError):
        Settings(ollama_timeout_seconds=timeout)


@pytest.mark.parametrize(
    "url",
    [
        "",
        "foo",
        "file:///tmp/x",
        "http://",
        "http://127.0.0.1:11434/api",
        "http://127.0.0.1:11434?x=1",
        "http://127.0.0.1:11434#frag",
        "http://usuario:clave@127.0.0.1:11434",
    ],
)
def test_malformed_ollama_urls_rejected(url: str) -> None:
    with pytest.raises(ConfigurationError):
        Settings(ollama_base_url=url)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:11434/api",  # path
        "http://127.0.0.1:11434?x=1",  # query
        "http://usuario:clave@127.0.0.1:11434",  # credenciales
    ],
)
def test_malformed_urls_rejected_even_with_optin(url: str) -> None:
    # Aunque se permita host remoto, una URL malformada se rechaza igualmente.
    with pytest.raises(ConfigurationError):
        Settings(ollama_base_url=url, ollama_allow_remote=True)
