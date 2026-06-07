"""Tests del cliente Ollama con httpx.MockTransport (sin red, sin servidor)."""

import json

import httpx
import pytest

from src.core.exceptions import GenerationContractError, OllamaApiError
from src.generation.ollama_client import OllamaClient

_VALID_CONTENT = json.dumps(
    {
        "answered": True,
        "answer": "El plazo es de un mes.",
        "citation_ids": ["E1"],
        "abstention_reason": "",
    }
)


def _client(handler) -> OllamaClient:
    transport = httpx.MockTransport(handler)
    return OllamaClient(model="qwen3:4b-instruct", client=httpx.Client(transport=transport))


def test_chat_payload_and_parsing() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": _VALID_CONTENT},
                "total_duration": 3_000_000_000,
                "load_duration": 1_000_000_000,
                "prompt_eval_count": 120,
                "prompt_eval_duration": 500_000_000,
                "eval_count": 22,
                "eval_duration": 10_000_000_000,
            },
        )

    answer, metrics = _client(handler).chat(
        [{"role": "user", "content": "¿plazo?"}], temperature=0, seed=42
    )
    # Payload correcto
    assert captured["path"] == "/api/chat"
    body = captured["body"]
    assert body["model"] == "qwen3:4b-instruct"
    assert body["stream"] is False and body["think"] is False
    assert "properties" in body["format"] and "citation_ids" in body["format"]["properties"]
    assert body["options"]["temperature"] == 0 and body["options"]["seed"] == 42
    # Parseo correcto
    assert answer.answered and answer.citation_ids == ["E1"]
    assert metrics.eval_count == 22
    assert metrics.tokens_per_second == pytest.approx(2.2)


def test_chat_keep_alive_included_when_set() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"message": {"role": "assistant", "content": _VALID_CONTENT}}
        )

    transport = httpx.MockTransport(handler)
    client = OllamaClient(model="m", keep_alive="5m", client=httpx.Client(transport=transport))
    client.chat([{"role": "user", "content": "x"}])
    assert captured["body"]["keep_alive"] == "5m"


def test_chat_http_error_raises_ollama_api_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "internal boom"})

    with pytest.raises(OllamaApiError, match="HTTP 500"):
        _client(handler).chat([{"role": "user", "content": "x"}])


def test_chat_timeout_raises_ollama_api_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout", request=request)

    with pytest.raises(OllamaApiError, match="timeout"):
        _client(handler).chat([{"role": "user", "content": "x"}])


def test_chat_network_error_raises_ollama_api_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    with pytest.raises(OllamaApiError, match="red"):
        _client(handler).chat([{"role": "user", "content": "x"}])


def test_chat_invalid_envelope_json_raises_ollama_api_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="no soy json")

    with pytest.raises(OllamaApiError, match="JSON"):
        _client(handler).chat([{"role": "user", "content": "x"}])


def test_chat_missing_message_raises_ollama_api_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"done": True})

    with pytest.raises(OllamaApiError, match="message.content"):
        _client(handler).chat([{"role": "user", "content": "x"}])


def test_chat_content_not_valid_contract_raises_contract_error() -> None:
    # JSON válido pero incumple invariantes (answered=true sin citas) → contrato.
    bad = json.dumps(
        {"answered": True, "answer": "algo", "citation_ids": [], "abstention_reason": ""}
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"role": "assistant", "content": bad}})

    with pytest.raises(GenerationContractError):
        _client(handler).chat([{"role": "user", "content": "x"}])


def test_chat_content_not_json_raises_contract_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"message": {"role": "assistant", "content": "esto no es json {"}}
        )

    with pytest.raises(GenerationContractError):
        _client(handler).chat([{"role": "user", "content": "x"}])


def test_unload_sends_keep_alive_zero() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"done": True})

    _client(handler).unload()
    assert captured["path"] == "/api/generate"
    assert captured["body"]["keep_alive"] == 0 and captured["body"]["model"] == "qwen3:4b-instruct"


def test_version_and_ps() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.30.6"})
        return httpx.Response(200, json={"models": []})

    client = _client(handler)
    assert client.version() == "0.30.6"
    assert client.ps() == {"models": []}


def test_no_network_on_import() -> None:
    # Construir el cliente no debe abrir red: sin client inyectado, el httpx.Client es perezoso.
    client = OllamaClient(model="m")
    assert client._client is None
