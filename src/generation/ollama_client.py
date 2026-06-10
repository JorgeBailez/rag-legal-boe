"""Cliente HTTP síncrono y aislado para Ollama (suficiente para el CLI; sin streaming).

Usa `httpx` (ya en `pyproject.toml`). No abre red al importar: el `httpx.Client` se crea de forma
perezosa y es **inyectable** para tests con `httpx.MockTransport`. Concreto para Ollama (no una
abstracción genérica de backends), siguiendo el estilo de la Fase 2.

Errores:
- HTTP, red, timeout o envoltorio no-JSON → `OllamaApiError` (no filtra secretos).
- `message.content` que no cumple `RagLlmAnswerV1` → `GenerationContractError`.
"""

from __future__ import annotations

import json
from types import TracebackType

import httpx
from pydantic import ValidationError

from src.contracts.generation_models import OllamaMetricsV1, RagLlmAnswerV1
from src.core.exceptions import GenerationContractError, OllamaApiError

DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT_SECONDS = 900.0


class OllamaClient:
    """Cliente mínimo para `/api/chat`, `/api/version`, `/api/ps` y descarga (`/api/generate`)."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        think: bool = False,
        keep_alive: str | int | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.think = think
        self.keep_alive = keep_alive
        self._client = client
        self._owns_client = client is None

    # -- ciclo de vida -------------------------------------------------------
    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- transporte ----------------------------------------------------------
    def _post(self, path: str, body: dict) -> dict:
        try:
            resp = self.client.post(f"{self.base_url}{path}", json=body)
        except httpx.TimeoutException as exc:
            raise OllamaApiError(f"timeout al llamar a Ollama ({path})") from exc
        except httpx.RequestError as exc:
            raise OllamaApiError(f"error de red al llamar a Ollama ({path})") from exc
        self._raise_for_status(resp, path)
        return self._parse_json(resp, path)

    def _get(self, path: str) -> dict:
        try:
            resp = self.client.get(f"{self.base_url}{path}")
        except httpx.TimeoutException as exc:
            raise OllamaApiError(f"timeout al llamar a Ollama ({path})") from exc
        except httpx.RequestError as exc:
            raise OllamaApiError(f"error de red al llamar a Ollama ({path})") from exc
        self._raise_for_status(resp, path)
        return self._parse_json(resp, path)

    @staticmethod
    def _raise_for_status(resp: httpx.Response, path: str) -> None:
        if resp.status_code < 400:
            return
        detail = ""
        try:
            payload = resp.json()
            if isinstance(payload, dict) and payload.get("error"):
                detail = str(payload["error"])[:200]
        except (json.JSONDecodeError, ValueError):
            detail = (resp.text or "")[:200]
        raise OllamaApiError(f"Ollama respondió HTTP {resp.status_code} en {path}: {detail}")

    @staticmethod
    def _parse_json(resp: httpx.Response, path: str) -> dict:
        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise OllamaApiError(f"respuesta de Ollama no es JSON válido ({path})") from exc
        if not isinstance(data, dict):
            raise OllamaApiError(f"respuesta de Ollama con forma inesperada ({path})")
        return data

    # -- API -----------------------------------------------------------------
    def version(self) -> str:
        """Devuelve la versión de Ollama (health check ligero)."""
        return str(self._get("/api/version").get("version", ""))

    def ps(self) -> dict:
        """Devuelve los modelos cargados en memoria (inspección)."""
        return self._get("/api/ps")

    def _chat_content(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: dict,
        temperature: float,
        seed: int,
        num_predict: int,
        num_ctx: int,
        keep_alive: str | int | None,
    ) -> tuple[str, OllamaMetricsV1]:
        """Llama a `/api/chat` con salida estructurada y devuelve (contenido crudo, métricas)."""
        body: dict = {
            "model": self.model,
            "messages": messages,
            "format": response_format,
            "stream": False,
            "think": self.think,
            "options": {
                "temperature": temperature,
                "seed": seed,
                "num_predict": num_predict,
                "num_ctx": num_ctx,
            },
        }
        effective_keep_alive = keep_alive if keep_alive is not None else self.keep_alive
        if effective_keep_alive is not None:
            body["keep_alive"] = effective_keep_alive

        data = self._post("/api/chat", body)
        if data.get("error"):
            raise OllamaApiError(f"Ollama devolvió error: {str(data['error'])[:200]}")
        message = data.get("message")
        if not isinstance(message, dict) or "content" not in message:
            raise OllamaApiError("respuesta de Ollama sin 'message.content'")
        return message["content"], self._metrics_from(data)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: dict | None = None,
        temperature: float = 0.0,
        seed: int = 42,
        num_predict: int = 256,
        num_ctx: int = 4096,
        keep_alive: str | int | None = None,
    ) -> tuple[RagLlmAnswerV1, OllamaMetricsV1]:
        """Llama a `/api/chat` y valida la salida contra `rag_llm_answer_v1`."""
        fmt = response_format if response_format is not None else RagLlmAnswerV1.model_json_schema()
        content, metrics = self._chat_content(
            messages,
            response_format=fmt,
            temperature=temperature,
            seed=seed,
            num_predict=num_predict,
            num_ctx=num_ctx,
            keep_alive=keep_alive,
        )
        try:
            answer = RagLlmAnswerV1.model_validate_json(content)
        except ValidationError as exc:
            raise GenerationContractError("la salida del LLM no cumple rag_llm_answer_v1") from exc
        return answer, metrics

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: dict,
        temperature: float = 0.0,
        seed: int = 42,
        num_predict: int = 256,
        num_ctx: int = 4096,
        keep_alive: str | int | None = None,
    ) -> tuple[dict, OllamaMetricsV1]:
        """Variante genérica: devuelve el JSON crudo (dict) para esquemas distintos (p. ej. juez).

        No valida contra `rag_llm_answer_v1`: el llamante valida contra su propio contrato. Usa el
        mismo transporte que `chat()` sin duplicar lógica.
        """
        content, metrics = self._chat_content(
            messages,
            response_format=response_format,
            temperature=temperature,
            seed=seed,
            num_predict=num_predict,
            num_ctx=num_ctx,
            keep_alive=keep_alive,
        )
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, ValueError) as exc:
            raise OllamaApiError("la salida del juez no es JSON válido") from exc
        if not isinstance(data, dict):
            raise OllamaApiError("la salida del juez no es un objeto JSON")
        return data, metrics

    def unload(self) -> dict:
        """Descarga el modelo de memoria (`keep_alive=0` vía `/api/generate`)."""
        return self._post(
            "/api/generate",
            {"model": self.model, "prompt": "", "keep_alive": 0, "stream": False},
        )

    @staticmethod
    def _metrics_from(data: dict) -> OllamaMetricsV1:
        def _as_int(key: str) -> int:
            return int(data.get(key) or 0)

        return OllamaMetricsV1(
            total_duration_ns=_as_int("total_duration"),
            load_duration_ns=_as_int("load_duration"),
            prompt_eval_count=_as_int("prompt_eval_count"),
            prompt_eval_duration_ns=_as_int("prompt_eval_duration"),
            eval_count=_as_int("eval_count"),
            eval_duration_ns=_as_int("eval_duration"),
        )
