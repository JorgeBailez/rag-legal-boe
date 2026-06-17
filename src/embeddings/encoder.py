"""Encoder denso sobre Sentence Transformers (CPU, float32, L2-normalizado, reproducible).

Un único `DenseEncoder`. Los documentos llegan **ya formateados** por `input_preparation`
(`format_document` aplicado antes de tokenizar, requisito para la prohibición de truncado), por lo
que `encode_documents` los codifica **verbatim**. Las queries son texto crudo del usuario, así que
`encode_queries` aplica `contract.format_query` antes de codificar. Se usa `model.encode(...)` con
los prefijos ya incrustados: respeta el contrato del modelo sin depender de registros de *prompts*
específicos que no todos los candidatos definen.

Seguridad: `HF_TOKEN` se lee solo del entorno y nunca se imprime, persiste ni se incluye en
excepciones. Reproducibilidad: si el commit hash no está fijado, se bloquea salvo
`allow_unpinned_revision` (acepta `main` de forma explícita).
"""

from __future__ import annotations

import os

from src.embeddings.model_registry import (
    ModelContract,
    default_query_profile_id_for_contract,
    format_query_with_profile,
)
from src.embeddings.tokenizer_profiler import resolve_effective_max


class RevisionUnpinnedError(RuntimeError):
    """El modelo no tiene commit hash fijado y no se autorizó `--allow-unpinned-revision`."""


class EncoderLoadError(RuntimeError):
    """No se pudo cargar el modelo (posible autenticación). No incluye el valor de HF_TOKEN."""


class RemoteCodeNotReviewedError(RuntimeError):
    """El contrato requiere código remoto pero no consta revisión local."""


def read_hf_token() -> str | None:
    """Devuelve HF_TOKEN del entorno (o None). No lo imprime ni lo persiste."""
    token = os.environ.get("HF_TOKEN")
    return token or None


def _ensure_remote_code_reviewed(contract: ModelContract) -> None:
    if contract.trust_remote_code and not contract.remote_code_reviewed:
        raise RemoteCodeNotReviewedError(
            f"{contract.alias}: trust_remote_code=True requiere revisar el código remoto, fijar "
            "el hash exacto y marcar remote_code_reviewed=True antes de ejecutar."
        )


def load_tokenizer(contract: ModelContract, *, allow_unpinned_revision: bool = False):
    """Carga el tokenizer real del contrato (import perezoso de transformers; puede requerir red).

    Usa `trust_remote_code` y el commit hash por modelo; nunca `main` silencioso. No imprime
    HF_TOKEN. Reutilizado por el perfilador, el preflight y la generación.
    """
    _ensure_remote_code_reviewed(contract)
    revision = contract.tokenizer_revision or contract.model_revision
    if revision is None and not allow_unpinned_revision:
        raise RevisionUnpinnedError(
            f"{contract.alias}: commit hash del tokenizer sin fijar. Fija las revisiones en "
            "model_registry.py o repite con --allow-unpinned-revision (acepta 'main')."
        )
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise SystemExit("Falta 'transformers'. Instala con: uv add transformers") from exc
    kwargs: dict = {"revision": revision, "trust_remote_code": contract.trust_remote_code}
    token = read_hf_token()
    if token:
        kwargs["token"] = token
    try:
        return AutoTokenizer.from_pretrained(contract.effective_tokenizer_id, **kwargs)
    except Exception:  # noqa: BLE001 - error accionable sin filtrar el token
        raise EncoderLoadError(
            f"No se pudo descargar/cargar el tokenizer {contract.effective_tokenizer_id!r}. Si el "
            "repositorio requiere autenticación, exporta HF_TOKEN y repite. No guardes el token en "
            "el repositorio."
        ) from None


def set_cpu_threads(threads: int) -> None:
    """Fija el número de hilos de torch (no usa todos los lógicos por defecto)."""
    try:
        import torch
    except ImportError:  # pragma: no cover - depende del entorno
        return
    torch.set_num_threads(int(threads))


def align_max_seq_length(model, contract: ModelContract) -> int:
    """Iguala el límite de truncado del SentenceTransformer al `effective_max` del contrato.

    POR QUÉ: `input_preparation` garantiza que ningún input supera el límite efectivo (parte en
    ventanas token-aware antes de codificar), pero `model.encode` aplica su PROPIO `max_seq_length`
    —fijado en el `sentence_bert_config.json` del modelo, a veces muy por debajo de su capacidad
    real (p. ej. 512 en un modelo de contexto 8192)— y truncaría **en silencio** cualquier input por
    encima de ese valor. Igualar ambos hace que la invariante "sin truncado silencioso" deje de
    depender del empaquetado del modelo, sin tocar la lógica de preparación. `effective_max` se
    deriva igual que en `input_preparation` (`resolve_effective_max`), así que nunca supera la
    capacidad real del modelo. Devuelve el límite aplicado, o -1 si el objeto no expone
    `max_seq_length` (p. ej. un modelo inyectado en tests).
    """
    tokenizer = getattr(model, "tokenizer", None)
    tml = getattr(tokenizer, "model_max_length", None)
    effective_max, _ = resolve_effective_max(contract.declared_max_tokens, tml)
    if not hasattr(model, "max_seq_length"):
        return -1
    model.max_seq_length = effective_max
    return effective_max


class DenseEncoder:
    """Codificador denso CPU para un contrato de modelo concreto."""

    backend = "sentence-transformers"

    def __init__(
        self,
        contract: ModelContract,
        *,
        device: str = "cpu",
        batch_size: int = 32,
        allow_unpinned_revision: bool = False,
        model=None,
    ) -> None:
        self.contract = contract
        self.device = device
        self.batch_size = batch_size
        self.dimension = contract.expected_embedding_dimension
        self._model = model if model is not None else self._load_model(allow_unpinned_revision)

    def _load_model(self, allow_unpinned_revision: bool):
        _ensure_remote_code_reviewed(self.contract)
        revision = self.contract.model_revision
        if revision is None and not allow_unpinned_revision:
            raise RevisionUnpinnedError(
                f"{self.contract.alias}: commit hash sin fijar. Fija model_revision/"
                "tokenizer_revision en model_registry.py o repite con --allow-unpinned-revision "
                "(acepta 'main'; reduce reproducibilidad)."
            )
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depende del entorno
            raise SystemExit(
                "Falta 'sentence-transformers'. Instala con: uv add sentence-transformers"
            ) from exc

        kwargs: dict = {
            "device": self.device,
            "revision": revision,
            "trust_remote_code": self.contract.trust_remote_code,
        }
        token = read_hf_token()
        if token:
            kwargs["token"] = token
        try:
            model = SentenceTransformer(self.contract.model_id, **kwargs)
        except Exception:  # noqa: BLE001 - se re-lanza un error accionable sin filtrar el token
            raise EncoderLoadError(
                f"No se pudo descargar/cargar el modelo {self.contract.model_id!r}. Si el "
                "repositorio requiere autenticación, exporta HF_TOKEN en la sesión actual y repite "
                "el comando. No guardes el token dentro del repositorio."
            ) from None
        # Cierra el hueco de truncado en la frontera de la librería (ver align_max_seq_length).
        align_max_seq_length(model, self.contract)
        return model

    def _validate_dim(self, emb) -> None:
        if emb.shape[1] != self.dimension:
            raise ValueError(
                f"dimensión inesperada para {self.contract.alias}: {emb.shape[1]} != "
                f"{self.dimension} (expected_embedding_dimension del contrato)"
            )

    def encode_documents(
        self, texts: list[str], *, show_progress: bool = True, batch_size: int | None = None
    ):
        """Codifica documentos YA formateados (verbatim) → np.float32 [n, dim] L2-normalizado."""
        import numpy as np

        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        emb = self._model.encode(
            texts,
            batch_size=batch_size or self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.contract.normalize_embeddings,
            convert_to_numpy=True,
        )
        emb = np.asarray(emb, dtype=np.float32)
        self._validate_dim(emb)
        return emb

    def encode_queries(
        self,
        queries: list[str],
        *,
        query_profile_id: str | None = None,
        show_progress: bool = False,
        batch_size: int | None = None,
    ):
        """Aplica el query profile del contrato a cada query y codifica → np.float32 [n, dim]."""
        import numpy as np

        if not queries:
            return np.zeros((0, self.dimension), dtype=np.float32)
        profile_id = query_profile_id or default_query_profile_id_for_contract(self.contract)
        formatted = [format_query_with_profile(self.contract, q, profile_id) for q in queries]
        emb = self._model.encode(
            formatted,
            batch_size=batch_size or self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.contract.normalize_embeddings,
            convert_to_numpy=True,
        )
        emb = np.asarray(emb, dtype=np.float32)
        self._validate_dim(emb)
        return emb
