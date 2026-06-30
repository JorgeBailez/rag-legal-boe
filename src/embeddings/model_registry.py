"""Registro declarativo de modelos de embeddings densos (contrato por modelo).

Cada candidato declara su **contrato**: identidad reproducible (model_id + commit hash),
cómo se formatea un documento y una query, su límite de tokens y su dimensión esperada. El
perfilador, el encoder y el benchmark aplican el formato propio de cada modelo, **no** un formato
común.

Separación conceptual (no mezclar):
- `document_embedding_contract`: lo que define la identidad de los embeddings documentales
  (`document_template`, `pooling`, `normalize_embeddings`, dimensión, límite de tokens).
- `query_profile`: cómo se formatea una query (`default_query_template`,
  `default_query_instruction`); es **configurable** y no altera los embeddings documentales.

Reproducibilidad: `model_revision`/`tokenizer_revision` deben fijar el **commit hash exacto**. Aquí
quedan como `None` (*unresolved*) porque no se pueden resolver de forma fiable sin red; el gate de
generación bloquea hasta fijarlos (o usar `--allow-unpinned-revision`, que acepta `main`
explícitamente). **Nunca** se usa `revision="main"` de forma silenciosa.

Sin dependencias pesadas: este módulo es datos + formato puro (testeable sin red ni `transformers`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

# Placeholders admitidos en las plantillas (se sustituyen por reemplazo literal, no `str.format`,
# para no romper con texto jurídico que contenga llaves).
_DOC_PLACEHOLDER = "{text}"
_QUERY_PLACEHOLDER = "{query}"
_TASK_PLACEHOLDER = "{task}"


@dataclass(frozen=True)
class ModelContract:
    """Contrato declarativo de un modelo de embeddings candidato."""

    alias: str
    model_id: str
    declared_max_tokens: int
    expected_embedding_dimension: int
    # Identidad reproducible (commit hash exacto). None = unresolved (gate bloqueante).
    model_revision: str | None = None
    tokenizer_id: str | None = None  # None ⇒ usa model_id
    tokenizer_revision: str | None = None
    # document_embedding_contract
    document_template: str = _DOC_PLACEHOLDER
    pooling: str = "mean"
    normalize_embeddings: bool = True
    # query_profile (configurable; no afecta a la identidad documental)
    default_query_template: str = _QUERY_PLACEHOLDER
    default_query_instruction: str | None = None
    # código remoto
    trust_remote_code: bool = False
    remote_code_reviewed: bool = False
    notes: str = ""

    @property
    def effective_tokenizer_id(self) -> str:
        """Tokenizer a cargar (por defecto, el del propio modelo)."""
        return self.tokenizer_id or self.model_id

    def format_document(self, text: str) -> str:
        """Formatea el texto a indexar según el contrato del modelo (reemplazo literal)."""
        return self.document_template.replace(_DOC_PLACEHOLDER, text)

    def format_query(self, query: str, task: str | None = None) -> str:
        """Formatea una query según el contrato (usa la instrucción del contrato si no se pasa)."""
        instruction = task if task is not None else (self.default_query_instruction or "")
        out = self.default_query_template.replace(_TASK_PLACEHOLDER, instruction)
        return out.replace(_QUERY_PLACEHOLDER, query)


@dataclass(frozen=True)
class QueryProfile:
    """Identidad reproducible de una forma de codificar queries."""

    profile_id: str
    instruction: str | None
    notes: str = ""
    allowed_aliases: tuple[str, ...] | None = None
    query_template_override: str | None = None


BASELINE_QUERY_PROFILE_ID = "BASELINE"
DEFAULT_QUERY_PROFILE_ID = BASELINE_QUERY_PROFILE_ID
INSTRUCT_QUERY_PROFILE_IDS = ("I0_GENERIC", "I1_LEGAL", "I2_CITIZEN_LEGISLATION")
QWEN_QUERY_PROFILE_IDS = (*INSTRUCT_QUERY_PROFILE_IDS, "I_MINUS_NONE")

QUERY_PROFILES: dict[str, QueryProfile] = {
    BASELINE_QUERY_PROFILE_ID: QueryProfile(
        profile_id=BASELINE_QUERY_PROFILE_ID,
        instruction=None,
        notes="Perfil canonico para modelos cuyo template no usa {task}.",
    ),
    "I0_GENERIC": QueryProfile(
        profile_id="I0_GENERIC",
        instruction=("Given a web search query, retrieve relevant passages that answer the query"),
        notes="Instrucción genérica, no jurídica.",
    ),
    "I1_LEGAL": QueryProfile(
        profile_id="I1_LEGAL",
        instruction=(
            "Given a user question about legislation, retrieve the relevant legal passages "
            "that help answer the question"
        ),
        notes="Perfil jurídico base.",
    ),
    "I2_CITIZEN_LEGISLATION": QueryProfile(
        profile_id="I2_CITIZEN_LEGISLATION",
        instruction=(
            "Given a citizen question about legislation, retrieve the legal passages needed "
            "to answer it accurately"
        ),
        notes="Perfil ciudadano sobre legislación.",
    ),
    "I_MINUS_NONE": QueryProfile(
        profile_id="I_MINUS_NONE",
        instruction=None,
        notes="Consulta cruda sin instrucción ni wrapper; diagnóstico exclusivo para Qwen3.",
        allowed_aliases=("qwen3-0.6b",),
        query_template_override="{query}",
    ),
}


def _canonical_json(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _fingerprint(obj: Any) -> str:
    return hashlib.sha256(_canonical_json(obj)).hexdigest()


def get_query_profile(profile_id: str) -> QueryProfile:
    if profile_id not in QUERY_PROFILES:
        raise KeyError(profile_id)
    return QUERY_PROFILES[profile_id]


def query_template_uses_task(contract: ModelContract) -> bool:
    """True si la query codificada cambia al variar la instruccion `{task}`."""
    return _TASK_PLACEHOLDER in contract.default_query_template


def allowed_query_profile_ids(contract: ModelContract) -> tuple[str, ...]:
    """Perfiles que representan inputs efectivamente distintos para este contrato."""
    if not query_template_uses_task(contract):
        return (BASELINE_QUERY_PROFILE_ID,)
    if contract.alias == "qwen3-0.6b":
        return QWEN_QUERY_PROFILE_IDS
    return INSTRUCT_QUERY_PROFILE_IDS


def default_query_profile_id_for_contract(contract: ModelContract) -> str:
    """Default operativo: baseline sin task; I1 para modelos instruct."""
    return "I1_LEGAL" if query_template_uses_task(contract) else BASELINE_QUERY_PROFILE_ID


def resolve_query_profile(
    contract: ModelContract, profile: QueryProfile | str | None
) -> QueryProfile:
    profile_id = (
        default_query_profile_id_for_contract(contract)
        if profile is None
        else (profile.profile_id if isinstance(profile, QueryProfile) else profile)
    )
    resolved = get_query_profile(profile_id)
    allowed = allowed_query_profile_ids(contract)
    if resolved.profile_id not in allowed:
        if not query_template_uses_task(contract):
            raise ValueError(
                f"{contract.alias}: query_profile_id={resolved.profile_id!r} no es compatible "
                "porque el template de query no usa {task}; usa BASELINE para evitar perfiles "
                "ficticiamente distintos."
            )
        raise ValueError(
            f"{contract.alias}: query_profile_id={resolved.profile_id!r} no es compatible; "
            f"perfiles permitidos: {', '.join(allowed)}"
        )
    if resolved.allowed_aliases is not None and contract.alias not in resolved.allowed_aliases:
        raise ValueError(f"{resolved.profile_id} no es compatible con {contract.alias}")
    return resolved


def effective_query_profile_ids(
    contract: ModelContract, requested_profile_ids: list[str] | None
) -> list[str]:
    """Lista deduplicada de perfiles efectivos para un benchmark de este modelo."""
    profile_ids = requested_profile_ids or list(allowed_query_profile_ids(contract))
    out: list[str] = []
    seen: set[str] = set()
    for profile_id in profile_ids:
        resolved = resolve_query_profile(contract, profile_id)
        if resolved.profile_id not in seen:
            seen.add(resolved.profile_id)
            out.append(resolved.profile_id)
    return out


def format_query_with_profile(
    contract: ModelContract, query: str, profile: QueryProfile | str | None
) -> str:
    profile = resolve_query_profile(contract, profile)
    template = profile.query_template_override or contract.default_query_template
    instruction = profile.instruction or ""
    out = template.replace(_TASK_PLACEHOLDER, instruction)
    return out.replace(_QUERY_PLACEHOLDER, query)


def query_profile_fingerprint(contract: ModelContract, profile: QueryProfile | str | None) -> str:
    profile = resolve_query_profile(contract, profile)
    effective_template = profile.query_template_override or contract.default_query_template
    return _fingerprint(
        {
            "query_profile_id": profile.profile_id,
            "model_id": contract.model_id,
            "model_revision": contract.model_revision,
            "tokenizer_id": contract.effective_tokenizer_id,
            "tokenizer_revision": contract.tokenizer_revision,
            "query_template": effective_template,
            "instruction": profile.instruction,
            "normalize_embeddings": contract.normalize_embeddings,
        }
    )


def query_profile_metadata(contract: ModelContract, profile: QueryProfile | str | None) -> dict:
    profile = resolve_query_profile(contract, profile)
    effective_template = profile.query_template_override or contract.default_query_template
    return {
        "query_profile_id": profile.profile_id,
        "query_profile_fingerprint": query_profile_fingerprint(contract, profile),
        "query_template": effective_template,
        "query_instruction": profile.instruction,
        "normalize_embeddings": contract.normalize_embeddings,
    }


def assert_bundle_compatible(contract: ModelContract, manifest: dict) -> None:
    """Valida que el registry actual puede codificar queries contra el bundle."""
    dec = manifest["document_embedding_contract"]
    checks = {
        "model_id": (contract.model_id, dec["model_id"]),
        "model_revision": (contract.model_revision, dec["model_revision"]),
        "tokenizer_id": (contract.effective_tokenizer_id, dec["tokenizer_id"]),
        "tokenizer_revision": (contract.tokenizer_revision, dec["tokenizer_revision"]),
        "embedding_dimension": (contract.expected_embedding_dimension, dec["embedding_dimension"]),
        "normalize_embeddings": (contract.normalize_embeddings, dec["normalize_embeddings"]),
    }
    mismatches = [name for name, (current, bundled) in checks.items() if current != bundled]
    if mismatches:
        details = ", ".join(
            f"{name}: registry={checks[name][0]!r} bundle={checks[name][1]!r}"
            for name in mismatches
        )
        raise ValueError(f"registry incompatible con bundle ({details})")


# Shortlist. `*_revision=None` (unresolved) hasta fijar el commit hash en el servidor (con red).
CANDIDATES: dict[str, ModelContract] = {
    "e5-base": ModelContract(
        alias="e5-base",
        model_id="intfloat/multilingual-e5-base",
        model_revision="d128750597153bb5987e10b1c3493a34e5a4502a",
        tokenizer_revision="d128750597153bb5987e10b1c3493a34e5a4502a",
        declared_max_tokens=512,
        expected_embedding_dimension=768,
        document_template="passage: {text}",
        pooling="mean",
        default_query_template="query: {query}",
        notes="Familia e5: exige prefijos passage:/query:. Límite 512 tokens (riesgo de overflow).",
    ),
    "e5-large": ModelContract(
        alias="e5-large",
        model_id="intfloat/multilingual-e5-large",
        model_revision="3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3",
        tokenizer_revision="3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3",
        declared_max_tokens=512,
        expected_embedding_dimension=1024,
        document_template="passage: {text}",
        pooling="mean",
        default_query_template="query: {query}",
        notes="Baseline histórico del repo. Límite 512 tokens (riesgo de overflow).",
    ),
    "e5-large-instruct": ModelContract(
        alias="e5-large-instruct",
        model_id="intfloat/multilingual-e5-large-instruct",
        model_revision="274baa43b0e13e37fafa6428dbc7938e62e5c439",
        tokenizer_revision="274baa43b0e13e37fafa6428dbc7938e62e5c439",
        declared_max_tokens=512,
        expected_embedding_dimension=1024,
        document_template="{text}",
        pooling="mean",
        default_query_template="Instruct: {task}\nQuery: {query}",
        default_query_instruction=(
            "Given a legal question from a citizen, retrieve the relevant consolidated BOE passages"
        ),
        notes="Variante instruct: documentos sin prefijo; query con instrucción configurable.",
    ),
    "bge-m3": ModelContract(
        alias="bge-m3",
        model_id="BAAI/bge-m3",
        model_revision="5617a9f61b028005a4858fdac845db406aefb181",
        tokenizer_revision="5617a9f61b028005a4858fdac845db406aefb181",
        declared_max_tokens=8192,
        expected_embedding_dimension=1024,
        document_template="{text}",
        pooling="cls",
        default_query_template="{query}",
        notes="Contexto largo (8192): apenas habrá overflow. Denso (sparse fuera de alcance).",
    ),
    "qwen3-0.6b": ModelContract(
        alias="qwen3-0.6b",
        model_id="Qwen/Qwen3-Embedding-0.6B",
        model_revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b",
        tokenizer_revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b",
        declared_max_tokens=32768,
        expected_embedding_dimension=1024,
        document_template="{text}",
        pooling="last_token",
        default_query_template="Instruct: {task}\nQuery:{query}",
        default_query_instruction=(
            "Given a legal question from a citizen, retrieve the relevant consolidated BOE passages"
        ),
        notes="Contexto 32k: no habrá overflow. Query con instrucción; documentos sin prefijo.",
    ),
    "gte-multilingual-base": ModelContract(
        alias="gte-multilingual-base",
        model_id="Alibaba-NLP/gte-multilingual-base",
        model_revision="9bbca17d9273fd0d03d5725c7a4b0f6b45142062",
        tokenizer_revision="9bbca17d9273fd0d03d5725c7a4b0f6b45142062",
        declared_max_tokens=8192,
        expected_embedding_dimension=768,
        document_template="{text}",
        pooling="cls",
        default_query_template="{query}",
        trust_remote_code=True,
        remote_code_reviewed=True,
        notes=(
            "Contexto largo (8192, RoPE+NTK). trust_remote_code: el modeling vive en el repo "
            "Alibaba-NLP/new-impl (sha 40ced75c, jul-2024), revisado 2026-06-16: solo define "
            "nn.Module de torch; sin red, IO de ficheros, subprocess/exec/eval, env ni secretos. "
            "Pesos+tokenizer fijados en 9bbca17d. Caveat reproducibilidad: el codigo remoto "
            "resuelve new-impl@main salvo code_revision, que el encoder actual no fija. "
            "DEFERIDO 2026-06-18: incompatible con el transformers del entorno "
            "(IndexError RoPE en new-impl); reactivar con transformers ~4.39 en venv "
            "aparte. Ver docs/known_issues.md."
        ),
    ),
}

# Índice auxiliar model_id → alias (para resolver por cualquiera de los dos).
_ALIAS_BY_MODEL_ID: dict[str, str] = {c.model_id: a for a, c in CANDIDATES.items()}


def resolve_alias(name: str) -> str:
    """Devuelve el alias canónico para un alias o un model_id. KeyError si no existe."""
    if name in CANDIDATES:
        return name
    if name in _ALIAS_BY_MODEL_ID:
        return _ALIAS_BY_MODEL_ID[name]
    raise KeyError(name)


def get_contract(name: str) -> ModelContract:
    """Devuelve el contrato de un candidato por alias o model_id. KeyError si no existe."""
    return CANDIDATES[resolve_alias(name)]


def list_models() -> list[ModelContract]:
    """Lista de contratos en orden de registro (para --list-models)."""
    return list(CANDIDATES.values())


def all_aliases() -> list[str]:
    """Lista de aliases cortos en orden de registro."""
    return list(CANDIDATES)


def all_model_ids() -> list[str]:
    """Lista de model_id en orden de registro."""
    return [c.model_id for c in CANDIDATES.values()]
