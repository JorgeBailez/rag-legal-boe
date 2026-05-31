"""Registro declarativo de modelos de embeddings candidatos (contratos por modelo).

Cada candidato declara su **contrato**: cómo se formatea un documento y una query, su límite de
tokens declarado y su dimensión esperada. El perfilador y (más adelante) el benchmark aplican el
formato propio de cada modelo, **no** un formato común. La instrucción de query queda registrada
y es **configurable**; no se fija aquí la definitiva.

Sin dependencias pesadas: este módulo es datos + formato puro (testeable sin red ni `transformers`).
"""

from __future__ import annotations

from dataclasses import dataclass

# Placeholders admitidos en las plantillas (se sustituyen por reemplazo literal, no `str.format`,
# para no romper con texto jurídico que contenga llaves).
_DOC_PLACEHOLDER = "{text}"
_QUERY_PLACEHOLDER = "{query}"
_TASK_PLACEHOLDER = "{task}"


@dataclass(frozen=True)
class ModelContract:
    """Contrato declarativo de un modelo de embeddings candidato."""

    model_id: str
    declared_max_tokens: int
    expected_embedding_dimension: int
    document_template: str = _DOC_PLACEHOLDER
    query_template: str = _QUERY_PLACEHOLDER
    query_instruction: str | None = None
    revision: str | None = None  # fijar (pin) antes del benchmark reproducible
    notes: str = ""

    def format_document(self, retrieval_text: str) -> str:
        """Formatea el texto a indexar según el contrato del modelo (reemplazo literal)."""
        return self.document_template.replace(_DOC_PLACEHOLDER, retrieval_text)

    def format_query(self, query: str, task: str | None = None) -> str:
        """Formatea una query según el contrato (usa la instrucción del contrato si no se pasa)."""
        instruction = task if task is not None else (self.query_instruction or "")
        out = self.query_template.replace(_TASK_PLACEHOLDER, instruction)
        return out.replace(_QUERY_PLACEHOLDER, query)


# Shortlist inicial. `revision=None` se fijará antes del benchmark reproducible (Fase 2.5).
CANDIDATES: dict[str, ModelContract] = {
    "intfloat/multilingual-e5-base": ModelContract(
        model_id="intfloat/multilingual-e5-base",
        declared_max_tokens=512,
        expected_embedding_dimension=768,
        document_template="passage: {text}",
        query_template="query: {query}",
        notes="Familia e5: exige prefijos passage:/query:. Límite 512 tokens (riesgo de truncado).",
    ),
    "intfloat/multilingual-e5-large": ModelContract(
        model_id="intfloat/multilingual-e5-large",
        declared_max_tokens=512,
        expected_embedding_dimension=1024,
        document_template="passage: {text}",
        query_template="query: {query}",
        notes="Default actual del repo → baseline histórico. Límite 512 tokens.",
    ),
    "intfloat/multilingual-e5-large-instruct": ModelContract(
        model_id="intfloat/multilingual-e5-large-instruct",
        declared_max_tokens=512,
        expected_embedding_dimension=1024,
        document_template="{text}",
        query_template="Instruct: {task}\nQuery: {query}",
        query_instruction=(
            "Given a legal question from a citizen, retrieve the relevant consolidated BOE passages"
        ),
        notes="Variante instruct: documentos sin prefijo; query con instrucción configurable.",
    ),
    "BAAI/bge-m3": ModelContract(
        model_id="BAAI/bge-m3",
        declared_max_tokens=8192,
        expected_embedding_dimension=1024,
        document_template="{text}",
        query_template="{query}",
        notes="Contexto largo (8192): apenas truncará. Soporta denso + sparse.",
    ),
    "Qwen/Qwen3-Embedding-0.6B": ModelContract(
        model_id="Qwen/Qwen3-Embedding-0.6B",
        declared_max_tokens=32768,
        expected_embedding_dimension=1024,
        document_template="{text}",
        query_template="{query}",
        query_instruction=None,
        notes="Contexto 32k: no truncará. Formatter/instrucción parametrizables por contrato.",
    ),
}


def get_contract(model_id: str) -> ModelContract:
    """Devuelve el contrato de un modelo candidato o lanza KeyError si no existe."""
    return CANDIDATES[model_id]


def all_model_ids() -> list[str]:
    """Lista de model_id de la shortlist, en orden de registro."""
    return list(CANDIDATES)
