"""Contratos Pydantic v2 de la generación fundamentada de Fase 3 (fuente única de verdad).

Dos familias, estrictas (`extra="forbid"`):

- `rag_llm_answer_v1`: contrato MÍNIMO que devuelve el LLM (Ollama). Solo IDs compactos de
  evidencia (`E1`, `E2`, ...); el LLM nunca genera etiquetas BOE ni URL. Sus invariantes
  (answered ⇒ respuesta+citas; ¬answered ⇒ motivo de abstención) se validan aquí, de modo que
  una salida incoherente del modelo se rechaza en la frontera.
- `rag_answer_v1`: contrato FINAL de la aplicación (salida del CLI). Enriquece las citas con datos
  autoritativos del corpus (etiqueta + URL), añade el aviso jurídico estático de forma
  determinista y adjunta trazabilidad de retrieval y métricas de generación.

Estos modelos generan los JSON Schema de `schemas/` vía `export_schemas.py` (test anti-drift).
No mezclan los contratos jurídicos (Fase 1) ni los densos (Fase 2); solo se registran como root
models exportables en `GENERATION_ROOT_MODELS`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    """Base estricta: prohíbe campos no declarados (detecta contratos ajenos/derivas)."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# rag_llm_answer_v1 — contrato reducido devuelto por el LLM
# --------------------------------------------------------------------------- #


class RagLlmAnswerV1(_Strict):
    """Salida estructurada del LLM: solo IDs compactos de evidencia, sin etiquetas ni URL.

    Todos los campos son obligatorios para que el JSON Schema enviado a Ollama (`format`) los
    exija explícitamente y el modelo no pueda omitirlos. Las invariantes se validan tras construir
    el objeto: si no se cumplen, es una violación de contrato (la captura el orquestador).
    """

    answered: bool
    answer: str
    citation_ids: list[str]
    abstention_reason: str

    @model_validator(mode="after")
    def _check_invariants(self) -> RagLlmAnswerV1:
        if self.answered:
            if not self.answer.strip():
                raise ValueError("answered=true requiere 'answer' no vacío")
            if not self.citation_ids:
                raise ValueError("answered=true requiere 'citation_ids' no vacío")
            if self.abstention_reason.strip():
                raise ValueError("answered=true exige 'abstention_reason' vacío")
        else:
            if self.answer.strip():
                raise ValueError("answered=false exige 'answer' vacío")
            if self.citation_ids:
                raise ValueError("answered=false exige 'citation_ids' vacío")
            if not self.abstention_reason.strip():
                raise ValueError("answered=false requiere 'abstention_reason' no vacío")
        return self


# --------------------------------------------------------------------------- #
# Citas enriquecidas + métricas + trazabilidad (subcomponentes del contrato final)
# --------------------------------------------------------------------------- #


class AnswerCitationV1(_Strict):
    """Cita autoritativa resuelta por la aplicación desde el corpus (nunca desde texto libre)."""

    evidence_id: str = Field(..., examples=["E1"])
    parent_id: str
    document_id: str
    block_id: str
    label: str = Field(..., examples=["Ley 39/2015, artículo 122"])
    url: str | None = Field(
        None, examples=["https://www.boe.es/buscar/act.php?id=BOE-A-2015-10565#a122"]
    )


class OllamaMetricsV1(_Strict):
    """Métricas crudas de una llamada a Ollama (nanosegundos y conteos de tokens).

    Las propiedades calculadas (segundos, tokens/s) NO se persisten: son ayudas de lectura.
    """

    total_duration_ns: int = 0
    load_duration_ns: int = 0
    prompt_eval_count: int = 0
    prompt_eval_duration_ns: int = 0
    eval_count: int = 0
    eval_duration_ns: int = 0

    @property
    def total_duration_s(self) -> float:
        return self.total_duration_ns / 1e9

    @property
    def load_duration_s(self) -> float:
        return self.load_duration_ns / 1e9

    @property
    def eval_duration_s(self) -> float:
        return self.eval_duration_ns / 1e9

    @property
    def tokens_per_second(self) -> float:
        return self.eval_count / self.eval_duration_s if self.eval_duration_ns > 0 else 0.0


class RetrievalHitTraceV1(_Strict):
    """Traza compacta de un hit (sin texto jurídico): score + relaciones + flag de selección."""

    rank: int
    score: float
    document_id: str
    block_id: str
    parent_id: str
    selected: bool = False
    evidence_id: str | None = None


class EvidenceOmissionTraceV1(_Strict):
    """Diagnóstico compacto de un hit candidato descartado al construir evidencias (sin texto)."""

    parent_id: str
    retrieval_rank: int
    reason: str
    char_count: int | None = None


class RetrievalTraceV1(_Strict):
    """Traza del retrieval: bundle/modelo/perfil + recuento, hits y diagnóstico de omisiones."""

    bundle_id: str
    model_alias: str
    query_profile_id: str
    top_k: int
    returned_hits: int
    selected_evidences: int
    duplicate_parents_removed: int = 0
    total_context_chars: int = 0
    hits: list[RetrievalHitTraceV1] = Field(default_factory=list)
    omitted_evidences: list[EvidenceOmissionTraceV1] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# rag_answer_v1 — contrato final de la aplicación (salida del CLI)
# --------------------------------------------------------------------------- #


class RagAnswerV1(_Strict):
    """Respuesta final auditable: respuesta/abstención + citas autoritativas + aviso + trazas."""

    schema_version: Literal["rag_answer_v1"] = "rag_answer_v1"
    answered: bool
    answer: str
    citations: list[AnswerCitationV1] = Field(default_factory=list)
    abstention_reason: str = ""
    disclaimer: str
    retrieval_trace: RetrievalTraceV1
    generation_metrics: OllamaMetricsV1 | None = None

    @model_validator(mode="after")
    def _check_invariants(self) -> RagAnswerV1:
        if self.answered:
            if not self.answer.strip():
                raise ValueError("answered=true requiere 'answer' no vacío")
            if not self.citations:
                raise ValueError("answered=true requiere 'citations' no vacío")
            if self.abstention_reason.strip():
                raise ValueError("answered=true exige 'abstention_reason' vacío")
        else:
            if self.answer.strip():
                raise ValueError("answered=false exige 'answer' vacío")
            if self.citations:
                raise ValueError("answered=false exige 'citations' vacío")
            if not self.abstention_reason.strip():
                raise ValueError("answered=false requiere 'abstention_reason' no vacío")
        return self


# Registro de contratos raíz de generación exportables (nombre de schema → modelo).
GENERATION_ROOT_MODELS: dict[str, type[BaseModel]] = {
    "rag_llm_answer_v1": RagLlmAnswerV1,
    "rag_answer_v1": RagAnswerV1,
}
