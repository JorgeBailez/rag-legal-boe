"""Orquestador de la respuesta fundamentada (Fase 3): retrieve → evidencias → LLM → contrato final.

Coordina, sin acoplarse a los detalles de cada capa:

    pregunta
      → DenseRetriever            (retrieval denso exacto, dense-only)
      → build_evidences           (dedup + contexto acotado + IDs E1, E2, ...)
      → build_messages            (prompt restrictivo + schema)
      → LlmClient.chat            (Ollama; salida estructurada validada)
      → validación de IDs citados (solo IDs entregados)
      → citas autoritativas       (etiqueta + URL del corpus, nunca del LLM)
      → RagAnswerV1

Garantías (fail closed):
- sin hits o sin evidencias utilizables → abstención determinista SIN llamar al LLM;
- el LLM cita un ID no entregado → `GenerationContractError`;
- el aviso jurídico es estático y aparece SIEMPRE;
- las etiquetas y URLs finales provienen del corpus, no del texto generado.

Dense-only: no introduce BM25, híbrido ni reranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.contracts.generation_models import (
    AnswerCitationV1,
    EvidenceOmissionTraceV1,
    OllamaMetricsV1,
    RagAnswerV1,
    RagLlmAnswerV1,
    RetrievalHitTraceV1,
    RetrievalTraceV1,
)
from src.core.exceptions import ConfigurationError, GenerationContractError
from src.generation.baselines import ClosedBookResult
from src.generation.baselines import generate_closed_book as _generate_closed_book
from src.generation.prompt import build_messages
from src.retrieval.context_assembler import P_EXPAND_BOUNDED, STRATEGIES
from src.retrieval.dense_retriever import DenseHit, DenseRetriever
from src.retrieval.evidence_builder import (
    DEFAULT_CONTEXT_BUDGET_CHARS,
    DEFAULT_MAX_EVIDENCES,
    DEFAULT_MAX_TOTAL_CONTEXT_CHARS,
    EvidenceSelection,
    GenerationEvidence,
    build_evidences,
)

# Aviso jurídico único, constante y determinista (lo añade SIEMPRE la aplicación, no el LLM).
DISCLAIMER = (
    "Aviso: respuesta de carácter informativo. Los textos consolidados del BOE no tienen valor "
    "jurídico oficial. Remítase a la publicación oficial en el BOE."
)

DEFAULT_QUERY_PROFILE_ID = "I1_LEGAL"


class LlmClient(Protocol):
    """Interfaz mínima del cliente de generación (la cumple `OllamaClient`; inyectable en tests)."""

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: dict | None = ...,
        temperature: float = ...,
        seed: int = ...,
        num_predict: int = ...,
        num_ctx: int = ...,
        keep_alive: str | int | None = ...,
    ) -> tuple[RagLlmAnswerV1, OllamaMetricsV1]: ...

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: dict,
        temperature: float = ...,
        seed: int = ...,
        num_predict: int = ...,
        num_ctx: int = ...,
        keep_alive: str | int | None = ...,
    ) -> tuple[dict, OllamaMetricsV1]: ...


@dataclass
class GenerationConfig:
    """Parámetros de la generación fundamentada."""

    query_profile_id: str = DEFAULT_QUERY_PROFILE_ID
    top_k: int = 3
    max_evidences: int = DEFAULT_MAX_EVIDENCES
    context_strategy: str = P_EXPAND_BOUNDED
    context_budget_chars: int = DEFAULT_CONTEXT_BUDGET_CHARS
    max_total_context_chars: int = DEFAULT_MAX_TOTAL_CONTEXT_CHARS
    temperature: float = 0.0
    seed: int = 42
    num_predict: int = 1536
    num_ctx: int = 8192
    keep_alive: str | int | None = None

    def __post_init__(self) -> None:
        """Rechaza parámetros inválidos (positivos obligatorios + estrategia conocida)."""
        positive = {
            "top_k": self.top_k,
            "max_evidences": self.max_evidences,
            "context_budget_chars": self.context_budget_chars,
            "max_total_context_chars": self.max_total_context_chars,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ConfigurationError(
                    f"GenerationConfig.{name} debe ser > 0 (recibido {value})."
                )
        if self.context_strategy not in STRATEGIES:
            raise ConfigurationError(
                f"GenerationConfig.context_strategy inválida: {self.context_strategy!r} "
                f"(esperado uno de {STRATEGIES})."
            )


class AnswerGenerator:
    """Orquesta una respuesta fundamentada o una abstención trazable a partir de una pregunta."""

    def __init__(
        self,
        *,
        retriever: DenseRetriever,
        llm_client: LlmClient,
        config: GenerationConfig | None = None,
        prompts_dir: str | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm_client = llm_client
        self.config = config or GenerationConfig()
        self.prompts_dir = prompts_dir

    def answer(
        self,
        question: str,
        *,
        query_profile_id: str | None = None,
        filters: dict | None = None,
    ) -> RagAnswerV1:
        """Devuelve `RagAnswerV1` (respuesta fundamentada o abstención determinista)."""
        profile_id = query_profile_id or self.config.query_profile_id
        resolved_profile = self.retriever.resolved_query_profile_id(profile_id)

        hits = self.retriever.retrieve(
            question,
            query_profile_id=profile_id,
            top_k=self.config.top_k,
            filters=filters,
        )
        if not hits:
            return self._abstain(
                "No se recuperó ninguna evidencia del corpus indexado para esta pregunta.",
                hits=[],
                selection=EvidenceSelection(),
                resolved_profile=resolved_profile,
            )

        selection = build_evidences(
            hits,
            parents_by_id=self.retriever.corpus.get("parents_by_id", {}),
            max_evidences=self.config.max_evidences,
            context_strategy=self.config.context_strategy,
            context_budget_chars=self.config.context_budget_chars,
            max_total_context_chars=self.config.max_total_context_chars,
        )
        if not selection.evidences:
            return self._abstain(
                "No se pudo construir evidencia utilizable a partir de los pasajes recuperados.",
                hits=hits,
                selection=selection,
                resolved_profile=resolved_profile,
            )

        messages = build_messages(
            question=question,
            evidences=selection.evidences,
            prompts_dir=self.prompts_dir,
        )
        llm_answer, metrics = self.llm_client.chat(
            messages,
            temperature=self.config.temperature,
            seed=self.config.seed,
            num_predict=self.config.num_predict,
            num_ctx=self.config.num_ctx,
            keep_alive=self.config.keep_alive,
        )

        trace = self._build_trace(hits, selection, resolved_profile)

        if not llm_answer.answered:
            return RagAnswerV1(
                answered=False,
                answer="",
                citations=[],
                abstention_reason=llm_answer.abstention_reason,
                disclaimer=DISCLAIMER,
                retrieval_trace=trace,
                generation_metrics=metrics,
            )

        allowed = {ev.evidence_id for ev in selection.evidences}
        unknown = [cid for cid in llm_answer.citation_ids if cid not in allowed]
        if unknown:
            raise GenerationContractError(
                f"el LLM citó identificadores no entregados: {sorted(set(unknown))}"
            )

        citations = self._enrich_citations(llm_answer.citation_ids, selection.evidences)
        return RagAnswerV1(
            answered=True,
            answer=llm_answer.answer,
            citations=citations,
            abstention_reason="",
            disclaimer=DISCLAIMER,
            retrieval_trace=trace,
            generation_metrics=metrics,
        )

    # -- baselines (descomposición de error: recuperación vs generación) -----
    def answer_with_evidences(
        self,
        question: str,
        selection: EvidenceSelection,
        *,
        query_profile_id: str | None = None,
    ) -> RagAnswerV1:
        """Genera con evidencias YA seleccionadas (baseline *oracle*): salta el retrieval.

        Usa la MISMA ruta de prompt/LLM/validación de IDs/citas que `answer()`, pero sobre las
        evidencias inyectadas (p. ej. las gold del banco). Sin evidencias utilizables ⇒ abstención
        determinista sin llamar al LLM.
        """
        resolved_profile = self.retriever.resolved_query_profile_id(
            query_profile_id or self.config.query_profile_id
        )
        trace = self._trace_for_injected(selection, resolved_profile)
        if not selection.evidences:
            return RagAnswerV1(
                answered=False,
                answer="",
                citations=[],
                abstention_reason="No se dispuso de evidencia gold utilizable para esta pregunta.",
                disclaimer=DISCLAIMER,
                retrieval_trace=trace,
                generation_metrics=None,
            )
        messages = build_messages(
            question=question, evidences=selection.evidences, prompts_dir=self.prompts_dir
        )
        llm_answer, metrics = self.llm_client.chat(
            messages,
            temperature=self.config.temperature,
            seed=self.config.seed,
            num_predict=self.config.num_predict,
            num_ctx=self.config.num_ctx,
            keep_alive=self.config.keep_alive,
        )
        if not llm_answer.answered:
            return RagAnswerV1(
                answered=False,
                answer="",
                citations=[],
                abstention_reason=llm_answer.abstention_reason,
                disclaimer=DISCLAIMER,
                retrieval_trace=trace,
                generation_metrics=metrics,
            )
        allowed = {ev.evidence_id for ev in selection.evidences}
        unknown = [cid for cid in llm_answer.citation_ids if cid not in allowed]
        if unknown:
            raise GenerationContractError(
                f"el LLM citó identificadores no entregados: {sorted(set(unknown))}"
            )
        citations = self._enrich_citations(llm_answer.citation_ids, selection.evidences)
        return RagAnswerV1(
            answered=True,
            answer=llm_answer.answer,
            citations=citations,
            abstention_reason="",
            disclaimer=DISCLAIMER,
            retrieval_trace=trace,
            generation_metrics=metrics,
        )

    def generate_closed_book(self, question: str) -> ClosedBookResult:
        """Baseline closed-book: responde SIN evidencia (solo conocimiento paramétrico)."""
        return _generate_closed_book(
            self.llm_client,
            question,
            temperature=self.config.temperature,
            seed=self.config.seed,
            num_predict=self.config.num_predict,
            num_ctx=self.config.num_ctx,
            keep_alive=self.config.keep_alive,
        )

    # -- helpers -------------------------------------------------------------
    def _abstain(
        self,
        reason: str,
        *,
        hits: list[DenseHit],
        selection: EvidenceSelection,
        resolved_profile: str,
    ) -> RagAnswerV1:
        """Abstención determinista (sin llamada al LLM): mantiene aviso y trazabilidad."""
        return RagAnswerV1(
            answered=False,
            answer="",
            citations=[],
            abstention_reason=reason,
            disclaimer=DISCLAIMER,
            retrieval_trace=self._build_trace(hits, selection, resolved_profile),
            generation_metrics=None,
        )

    def _build_trace(
        self,
        hits: list[DenseHit],
        selection: EvidenceSelection,
        resolved_profile: str,
    ) -> RetrievalTraceV1:
        # Clave por hit exacto (parent_id, rank ganador): evita marcar duplicados del mismo parent.
        evidence_by_hit = {
            (ev.parent_id, ev.retrieval_rank): ev.evidence_id for ev in selection.evidences
        }
        hit_traces = [
            RetrievalHitTraceV1(
                rank=h.rank,
                score=h.score,
                document_id=h.document_id,
                block_id=h.block_id,
                parent_id=h.parent_id,
                selected=(h.parent_id, h.rank) in evidence_by_hit,
                evidence_id=evidence_by_hit.get((h.parent_id, h.rank)),
            )
            for h in hits
        ]
        omitted = [
            EvidenceOmissionTraceV1(
                parent_id=o["parent_id"],
                retrieval_rank=o["retrieval_rank"],
                reason=o["reason"],
                char_count=o.get("char_count"),
            )
            for o in selection.omitted
        ]
        return RetrievalTraceV1(
            bundle_id=self.retriever.bundle_id,
            model_alias=self.retriever.model_alias,
            query_profile_id=resolved_profile,
            top_k=self.config.top_k,
            returned_hits=len(hits),
            selected_evidences=len(selection.evidences),
            duplicate_parents_removed=selection.duplicate_parents_removed,
            total_context_chars=selection.total_char_count,
            hits=hit_traces,
            omitted_evidences=omitted,
        )

    def _trace_for_injected(
        self, selection: EvidenceSelection, resolved_profile: str
    ) -> RetrievalTraceV1:
        """Traza para evidencias inyectadas (oracle): las gold hacen de 'hits' entregados."""
        hits = [
            RetrievalHitTraceV1(
                rank=ev.retrieval_rank,
                score=ev.score,
                document_id=ev.document_id,
                block_id=ev.block_id,
                parent_id=ev.parent_id,
                selected=True,
                evidence_id=ev.evidence_id,
            )
            for ev in selection.evidences
        ]
        omitted = [
            EvidenceOmissionTraceV1(
                parent_id=o["parent_id"],
                retrieval_rank=o["retrieval_rank"],
                reason=o["reason"],
                char_count=o.get("char_count"),
            )
            for o in selection.omitted
        ]
        return RetrievalTraceV1(
            bundle_id=self.retriever.bundle_id,
            model_alias=self.retriever.model_alias,
            query_profile_id=resolved_profile,
            top_k=len(selection.evidences),
            returned_hits=len(hits),
            selected_evidences=len(selection.evidences),
            duplicate_parents_removed=selection.duplicate_parents_removed,
            total_context_chars=selection.total_char_count,
            hits=hits,
            omitted_evidences=omitted,
        )

    @staticmethod
    def _enrich_citations(
        citation_ids: list[str], evidences: list[GenerationEvidence]
    ) -> list[AnswerCitationV1]:
        """Resuelve cada ID citado a su cita autoritativa (label/url del corpus), sin duplicar."""
        evidence_by_id = {ev.evidence_id: ev for ev in evidences}
        citations: list[AnswerCitationV1] = []
        seen: set[str] = set()
        for cid in citation_ids:
            if cid in seen:
                continue
            seen.add(cid)
            ev = evidence_by_id[cid]
            citations.append(
                AnswerCitationV1(
                    evidence_id=ev.evidence_id,
                    parent_id=ev.parent_id,
                    document_id=ev.document_id,
                    block_id=ev.block_id,
                    label=ev.label,
                    url=ev.url,
                )
            )
        return citations
