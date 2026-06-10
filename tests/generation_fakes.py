"""Fakes deterministas para los tests de generación de Fase 3 (offline, sin Ollama ni pesos).

Reutiliza el corpus sintético y los fakes densos de `tests/dense_fakes.py`. Añade:
- `FakeLlmClient`: cumple el protocolo `LlmClient` (devuelve una `RagLlmAnswerV1` controlada).
- `FakeRetriever`: cumple la interfaz que consume `AnswerGenerator` sin bundle ni encoder reales.
- helpers para construir hits/corpus mínimos y un retriever real sobre un bundle temporal.
"""

from __future__ import annotations

from src.contracts.generation_models import OllamaMetricsV1, RagLlmAnswerV1
from src.evaluation.judge import (
    ClaimVerdictV1,
    CorrectnessVerdictV1,
    FaithfulnessVerdictV1,
)
from src.retrieval.dense_retriever import DenseHit, DenseRetriever


class FakeJudge:
    """Juez falso con la interfaz que consume `evaluate_generation` (offline, sin LLM).

    `faithfulness_claims` es una lista de bools (supported) y `correctness` una etiqueta. Registra
    las llamadas para inspección en los tests.
    """

    def __init__(
        self,
        *,
        faithfulness_claims: list[bool] | None = None,
        correctness: str = "correct",
        model_label: str = "fake-judge",
    ) -> None:
        self._claims = [True] if faithfulness_claims is None else faithfulness_claims
        self._correctness = correctness
        self.model_label = model_label
        self.faithfulness_calls: list[dict] = []
        self.correctness_calls: list[dict] = []

    def judge_faithfulness(self, *, answer: str, evidences_block: str):
        self.faithfulness_calls.append({"answer": answer, "evidences_block": evidences_block})
        verdict = FaithfulnessVerdictV1(
            claims=[ClaimVerdictV1(claim=f"c{i}", supported=s) for i, s in enumerate(self._claims)]
        )
        return verdict, OllamaMetricsV1(eval_count=5)

    def judge_correctness(self, *, question: str, answer: str, reference: str):
        self.correctness_calls.append(
            {"question": question, "answer": answer, "reference": reference}
        )
        return CorrectnessVerdictV1(verdict=self._correctness), OllamaMetricsV1(eval_count=5)


class FakeLlmClient:
    """Cliente LLM falso: devuelve una respuesta fija y registra las llamadas recibidas."""

    def __init__(
        self,
        answer: RagLlmAnswerV1,
        *,
        metrics: OllamaMetricsV1 | None = None,
    ) -> None:
        self.answer = answer
        self.metrics = metrics or OllamaMetricsV1(
            total_duration_ns=1_000_000_000, eval_count=10, eval_duration_ns=5_000_000_000
        )
        self.calls: list[dict] = []

    def chat(self, messages, **kwargs):  # noqa: ANN001, ANN003 - firma laxa para test
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return self.answer, self.metrics


class FakeRetriever:
    """Retriever falso con la interfaz que usa `AnswerGenerator` (duck typing)."""

    def __init__(
        self,
        hits: list[DenseHit],
        corpus: dict,
        *,
        bundle_id: str = "fake__j1__deadbeefcafe",
        model_alias: str = "e5-large-instruct",
        resolved_profile: str = "I2_CITIZEN_LEGISLATION",
    ) -> None:
        self._hits = hits
        self.corpus = corpus
        self.bundle_id = bundle_id
        self.model_alias = model_alias
        self._resolved_profile = resolved_profile
        self.retrieve_calls = 0

    def resolved_query_profile_id(self, query_profile_id: str | None) -> str:
        return self._resolved_profile

    def retrieve(
        self,
        query: str,
        *,
        query_profile_id: str | None = None,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[DenseHit]:
        self.retrieve_calls += 1
        return list(self._hits)[:top_k]


def make_hit(
    *,
    rank: int,
    parent_id: str,
    document_id: str = "BOE-A-0001",
    block_id: str = "a1",
    score: float = 0.9,
    label: str = "Ley 1/2000, artículo 1",
    url: str | None = "https://boe/BOE-A-0001#a1",
    text: str = "Texto recuperado del artículo.",
    anchor: dict | None = None,
) -> DenseHit:
    """Construye un `DenseHit` listo para evidencias (con context_anchor para P_EXPAND_BOUNDED)."""
    return DenseHit(
        rank=rank,
        score=score,
        row_index=rank - 1,
        embedding_input_id=f"ein_{rank:06d}",
        document_id=document_id,
        block_id=block_id,
        parent_id=parent_id,
        source={"kind": "chunk_field", "chunk_id": f"{parent_id}__c001", "field": "retrieval_text"},
        context_anchor=anchor or {"paragraph_start": 1, "paragraph_end": 1},
        retrieval_text=text,
        citation_label=label,
        citation_url=url,
    )


def make_corpus_for_parents(parent_ids: list[str]) -> dict:
    """Corpus mínimo con un parent (2 párrafos) por id, suficiente para `assemble_context`."""
    parents_by_id: dict[str, dict] = {}
    for pid in parent_ids:
        parents_by_id[pid] = {
            "parent_id": pid,
            "document_id": pid.split("__")[0],
            "block_id": pid.split("__")[-1],
            "paragraphs": [
                {"order": 1, "class": "parrafo", "text": f"Primer párrafo de {pid}."},
                {"order": 2, "class": "parrafo", "text": f"Segundo párrafo de {pid}."},
            ],
        }
    return {"chunks": [], "parents_by_id": parents_by_id, "documents_by_id": {}}


def build_bundle_retriever(tmp_path, contract, encoder) -> DenseRetriever:
    """Publica un bundle temporal con el corpus sintético y devuelve un `DenseRetriever` real.

    Construye el retriever por constructor (no `from_bundle`) para no exigir que el contrato esté
    registrado en `model_registry`: el objetivo es ejercitar índice/joins reales con fakes.
    """
    from src.embeddings.bundle import ExecutionMeta, publish_bundle
    from src.embeddings.fingerprints import source_corpus_fingerprint
    from src.embeddings.input_preparation import prepare_inputs
    from src.indexing.vector_index import ExactDenseIndex
    from tests.dense_fakes import FakeWordTokenizer, synthetic_corpus

    corpus = synthetic_corpus()
    tok = FakeWordTokenizer(model_max_length=512, special=2)
    prepared = prepare_inputs(
        "J1",
        chunks=corpus["chunks"],
        parents_by_id=corpus["parents_by_id"],
        contract=contract,
        tokenizer=tok,
    )
    embeddings = encoder.encode_documents(prepared.texts)
    result = publish_bundle(
        contract=contract,
        view="J1",
        prepared=prepared,
        embeddings=embeddings,
        source_corpus_fingerprint=source_corpus_fingerprint(
            corpus["chunks"], corpus["parents_by_id"]
        ),
        n_norms=2,
        execution=ExecutionMeta(encoder_backend="fake"),
        output_root=tmp_path,
    )
    index = ExactDenseIndex.from_bundle(result["path"], corpus=corpus)
    return DenseRetriever(index=index, encoder=encoder, contract=contract, corpus=corpus)
