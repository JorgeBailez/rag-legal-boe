"""Tests de la preparación de inputs (J1/J2/C1, overflow, anchors) con fakes deterministas."""

import hashlib

import pytest

from src.embeddings.input_preparation import (
    AnchorResolutionError,
    OverflowNotRepairedError,
    prepare_inputs,
)
from src.embeddings.model_registry import get_contract
from tests.dense_fakes import FakeWordTokenizer, synthetic_corpus


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _corpus():
    c = synthetic_corpus()
    return c["chunks"], c["parents_by_id"]


# --- J1 / J2 ----------------------------------------------------------------


def test_j1_uses_retrieval_text_and_sets_anchor() -> None:
    chunks, parents = _corpus()
    tok = FakeWordTokenizer(model_max_length=512, special=2)
    out = prepare_inputs(
        "J1", chunks=chunks, parents_by_id=parents, contract=get_contract("e5-base"), tokenizer=tok
    )
    assert len(out.rows) == len(chunks)  # sin overflow → 1 row por chunk
    a2 = next(r for r in out.rows if r["block_id"] == "a2")
    assert a2["source"] == {
        "kind": "chunk_field",
        "chunk_id": "BOE-A-0001__a2__c001",
        "field": "retrieval_text",
    }
    assert a2["context_anchor"] == {"paragraph_start": 1, "paragraph_end": 3}
    # El texto codificado es format_document(retrieval_text) y su sha está en la row.
    idx = out.rows.index(a2)
    assert out.texts[idx].startswith("passage: Contexto.")
    assert a2["formatted_input_sha256"] == _sha(out.texts[idx])


def test_j2_uses_text_field() -> None:
    chunks, parents = _corpus()
    tok = FakeWordTokenizer(model_max_length=512, special=2)
    out = prepare_inputs(
        "J2", chunks=chunks, parents_by_id=parents, contract=get_contract("e5-base"), tokenizer=tok
    )
    a1 = next(r for r in out.rows if r["block_id"] == "a1" and r["document_id"] == "BOE-A-0001")
    assert a1["source"]["field"] == "text"
    idx = out.rows.index(a1)
    assert "Contexto." not in out.texts[idx]  # J2 no lleva el prefijo de contexto del chunk


def test_row_ids_unique_and_index_continuous() -> None:
    chunks, parents = _corpus()
    tok = FakeWordTokenizer(model_max_length=512, special=2)
    out = prepare_inputs(
        "J1", chunks=chunks, parents_by_id=parents, contract=get_contract("bge-m3"), tokenizer=tok
    )
    assert [r["row_index"] for r in out.rows] == list(range(len(out.rows)))
    ids = [r["embedding_input_id"] for r in out.rows]
    assert len(set(ids)) == len(ids)
    assert out.report["n_truncated"] == 0
    assert out.report["max_token_count"] <= out.effective_max_tokens


# --- C1 ---------------------------------------------------------------------


def test_c1_windows_within_parent_no_cross_and_overlap() -> None:
    chunks, parents = _corpus()
    tok = FakeWordTokenizer(model_max_length=30, special=0)
    out = prepare_inputs(
        "C1",
        chunks=chunks,
        parents_by_id=parents,
        contract=get_contract("bge-m3"),  # sin prefijo documental
        tokenizer=tok,
        overlap=10,
        safety_margin=0,
    )
    assert out.rows, "C1 debe producir rows"
    for r in out.rows:
        assert r["source"]["kind"] == "derived_text"
        assert r["source"]["origin"] == "fixed_token_window"
        assert r["source"]["chunk_id"] is None
        assert r["context_anchor"] is not None
        assert r["token_count"] <= out.effective_max_tokens  # sin truncado

    # El parent largo se divide en varias ventanas con overlap y cobertura completa.
    long_rows = [r for r in out.rows if r["parent_id"] == "BOE-A-0002__a1"]
    assert len(long_rows) > 1
    long_rows.sort(key=lambda r: r["source"]["segment_index"])
    for prev, nxt in zip(long_rows, long_rows[1:], strict=False):
        assert nxt["source"]["token_start"] == prev["source"]["token_end"] - 10
    assert long_rows[0]["source"]["token_start"] == 0
    # Cobertura: la unión de ventanas cubre todos los tokens del parent.
    covered = set()
    for r in long_rows:
        covered |= set(range(r["source"]["token_start"], r["source"]["token_end"]))
    total = len(tok.encode(parents["BOE-A-0002__a1"]["text"], add_special_tokens=False))
    assert covered == set(range(total))


# --- overflow repair --------------------------------------------------------


def test_overflow_repair_splits_long_chunk() -> None:
    chunks, parents = _corpus()
    # Tokenizer pequeño: el chunk largo (parent a1 de D2) excede el límite y se repara.
    tok = FakeWordTokenizer(model_max_length=40, special=2)
    out = prepare_inputs(
        "J2",
        chunks=chunks,
        parents_by_id=parents,
        contract=get_contract("bge-m3"),
        tokenizer=tok,
        overlap=10,
        safety_margin=0,
    )
    derived = [r for r in out.rows if r["source"]["kind"] == "derived_text"]
    assert derived, "debe haber rows derivadas por overflow"
    assert all(r["source"]["origin"] == "overflow_repair" for r in derived)
    assert all(r["source"]["chunk_id"] == "BOE-A-0002__a1__c001" for r in derived)
    assert all(r["context_anchor"] is not None for r in derived)
    assert all(r["token_count"] <= out.effective_max_tokens for r in out.rows)  # sin truncado
    assert out.report["n_overflow_repaired_inputs"] >= 1
    # segment_count coherente.
    counts = {r["source"]["segment_count"] for r in derived}
    assert counts == {len(derived)} or len(derived) >= 2


def test_overflow_repair_refines_anchor_per_window_and_discards_auxiliary_context() -> None:
    parent_id = "BOE-A-0001__a1"
    parent = {
        "parent_id": parent_id,
        "document_id": "BOE-A-0001",
        "block_id": "a1",
        "text": "p1a p1b\np2a p2b",
        "paragraphs": [
            {"order": 1, "text": "p1a p1b"},
            {"order": 2, "text": "p2a p2b"},
        ],
    }
    chunk = {
        "chunk_id": "BOE-A-0001__a1__c001",
        "parent_id": parent_id,
        "document_id": "BOE-A-0001",
        "block_id": "a1",
        "text": parent["text"],
        "retrieval_text": "CTX AUX\np1a p1b\np2a p2b",
    }
    out = prepare_inputs(
        "J1",
        chunks=[chunk],
        parents_by_id={parent_id: parent},
        contract=get_contract("bge-m3"),
        tokenizer=FakeWordTokenizer(model_max_length=2, special=0),
        overlap=0,
        safety_margin=0,
    )
    assert [r["context_anchor"] for r in out.rows] == [
        {"paragraph_start": 1, "paragraph_end": 1},
        {"paragraph_start": 2, "paragraph_end": 2},
    ]
    assert all(r["source"]["origin"] == "overflow_repair" for r in out.rows)
    assert all(r["source"]["text"] != "CTX AUX" for r in out.rows)
    assert out.report["n_auxiliary_context_windows_discarded"] == 1


def test_overflow_repair_blocks_when_legal_anchor_cannot_be_resolved() -> None:
    parent_id = "BOE-A-0001__a1"
    parent = {
        "parent_id": parent_id,
        "document_id": "BOE-A-0001",
        "block_id": "a1",
        "text": "p1a p1b",
        "paragraphs": [{"order": 1, "text": "p1a p1b"}],
    }
    chunk = {
        "chunk_id": "BOE-A-0001__a1__c001",
        "parent_id": parent_id,
        "document_id": "BOE-A-0001",
        "block_id": "a1",
        "text": parent["text"],
        "retrieval_text": "resumen auxiliar sin literal juridico largo",
    }
    with pytest.raises(AnchorResolutionError, match="context_anchor preciso"):
        prepare_inputs(
            "J1",
            chunks=[chunk],
            parents_by_id={parent_id: parent},
            contract=get_contract("bge-m3"),
            tokenizer=FakeWordTokenizer(model_max_length=2, special=0),
            overlap=0,
            safety_margin=0,
        )


def test_anchor_resolves_repeated_paragraph_sequence_by_cursor() -> None:
    # Párrafos repetidos en el parent (equipamiento de vehículos a)/b), como el art. 9 del RGC):
    # el último chunk casa en 2 posiciones; el cursor por parent lo ancla a la SEGUNDA (no falla
    # con "secuencia ambigua" ni lo confunde con la primera ocurrencia).
    parent_id = "BOE-A-0003__a9"
    paras = [
        {"order": 1, "text": "Intro."},
        {"order": 2, "text": "a) Apertura:"},
        {"order": 3, "text": "Rotativo naranja."},
        {"order": 4, "text": "Luces encendidas."},
        {"order": 5, "text": "b) Cierre:"},
        {"order": 6, "text": "Rotativo naranja."},
        {"order": 7, "text": "Luces encendidas."},
    ]
    parent = {
        "parent_id": parent_id,
        "document_id": "BOE-A-0003",
        "block_id": "a9",
        "text": "\n".join(p["text"] for p in paras),
        "paragraphs": paras,
    }

    def _chunk(idx: int, text: str) -> dict:
        return {
            "chunk_id": f"{parent_id}__c{idx:03d}",
            "parent_id": parent_id,
            "document_id": "BOE-A-0003",
            "block_id": "a9",
            "text": text,
            "retrieval_text": text,
        }

    chunks = [
        _chunk(1, "Intro.\na) Apertura:\nRotativo naranja.\nLuces encendidas.\nb) Cierre:"),
        _chunk(2, "Rotativo naranja.\nLuces encendidas."),  # casa en orders 3-4 y 6-7
    ]
    out = prepare_inputs(
        "J1",
        chunks=chunks,
        parents_by_id={parent_id: parent},
        contract=get_contract("e5-base"),
        tokenizer=FakeWordTokenizer(model_max_length=10000, special=0),
    )
    anchors = {r["source"].get("chunk_id"): r["context_anchor"] for r in out.rows}
    assert anchors[f"{parent_id}__c001"] == {"paragraph_start": 1, "paragraph_end": 5}
    # Sin el cursor esto fallaría (ambigua) o anclaría a 3-4; debe ser la segunda ocurrencia.
    assert anchors[f"{parent_id}__c002"] == {"paragraph_start": 6, "paragraph_end": 7}


def test_unrepairable_overflow_raises() -> None:
    chunks, parents = _corpus()
    # Presupuesto imposible: límite efectivo = special tokens ⇒ no hay sitio para contenido.
    tok = FakeWordTokenizer(model_max_length=2, special=2)
    with pytest.raises(OverflowNotRepairedError):
        prepare_inputs(
            "J2",
            chunks=chunks,
            parents_by_id=parents,
            contract=get_contract("bge-m3"),
            tokenizer=tok,
            safety_margin=0,
        )


def test_unknown_view_rejected() -> None:
    chunks, parents = _corpus()
    tok = FakeWordTokenizer()
    with pytest.raises(ValueError):
        prepare_inputs(
            "ZZ",
            chunks=chunks,
            parents_by_id=parents,
            contract=get_contract("bge-m3"),
            tokenizer=tok,
        )


def test_anchor_uses_contiguous_sequence_with_repeated_paragraphs() -> None:
    parent_id = "BOE-A-0001__a1"
    parent = {
        "parent_id": parent_id,
        "document_id": "BOE-A-0001",
        "block_id": "a1",
        "text": "Intro\nRepetido\nRepetido\nFinal",
        "paragraphs": [
            {"order": 1, "text": "Intro"},
            {"order": 2, "text": "Repetido"},
            {"order": 3, "text": "Repetido"},
            {"order": 4, "text": "Final"},
        ],
    }
    chunk = {
        "chunk_id": "BOE-A-0001__a1__c001",
        "parent_id": parent_id,
        "document_id": "BOE-A-0001",
        "block_id": "a1",
        "text": "Repetido\nFinal",
        "retrieval_text": "Repetido\nFinal",
    }
    out = prepare_inputs(
        "J2",
        chunks=[chunk],
        parents_by_id={parent_id: parent},
        contract=get_contract("bge-m3"),
        tokenizer=FakeWordTokenizer(),
    )
    assert out.rows[0]["context_anchor"] == {"paragraph_start": 3, "paragraph_end": 4}


def test_anchor_repeated_sequence_resolves_to_first_without_cursor() -> None:
    # Chunk único cuyo texto se repite en el parent: sin cursor previo (es el primero) el anclaje
    # se resuelve DETERMINISTA a la primera ocurrencia, en vez de fallar. Un mal anclaje solo
    # afectaría al ventaneo de overflow, no a la identidad del chunk → preferible a bloquear el
    # índice. Para sub-chunks con hermanos, el cursor elige la ocurrencia correcta (test by_cursor).
    parent_id = "BOE-A-0001__a1"
    parent = {
        "parent_id": parent_id,
        "document_id": "BOE-A-0001",
        "block_id": "a1",
        "text": "Repetido\nRepetido\nRepetido",
        "paragraphs": [
            {"order": 1, "text": "Repetido"},
            {"order": 2, "text": "Repetido"},
            {"order": 3, "text": "Repetido"},
        ],
    }
    chunk = {
        "chunk_id": "BOE-A-0001__a1__c001",
        "parent_id": parent_id,
        "document_id": "BOE-A-0001",
        "block_id": "a1",
        "text": "Repetido\nRepetido",
        "retrieval_text": "Repetido\nRepetido",
    }
    out = prepare_inputs(
        "J2",
        chunks=[chunk],
        parents_by_id={parent_id: parent},
        contract=get_contract("bge-m3"),
        tokenizer=FakeWordTokenizer(),
    )
    assert out.rows[0]["context_anchor"] == {"paragraph_start": 1, "paragraph_end": 2}


class NewlineTokenizer:
    model_max_length = 4

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokens: list[str] = []
        current = []
        for ch in text:
            if ch == "\n":
                if current:
                    tokens.append("".join(current))
                    current = []
                tokens.append("\n")
            elif ch.isspace():
                if current:
                    tokens.append("".join(current))
                    current = []
            else:
                current.append(ch)
        if current:
            tokens.append("".join(current))
        return list(range(len(tokens)))

    def decode(self, ids: list[int]) -> str:
        # Solo se usa para este texto de prueba: a b c \n d \n e f
        vocab = ["a", "b", "c", "\n", "d", "\n", "e", "f"]
        out = ""
        for token in (vocab[i] for i in ids):
            if token == "\n":
                out = out.rstrip() + "\n"
            else:
                out += ("" if not out or out.endswith("\n") else " ") + token
        return out


def test_c1_anchor_offsets_include_newline_tokens() -> None:
    parent_id = "BOE-A-0001__a1"
    parent = {
        "parent_id": parent_id,
        "document_id": "BOE-A-0001",
        "block_id": "a1",
        "text": "a b c\nd\ne f",
        "paragraphs": [
            {"order": 1, "text": "a b c"},
            {"order": 2, "text": "d"},
            {"order": 3, "text": "e f"},
        ],
    }
    chunk = {
        "chunk_id": "BOE-A-0001__a1__c001",
        "parent_id": parent_id,
        "document_id": "BOE-A-0001",
        "block_id": "a1",
        "text": parent["text"],
        "retrieval_text": parent["text"],
    }
    out = prepare_inputs(
        "C1",
        chunks=[chunk],
        parents_by_id={parent_id: parent},
        contract=get_contract("bge-m3"),
        tokenizer=NewlineTokenizer(),
        overlap=0,
        safety_margin=0,
    )
    segment = next(r for r in out.rows if r["source"]["text"] == "d\ne f")
    assert segment["context_anchor"] == {"paragraph_start": 2, "paragraph_end": 3}
