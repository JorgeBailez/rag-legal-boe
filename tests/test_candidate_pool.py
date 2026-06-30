"""Tests del pooling de candidatos para el gold (lógica pura, offline, sin torch)."""

from src.evaluation.candidate_pool import (
    build_pool,
    pool_to_jsonl_records,
    render_worksheet,
)

# Dos "sistemas" (bundle+perfil) con solapamiento parcial de parents por pregunta.
_SYSTEMS = [
    {
        "bundle_id": "e5-large-instruct__j1__abc",
        "query_profile_id": "I1_LEGAL",
        "hits_by_qid": {
            "q1": [
                {"parent_id": "N__a21", "rank": 1, "score": 0.90},
                {"parent_id": "N__a24", "rank": 2, "score": 0.70},
                {"parent_id": "N__a21", "rank": 5, "score": 0.40},  # dup en el mismo sistema
            ],
            "qooc": [{"parent_id": "N__a99", "rank": 1, "score": 0.30}],
        },
    },
    {
        "bundle_id": "bge-m3__j1__def",
        "query_profile_id": "BASELINE",
        "hits_by_qid": {
            "q1": [
                {"parent_id": "N__a21", "rank": 2, "score": 0.85},
                {"parent_id": "N__a30", "rank": 3, "score": 0.60},
            ],
        },
    },
]

_JUDGED = {
    "q1": {
        "N__a21": {"relevance": 2, "review_status": "draft"},
        "N__aXX": {"relevance": 1, "review_status": "draft"},  # juzgado, NO recuperado
    },
}

_PARENTS = {
    "N__a21": {
        "parent_id": "N__a21",
        "citation": {"label": "Ley X art. 21", "url": "https://boe/a21"},
        "paragraphs": [{"order": 1, "text": "Primer parrafo."}, {"order": 2, "text": "Segundo."}],
    },
    "N__a24": {"parent_id": "N__a24", "citation": {"label": "Ley X art. 24"}, "paragraphs": []},
    "N__a30": {
        "parent_id": "N__a30",
        "is_annex": True,
        "paragraphs": [{"order": 1, "text": "Anexo."}],
    },
    "N__aXX": {"parent_id": "N__aXX", "paragraphs": [{"order": 1, "text": "Juzgado no pooled."}]},
    "N__a99": {"parent_id": "N__a99", "is_without_content": True, "paragraphs": []},
}

_QUESTIONS = {
    "q1": {
        "query_id": "q1",
        "split": "development",
        "query": "¿plazo?",
        "query_style": "ciudadana",
        "answer_scope": "single_parent",
        "failure_mode": None,
        "difficulty": "media",
    },
    "qooc": {
        "query_id": "qooc",
        "split": "out_of_corpus",
        "query": "¿horario oficina?",
        "query_style": "sin_respuesta",
        "answer_scope": "none",
        "failure_mode": "out_of_corpus",
    },
}


def _pool():
    return build_pool(_SYSTEMS, _JUDGED, ["q1", "qooc"])


def test_build_pool_dedup_y_consenso() -> None:
    pool = _pool()
    cands = {c["parent_id"]: c for c in pool["q1"]["candidates"]}
    a21 = cands["N__a21"]
    # a21 lo encuentran los DOS sistemas (n_systems=2); dentro de e5 se queda el mejor rank (1).
    assert a21["n_systems"] == 2
    assert a21["best_rank"] == 1
    assert a21["best_score"] == 0.90
    aliases = {f["bundle_id"].split("__")[0] for f in a21["found_by"]}
    assert aliases == {"e5-large-instruct", "bge-m3"}
    # current_relevance se copia del judgment existente.
    assert a21["current_relevance"] == 2 and a21["from_judgment"] is False


def test_build_pool_orden_estable_por_consenso() -> None:
    pool = _pool()
    pids = [c["parent_id"] for c in pool["q1"]["candidates"]]
    # a21 (2 sistemas) va primero; aXX (juzgado no pooled, 0 sistemas) va al final.
    assert pids[0] == "N__a21"
    assert pids[-1] == "N__aXX"


def test_build_pool_inyecta_juzgado_no_recuperado() -> None:
    pool = _pool()
    assert pool["q1"]["judged_not_pooled"] == ["N__aXX"]
    axx = next(c for c in pool["q1"]["candidates"] if c["parent_id"] == "N__aXX")
    assert axx["found_by"] == [] and axx["from_judgment"] is True
    assert axx["n_systems"] == 0 and axx["current_relevance"] == 1


def test_build_pool_ooc_sin_juzgados() -> None:
    pool = _pool()
    ooc = pool["qooc"]
    assert ooc["judged_not_pooled"] == []
    # Tiene candidatos (para confirmar abstención / negativos duros) pero ninguno juzgado.
    assert [c["current_relevance"] for c in ooc["candidates"]] == [None]


def test_render_worksheet_contenido() -> None:
    md = render_worksheet(_pool(), _PARENTS, _QUESTIONS, worksheet_top=10)
    assert "Ley X art. 21" in md and "`N__a21`" in md
    assert "[JUZGADO rel=2 draft]" in md  # marcador de juzgado
    assert "[1] Primer parrafo." in md and "[2] Segundo." in md  # párrafos numerados por order
    assert "e5-large-instruct/I1_LEGAL#1" in md  # found_by compacto
    assert "Sin contenido" in md  # flag del parent OOC a99
    assert "NINGÚN sistema recuperó" in md  # aviso de judged_not_pooled


def test_render_worksheet_juzgado_siempre_visible_pese_al_corte() -> None:
    # Con worksheet_top=1, el juzgado-no-pooled (último) igual debe aparecer.
    md = render_worksheet(_pool(), _PARENTS, _QUESTIONS, worksheet_top=1)
    assert "`N__aXX`" in md


def test_pool_to_jsonl_records() -> None:
    records = pool_to_jsonl_records(_pool(), _QUESTIONS)
    by_qid = {r["query_id"]: r for r in records}
    assert set(by_qid) == {"q1", "qooc"}
    assert by_qid["q1"]["query_style"] == "ciudadana"
    assert by_qid["qooc"]["answer_scope"] == "none"
    assert any(c["parent_id"] == "N__a21" for c in by_qid["q1"]["candidates"])
