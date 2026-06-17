"""Tests de los experimentos de evaluación añadidos al bake-off denso (lógica pura, offline).

Cubre: estratificación por grupo, bootstrap pareado vs baseline, análisis de umbral de abstención
(ROC-AUC + balanced accuracy) y frontera de Pareto calidad/coste.
"""

import pytest

from src.evaluation.metrics import (
    PRIMARY_METRIC,
    abstention_threshold_analysis,
    aggregate_metric_groups,
    paired_vs_baseline,
    pareto_front,
)


def test_aggregate_metric_groups_media_y_tamano_por_estrato() -> None:
    groups = {
        "ciudadana": [{PRIMARY_METRIC: 0.8}, {PRIMARY_METRIC: 1.0}],
        "lexica": [{PRIMARY_METRIC: 0.2}],
    }
    out = aggregate_metric_groups(groups, seed=7)
    assert out["ciudadana"]["n"] == 2
    assert out["ciudadana"][PRIMARY_METRIC] == 0.9  # media (0.8 + 1.0) / 2
    assert out["lexica"]["n"] == 1
    assert out["lexica"][PRIMARY_METRIC] == 0.2
    # El IC de la primaria queda disponible por estrato.
    ci_keys = {"mean", "ci_low", "ci_high", "seed", "n_resamples"}
    assert ci_keys <= set(out["ciudadana"]["primary_ci"])


def test_paired_vs_baseline_excluye_baseline_y_calcula_diferencia() -> None:
    primary_by_run = {
        "base": [0.5, 0.5, 0.5, 0.5],
        "mejor": [0.7, 0.7, 0.7, 0.7],
        "peor": [0.3, 0.3, 0.3, 0.3],
    }
    out = paired_vs_baseline(primary_by_run, "base", seed=7)
    assert set(out) == {"mejor", "peor"}  # el baseline no se compara consigo mismo
    assert out["mejor"]["mean_diff"] == pytest.approx(0.2)  # 0.7 - 0.5
    assert out["peor"]["mean_diff"] == pytest.approx(-0.2)
    # Diferencia constante ⇒ IC degenerado en el propio valor (no cruza 0).
    assert out["mejor"]["ci_low"] > 0 and out["mejor"]["ci_high"] > 0


def test_paired_vs_baseline_baseline_inexistente() -> None:
    try:
        paired_vs_baseline({"a": [0.1]}, "no-existe")
    except KeyError:
        return
    raise AssertionError("se esperaba KeyError por baseline inexistente")


def test_abstention_separacion_perfecta() -> None:
    # In-corpus con score alto, out_of_corpus con score bajo: separación total.
    out = abstention_threshold_analysis([0.8, 0.9, 0.85], [0.1, 0.2, 0.15])
    assert out["auc"] == 1.0
    assert out["balanced_accuracy"] == 1.0
    assert out["tpr"] == 1.0 and out["tnr"] == 1.0
    assert 0.2 < out["best_threshold"] <= 0.8
    assert out["n_answerable"] == 3 and out["n_unanswerable"] == 3


def test_abstention_sin_separacion() -> None:
    # Distribuciones idénticas: AUC 0.5 (puro azar).
    out = abstention_threshold_analysis([0.5, 0.5], [0.5, 0.5])
    assert out["auc"] == 0.5


def test_abstention_listas_vacias() -> None:
    out = abstention_threshold_analysis([], [0.1, 0.2])
    assert out["auc"] == 0.0 and out["n_answerable"] == 0 and out["n_unanswerable"] == 2


def test_pareto_front_descarta_dominados() -> None:
    # quality = nDCG (más alto mejor), cost = latencia (más bajo mejor).
    points = [
        {"run_key": "A", "q": 0.9, "c": 10.0},  # mejor calidad, coste medio → frontera
        {"run_key": "B", "q": 0.7, "c": 2.0},  # peor calidad, coste mínimo → frontera
        {"run_key": "C", "q": 0.7, "c": 8.0},  # dominado por B (igual q, más coste)
        {"run_key": "D", "q": 0.6, "c": 12.0},  # dominado por A y por B
    ]
    front = {p["run_key"] for p in pareto_front(points, quality_key="q", cost_key="c")}
    assert front == {"A", "B"}


def test_pareto_front_un_solo_punto() -> None:
    pts = [{"run_key": "X", "q": 0.5, "c": 1.0}]
    assert [p["run_key"] for p in pareto_front(pts, quality_key="q", cost_key="c")] == ["X"]
