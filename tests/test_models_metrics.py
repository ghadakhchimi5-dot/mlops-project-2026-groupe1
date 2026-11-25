"""Unit tests for carereturn.models.metrics on hand-made examples."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from carereturn.models.metrics import (
    compute_metrics,
    precision_at_k,
    recall_at_k,
    threshold_for_budget,
)


def test_recall_at_k_handmade() -> None:
    y_true = [1, 1, 0, 0, 0, 1, 0, 0, 1, 0]
    scores = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5, 0.4, 0.1]
    # Top 30% = first 3 rows, containing 2 of the 4 positives.
    assert recall_at_k(y_true, scores, 0.3) == pytest.approx(0.5)


def test_precision_at_k_handmade() -> None:
    y_true = [1, 1, 0, 0, 0, 1, 0, 0, 1, 0]
    scores = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5, 0.4, 0.1]
    assert precision_at_k(y_true, scores, 0.3) == pytest.approx(2 / 3)
    assert precision_at_k(y_true, scores, 1.0) == pytest.approx(0.4)


def test_recall_at_k_with_pandas_inputs() -> None:
    y_true = pd.Series([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    scores = pd.Series(np.linspace(1.0, 0.1, 10))
    # Top 20% of 10 rows = 2 rows, holding 1 of the 2 positives.
    assert recall_at_k(y_true, scores, 0.2) == pytest.approx(0.5)
    assert precision_at_k(y_true, scores, 0.2) == pytest.approx(0.5)


def test_recall_at_k_no_positives_returns_zero() -> None:
    y_true = [0, 0, 0, 0]
    scores = [0.3, 0.9, 0.5, 0.1]
    assert recall_at_k(y_true, scores, 0.5) == 0.0


def test_recall_at_k_ties_use_first_rows() -> None:
    # Constant scores: the stable argsort flags the first k rows.
    y_true = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    scores = [0.5] * 10
    assert recall_at_k(y_true, scores, 0.2) == pytest.approx(0.5)


def test_threshold_for_budget_flags_top_k_fraction() -> None:
    scores = np.arange(100, dtype=float)  # 0..99
    threshold = threshold_for_budget(scores, 0.2)
    assert threshold == pytest.approx(79.2)
    flagged = (scores >= threshold).sum()
    assert flagged == 20


def test_recall_at_k_matches_threshold_semantics() -> None:
    rng = np.random.default_rng(0)
    scores = rng.random(500)
    y_true = rng.binomial(1, 0.2, 500)
    threshold = threshold_for_budget(scores, 0.2)
    flagged = scores >= threshold
    manual_recall = y_true[flagged].sum() / max(1, y_true.sum())
    assert recall_at_k(y_true, scores, 0.2) == pytest.approx(manual_recall)


def test_compute_metrics_handmade_values() -> None:
    y_true = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.35, 0.8]
    metrics = compute_metrics(y_true, scores, threshold=0.5, k_fraction=0.5)
    # Predictions at 0.5: [0, 0, 0, 1]
    assert metrics["average_precision"] == pytest.approx(5 / 6)
    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(2 * (1.0 * 0.5) / 1.5)
    assert metrics["brier"] == pytest.approx(0.158125)
    assert metrics["prevalence"] == pytest.approx(0.5)
    assert metrics["threshold"] == pytest.approx(0.5)
    # Top 50% = 2 highest scores: [0.8 (pos), 0.4 (neg)] -> captures 1 of 2.
    assert metrics["recall_at_k"] == pytest.approx(0.5)
    assert metrics["precision_at_k"] == pytest.approx(0.5)
    assert metrics["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 1, "tp": 1}


def test_compute_metrics_is_json_safe() -> None:
    y_true = np.array([1, 0, 1, 0])
    scores = np.array([0.8, 0.2, 0.6, 0.1])
    metrics = compute_metrics(y_true, scores, threshold=0.5, k_fraction=0.25)
    for key, value in metrics.items():
        if key == "confusion_matrix":
            assert all(isinstance(v, int) for v in value.values())
        else:
            assert isinstance(value, float)


def test_compute_metrics_single_class_roc_auc_is_nan() -> None:
    y_true = [1, 1, 1, 1]  # no negative label: ROC-AUC is undefined
    scores = [0.9, 0.8, 0.7, 0.6]
    metrics = compute_metrics(y_true, scores, threshold=0.5, k_fraction=0.75)
    assert math.isnan(metrics["roc_auc"])
    assert metrics["average_precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError, match="k_fraction"):
        recall_at_k([1, 0], [0.5, 0.2], 0.0)
    with pytest.raises(ValueError, match="k_fraction"):
        threshold_for_budget([0.1, 0.9], 1.5)
    with pytest.raises(ValueError, match="equal length"):
        recall_at_k([1, 0, 0], [0.5, 0.2], 0.5)
    with pytest.raises(ValueError):
        recall_at_k([], [], 0.5)
