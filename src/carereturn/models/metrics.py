"""Ranking and threshold metrics plus matplotlib figures for model reports.

The business goal is to capture as many early readmissions as possible inside
a limited follow-up capacity: ``recall_at_k`` and ``precision_at_k`` measure
performance when only the top ``k`` fraction of scored patients is flagged.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TypeAlias

import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg")  # headless backend for batch jobs and tests
import matplotlib.pyplot as plt  # noqa: E402

Labels: TypeAlias = "np.ndarray | pd.Series | list[int]"
Scores: TypeAlias = "np.ndarray | pd.Series | list[float]"
Confusion: TypeAlias = "dict[str, int]"


def _as_arrays(y_true: Labels, scores: Scores) -> tuple[np.ndarray, np.ndarray]:
    """Convert labels and scores to aligned numpy arrays of the same length.

    Args:
        y_true: Binary labels (0/1).
        scores: Positive-class scores or probabilities.

    Returns:
        Tuple ``(y, s)`` as numpy arrays (int and float respectively).

    Raises:
        ValueError: If inputs are empty or have different lengths.

    """
    y = np.asarray(y_true, dtype=int).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    if y.size == 0 or y.size != s.size:
        raise ValueError("y_true and scores must be non-empty and have equal length")
    return y, s


def _top_k_indices(scores: np.ndarray, k_fraction: float) -> np.ndarray:
    """Return indices of the top ``k_fraction`` highest scores.

    Args:
        scores: Positive-class scores.
        k_fraction: Fraction of patients to flag, in ``(0, 1]``.

    Returns:
        Array of indices, highest score first (stable on ties).

    Raises:
        ValueError: If ``k_fraction`` is out of range.

    """
    if not 0 < k_fraction <= 1:
        raise ValueError(f"k_fraction must be in (0, 1], got {k_fraction}")
    n = scores.size
    k = min(n, max(1, int(np.ceil(k_fraction * n))))
    return np.argsort(-scores, kind="stable")[:k]


def recall_at_k(y_true: Labels, scores: Scores, k_fraction: float) -> float:
    """Recall achieved when only the top ``k_fraction`` scored rows are flagged.

    Args:
        y_true: Binary labels (0/1).
        scores: Positive-class scores.
        k_fraction: Fraction of rows to flag (follow-up budget).

    Returns:
        Share of all positives captured inside the top-k set; 0.0 when the
        data contains no positive label.

    """
    y, s = _as_arrays(y_true, scores)
    top = _top_k_indices(s, k_fraction)
    total_positives = int(y.sum())
    if total_positives == 0:
        return 0.0
    return float(y[top].sum() / total_positives)


def precision_at_k(y_true: Labels, scores: Scores, k_fraction: float) -> float:
    """Share of true positives among the top ``k_fraction`` scored rows.

    Args:
        y_true: Binary labels (0/1).
        scores: Positive-class scores.
        k_fraction: Fraction of rows to flag (follow-up budget).

    Returns:
        Precision inside the top-k set (the flagged set holds at least one row).

    """
    y, s = _as_arrays(y_true, scores)
    top = _top_k_indices(s, k_fraction)
    return float(y[top].mean())


def threshold_for_budget(scores: Scores, k_fraction: float) -> float:
    """Score quantile flagging the top ``k_fraction`` of patients.

    Production flags ``scores >= threshold``. Ties (e.g. a constant dummy
    score) may flag more than ``k_fraction`` of rows.

    Args:
        scores: Positive-class scores.
        k_fraction: Fraction of rows to flag.

    Returns:
        The ``1 - k_fraction`` quantile of the scores.

    Raises:
        ValueError: If scores are empty or ``k_fraction`` is out of range.

    """
    s = np.asarray(scores, dtype=float).ravel()
    if s.size == 0:
        raise ValueError("scores must be non-empty")
    if not 0 < k_fraction <= 1:
        raise ValueError(f"k_fraction must be in (0, 1], got {k_fraction}")
    return float(np.quantile(s, 1.0 - k_fraction))


def compute_metrics(y_true: Labels, scores: Scores, threshold: float, k_fraction: float) -> dict:
    """Compute the full report metric set for given scores and threshold.

    Args:
        y_true: Binary labels (0/1).
        scores: Positive-class scores.
        threshold: Decision threshold; rows with ``scores >= threshold`` are
            considered flagged.
        k_fraction: Follow-up budget fraction used for recall@k/precision@k.

    Returns:
        JSON-serializable dict with ``average_precision``, ``roc_auc`` (NaN
        when undefined, e.g. single-class labels), ``recall_at_k``,
        ``precision_at_k``, ``precision``, ``recall``, ``f1``, ``brier``,
        ``prevalence``, ``threshold`` and a ``confusion_matrix`` sub-dict with
        ``tn``/``fp``/``fn``/``tp`` counts.

    """
    y, s = _as_arrays(y_true, scores)
    y_pred = (s >= float(threshold)).astype(int)

    average_precision = float(average_precision_score(y, s))
    try:
        roc_auc: float = float(roc_auc_score(y, s))
    except ValueError:  # only one class present in y_true
        roc_auc = float("nan")

    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, y_pred, labels=[0, 1]).ravel())

    return {
        "average_precision": average_precision,
        "roc_auc": roc_auc,
        "recall_at_k": recall_at_k(y, s, k_fraction),
        "precision_at_k": precision_at_k(y, s, k_fraction),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y, s)),
        "prevalence": float(y.mean()),
        "threshold": float(threshold),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


# --- Figures ---------------------------------------------------------------


def _save_figure(fig: plt.Figure, path: Path, title: str) -> Path:
    """Save a matplotlib figure to ``path`` and close it.

    Args:
        fig: Figure to save.
        path: Destination PNG path (parent directories are created).
        title: Title written on the figure.

    Returns:
        The path of the written file.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(title)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pr_curve(
    y_true: Labels,
    scores: Scores,
    path: Path,
    title: str = "Precision-Recall curve",
) -> Path:
    """Plot the precision-recall curve with the average precision in the legend.

    Args:
        y_true: Binary labels (0/1).
        scores: Positive-class scores.
        path: Destination PNG path.
        title: Figure title.

    Returns:
        The path of the written file.

    """
    y, s = _as_arrays(y_true, scores)
    precision, recall, _ = precision_recall_curve(y, s)
    ap = average_precision_score(y, s)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.step(recall, precision, where="post", label=f"AP={ap:.3f}")
    ax.axhline(y.mean(), color="grey", linestyle="--", label=f"prevalence={y.mean():.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="best")
    return _save_figure(fig, path, title)


def plot_roc_curve(y_true: Labels, scores: Scores, path: Path, title: str = "ROC curve") -> Path:
    """Plot the ROC curve with the ROC-AUC in the legend.

    Args:
        y_true: Binary labels (0/1).
        scores: Positive-class scores.
        path: Destination PNG path.
        title: Figure title.

    Returns:
        The path of the written file.

    """
    y, s = _as_arrays(y_true, scores)
    fpr, tpr, _ = roc_curve(y, s)
    auc = roc_auc_score(y, s)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"ROC-AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="best")
    return _save_figure(fig, path, title)


def plot_calibration(
    y_true: Labels,
    scores: Scores,
    path: Path,
    n_bins: int = 10,
    title: str = "Calibration",
) -> Path:
    """Plot observed vs predicted probability on quantile bins.

    Args:
        y_true: Binary labels (0/1).
        scores: Positive-class probabilities.
        path: Destination PNG path.
        n_bins: Number of quantile bins.
        title: Figure title.

    Returns:
        The path of the written file.

    """
    y, s = _as_arrays(y_true, scores)
    prob_true, prob_pred = calibration_curve(y, s, n_bins=n_bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(prob_pred, prob_true, marker="o", label="model")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="best")
    return _save_figure(fig, path, title)


def plot_confusion_matrix(
    confusion: Mapping[str, int],
    path: Path,
    title: str = "Confusion matrix",
) -> Path:
    """Plot a 2x2 confusion matrix from its ``tn``/``fp``/``fn``/``tp`` counts.

    Args:
        confusion: Mapping with keys ``tn``, ``fp``, ``fn``, ``tp``.
        path: Destination PNG path.
        title: Figure title.

    Returns:
        The path of the written file.

    """
    matrix = np.array([[confusion["tn"], confusion["fp"]], [confusion["fn"], confusion["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(matrix, cmap="Blues")
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, str(int(value)), ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], labels=["pred 0", "pred 1"])
    ax.set_yticks([0, 1], labels=["true 0", "true 1"])
    return _save_figure(fig, path, title)
