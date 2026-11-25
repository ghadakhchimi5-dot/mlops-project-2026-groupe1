"""Evaluate the selected champion on the held-out test split.

Usage: ``python -m carereturn.models.evaluate``.

Loads the pipeline and metadata saved by the training stage, scores the test
split once, computes metrics at the stored budget threshold plus per-age-band
performance, writes the test metrics report and figures, and appends the
results to the champion's MLflow run.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from carereturn import config
from carereturn.models.metrics import (
    compute_metrics,
    plot_calibration,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    recall_at_k,
)
from carereturn.models.tracking import setup_mlflow

LOGGER = logging.getLogger(__name__)

AGE_BAND_PATTERN = re.compile(r"\[(\d+)-(\d+)\)")
AGE_BAND_MIN_N = 200
TEST_FIGURES = ("pr_curve.png", "roc_curve.png", "calibration.png", "confusion_matrix.png")


def _age_band_sort_key(band: str) -> tuple[int, str]:
    """Sort key putting age bands like ``[70-80)`` in demographic order.

    Args:
        band: Raw age band label from the source data.

    Returns:
        Tuple of the band lower bound and the label itself.

    """
    match = AGE_BAND_PATTERN.match(band)
    return (int(match.group(1)) if match else 10**6, band)


def age_band_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    age_values: pd.Series,
    k_fraction: float,
    min_n: int = AGE_BAND_MIN_N,
) -> dict[str, dict[str, float | int | None]]:
    """Compute AP and recall@k per age band for bands with enough samples.

    Args:
        y_true: Binary test labels (0/1).
        scores: Test scores from the champion pipeline.
        age_values: Raw ``age`` band label per test row.
        k_fraction: Follow-up budget fraction.
        min_n: Minimum number of rows for a band to be reported.

    Returns:
        Mapping from age band to ``{"n", "average_precision", "recall_at_k"}``;
        ``average_precision`` is None when the band holds a single class.

    """
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    frame = pd.DataFrame({"age": pd.Series(age_values).astype(str).to_numpy(), "y": y, "s": s})
    bands: dict[str, dict[str, float | int | None]] = {}
    for band, group in frame.groupby("age"):
        if len(group) < min_n:
            continue
        try:
            ap: float | None = float(average_precision_score(group["y"], group["s"]))
        except ValueError:  # single-class band
            ap = None
        bands[str(band)] = {
            "n": int(len(group)),
            "average_precision": ap,
            "recall_at_k": recall_at_k(group["y"], group["s"], k_fraction),
        }
    return {band: bands[band] for band in sorted(bands, key=_age_band_sort_key)}


def run_evaluation(
    *,
    test_file: Path = config.SPLIT_FILES["test"],
    model_file: Path = config.MODEL_FILE,
    meta_file: Path = config.MODEL_META_FILE,
    test_metrics_file: Path = config.TEST_METRICS_FILE,
    figures_dir: Path = config.FIGURES_DIR,
    tracking_uri: str = config.MLFLOW_TRACKING_URI,
    experiment_name: str = config.MLFLOW_EXPERIMENT,
) -> dict[str, Any]:
    """Score the test split with the champion model and write the test report.

    Args:
        test_file: Test split parquet.
        model_file: Fitted pipeline saved by the training stage.
        meta_file: Model metadata JSON saved by the training stage.
        test_metrics_file: Destination for the test metrics report.
        figures_dir: Directory for the test figures.
        tracking_uri: MLflow tracking URI (used to log onto the champion run).
        experiment_name: MLflow experiment name.

    Returns:
        The metrics report dict that was written to ``test_metrics_file``.

    Raises:
        FileNotFoundError: If the model, metadata or test split is missing.

    """
    for path in (model_file, meta_file, test_file):
        if not path.exists():
            raise FileNotFoundError(f"Required artifact not found: {path} (run training first)")
    with open(meta_file, encoding="utf-8") as handle:
        meta: dict[str, Any] = json.load(handle)

    threshold = float(meta["threshold"])
    k_fraction = float(meta["budget_fraction"])
    pipeline = joblib.load(model_file)

    frame = pd.read_parquet(test_file)
    x_test = frame[meta.get("features", config.FEATURES)]  # noqa: N806 - ML convention
    y_test = frame[config.TARGET].astype(int)
    LOGGER.info("Loaded test split: %d rows, prevalence=%.4f", len(frame), y_test.mean())

    scores = pipeline.predict_proba(x_test)[:, 1]  # scored once, reused below
    metrics = compute_metrics(y_test, scores, threshold, k_fraction)
    LOGGER.info(
        "Test metrics: AP=%.4f ROC-AUC=%.4f recall@k=%.4f",
        float(metrics["average_precision"]),
        float(metrics["roc_auc"]),
        float(metrics["recall_at_k"]),
    )

    age_bands: dict[str, dict[str, float | int | None]] = {}
    if "age" in frame.columns:
        age_bands = age_band_metrics(y_test.to_numpy(), scores, frame["age"], k_fraction)
        LOGGER.info("Reported %d age bands with n >= %d", len(age_bands), AGE_BAND_MIN_N)

    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = {
        "pr_curve.png": plot_pr_curve(
            y_test, scores, figures_dir / "pr_curve.png", "Test PR curve"
        ),
        "roc_curve.png": plot_roc_curve(
            y_test, scores, figures_dir / "roc_curve.png", "Test ROC curve"
        ),
        "calibration.png": plot_calibration(
            y_test, scores, figures_dir / "calibration.png", title="Test calibration"
        ),
        "confusion_matrix.png": plot_confusion_matrix(
            metrics["confusion_matrix"],
            figures_dir / "confusion_matrix.png",
            "Test confusion matrix",
        ),
    }

    report = {
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_name": meta.get("model_name"),
        "model_version": meta.get("model_version"),
        "algorithm": meta.get("algorithm"),
        "mlflow_run_id": meta.get("mlflow_run_id"),
        "threshold": threshold,
        "budget_fraction": k_fraction,
        "n_test": int(len(y_test)),
        "metrics": metrics,
        "age_bands": age_bands,
    }
    test_metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(test_metrics_file, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    LOGGER.info("Wrote %s", test_metrics_file)

    run_id = meta.get("mlflow_run_id")
    if run_id:
        try:
            setup_mlflow(tracking_uri, experiment_name)
            with mlflow.start_run(run_id=run_id):
                scalar_metrics = {
                    f"test_{key}": float(value)
                    for key, value in metrics.items()
                    if key != "confusion_matrix"
                }
                cm_metrics = {
                    f"test_cm_{key}": float(value)
                    for key, value in metrics["confusion_matrix"].items()
                }
                mlflow.log_metrics({**scalar_metrics, **cm_metrics})
                for figure_path in figure_paths.values():
                    mlflow.log_artifact(str(figure_path), artifact_path="test_figures")
            LOGGER.info("Logged test metrics onto run %s", run_id)
        except Exception as exc:  # noqa: BLE001 - tracking must not void the report
            LOGGER.warning("Could not log test results to MLflow run %s: %s", run_id, exc)
    else:
        LOGGER.info("No mlflow_run_id in model metadata; skipping MLflow logging")

    return report


def main() -> None:
    """Run the evaluation stage with default config paths."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    report = run_evaluation()
    metrics = report["metrics"]
    LOGGER.info(
        "Evaluation done: AP=%.4f ROC-AUC=%.4f recall@k=%.4f (threshold=%.4f)",
        float(metrics["average_precision"]),
        float(metrics["roc_auc"]),
        float(metrics["recall_at_k"]),
        float(metrics["threshold"]),
    )


if __name__ == "__main__":
    main()
