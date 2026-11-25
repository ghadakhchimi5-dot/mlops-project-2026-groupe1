"""Train candidate models, select the champion and register it with MLflow.

Usage: ``python -m carereturn.models.train``.

Candidates (dummy, logistic regression, random forest) are fit on the train
split, evaluated on the validation split, and the best by average precision
(ties broken by recall@k) is saved with a decision threshold matching the
follow-up budget. The champion is only registered in the MLflow Model
Registry when it beats the dummy baseline on average precision.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from carereturn import config
from carereturn.models.features import build_pipeline
from carereturn.models.metrics import (
    compute_metrics,
    plot_calibration,
    plot_pr_curve,
    plot_roc_curve,
    threshold_for_budget,
)
from carereturn.models.tracking import make_client, provenance_params, setup_mlflow

LOGGER = logging.getLogger(__name__)

THRESHOLD_FALLBACK = 0.5

ESTIMATOR_FACTORIES: Mapping[str, type] = {
    "dummy": DummyClassifier,
    "logreg": LogisticRegression,
    "random_forest": RandomForestClassifier,
}


def _build_estimators(models_params: Mapping[str, Mapping[str, Any]], seed: int) -> dict[str, Any]:
    """Instantiate the candidate estimators described in params.yaml.

    Args:
        models_params: Mapping from candidate name to its hyperparameters
            (``params["train"]["models"]``).
        seed: Random seed shared by all candidates.

    Returns:
        Mapping from candidate name to an unfitted estimator.

    Raises:
        ValueError: If a candidate name is not supported.

    """
    estimators: dict[str, Any] = {}
    for name, hyperparams in models_params.items():
        factory = ESTIMATOR_FACTORIES.get(name)
        if factory is None:
            raise ValueError(
                f"Unknown model '{name}' in params.yaml; supported: {sorted(ESTIMATOR_FACTORIES)}"
            )
        estimators[name] = factory(**dict(hyperparams), random_state=seed)
    return estimators


def _load_split(path: Path) -> pd.DataFrame:
    """Load a split parquet file and check the feature/target schema.

    Args:
        path: Parquet file holding ``ID_COLUMNS + FEATURES + TARGET``.

    Returns:
        The loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.

    """
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path} (run the data pipeline first)")
    frame = pd.read_parquet(path)
    required = [*config.FEATURES, config.TARGET]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return frame


def _select_candidate(candidate_metrics: Mapping[str, Mapping[str, Any]]) -> str:
    """Pick the best candidate by average precision, ties by recall@k.

    Args:
        candidate_metrics: Candidate name to validation metrics.

    Returns:
        Name of the selected candidate.

    """
    ranked = sorted(
        candidate_metrics.items(),
        key=lambda item: (
            -round(float(item[1]["average_precision"]), 10),
            -float(item[1]["recall_at_k"]),
        ),
    )
    return ranked[0][0]


def _flatten_metrics(prefix: str, metrics: Mapping[str, Any]) -> dict[str, float]:
    """Flatten a metrics dict into MLflow-logged numeric metrics.

    Args:
        prefix: Key prefix such as ``test_``.
        metrics: Metrics dict possibly containing a confusion matrix sub-dict.

    Returns:
        Flat mapping of metric name to float (NaN kept for undefined values).

    """
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, Mapping):
            flat.update({f"{prefix}cm_{sub}": float(v) for sub, v in value.items()})
        else:
            flat[f"{prefix}{key}"] = float(value)
    return flat


def run_training(
    *,
    train_file: Path = config.SPLIT_FILES["train"],
    val_file: Path = config.SPLIT_FILES["val"],
    model_file: Path = config.MODEL_FILE,
    meta_file: Path = config.MODEL_META_FILE,
    val_metrics_file: Path = config.VAL_METRICS_FILE,
    figures_dir: Path = config.FIGURES_DIR,
    tracking_uri: str = config.MLFLOW_TRACKING_URI,
    experiment_name: str = config.MLFLOW_EXPERIMENT,
    registered_model_name: str = config.REGISTERED_MODEL_NAME,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit all candidates, select and save the champion, log everything to MLflow.

    Args:
        train_file: Train split parquet.
        val_file: Validation split parquet.
        model_file: Destination for the selected fitted pipeline (joblib).
        meta_file: Destination for the model metadata JSON.
        val_metrics_file: Destination for the candidates validation report.
        figures_dir: Directory for candidate validation figures.
        tracking_uri: MLflow tracking URI.
        experiment_name: MLflow experiment name.
        registered_model_name: Model Registry name used for the champion.
        params: Full params dict (defaults to ``config.load_params()``).

    Returns:
        Summary dict with ``selected``, ``threshold``, ``registered``,
        ``model_version`` and per-candidate validation metrics.

    """
    full_params: Mapping[str, Any] = params if params is not None else config.load_params()
    train_params: Mapping[str, Any] = full_params.get("train", {})
    seed = int(full_params.get("seed", 0))
    budget_fraction = float(train_params.get("budget_fraction", 0.20))
    models_params: Mapping[str, Mapping[str, Any]] = train_params.get("models", {})

    train_frame = _load_split(train_file)
    val_frame = _load_split(val_file)
    X_train, y_train = train_frame[config.FEATURES], train_frame[config.TARGET].astype(int)  # noqa: N806
    X_val, y_val = val_frame[config.FEATURES], val_frame[config.TARGET].astype(int)  # noqa: N806
    LOGGER.info(
        "Loaded train=%d rows (prevalence=%.4f), val=%d rows (prevalence=%.4f)",
        len(X_train),
        y_train.mean(),
        len(X_val),
        y_val.mean(),
    )

    experiment_id = setup_mlflow(tracking_uri, experiment_name)
    provenance = provenance_params({"train": train_file, "val": val_file})
    git_commit = provenance.get("git_commit", "unknown")

    estimators = _build_estimators(models_params, seed)
    fitted: dict[str, Pipeline] = {}
    candidate_metrics: dict[str, dict[str, Any]] = {}
    run_ids: dict[str, str] = {}

    for name, estimator in estimators.items():
        LOGGER.info("Training candidate '%s'", name)
        pipeline = build_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline

        val_scores = pipeline.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val, val_scores, THRESHOLD_FALLBACK, budget_fraction)
        candidate_metrics[name] = metrics

        candidate_figures = figures_dir / "val" / name
        figure_paths = [
            plot_pr_curve(
                y_val, val_scores, candidate_figures / "pr_curve.png", f"VAL PR - {name}"
            ),
            plot_roc_curve(
                y_val, val_scores, candidate_figures / "roc_curve.png", f"VAL ROC - {name}"
            ),
            plot_calibration(
                y_val,
                val_scores,
                candidate_figures / "calibration.png",
                title=f"VAL calibration - {name}",
            ),
        ]

        with mlflow.start_run(
            run_name=f"train_{name}",
            experiment_id=experiment_id,
            tags={"stage": "train", "candidate": name, "git_commit": git_commit},
        ) as run:
            candidate_params = {
                f"{name}_{key}": value for key, value in models_params.get(name, {}).items()
            }
            mlflow.log_params(
                {
                    "model": name,
                    "seed": seed,
                    "budget_fraction": budget_fraction,
                    **candidate_params,
                    **provenance,
                }
            )
            mlflow.log_metrics(_flatten_metrics("", metrics))
            for figure_path in figure_paths:
                mlflow.log_artifact(str(figure_path), artifact_path="val_figures")
            run_ids[name] = run.info.run_id
        headline = {
            key: metrics[key]
            for key in ("average_precision", "roc_auc", "recall_at_k", "precision_at_k")
        }
        LOGGER.info("Candidate '%s' validation metrics: %s", name, headline)

    selected = _select_candidate(candidate_metrics)
    selected_ap = float(candidate_metrics[selected]["average_precision"])
    dummy_ap = candidate_metrics.get("dummy", {}).get("average_precision")
    beats_dummy = dummy_ap is None or selected_ap > float(dummy_ap)
    LOGGER.info(
        "Selected candidate: '%s' (AP=%.4f, beats dummy=%s)", selected, selected_ap, beats_dummy
    )

    champion = fitted[selected]
    val_scores = champion.predict_proba(X_val)[:, 1]
    threshold = threshold_for_budget(val_scores, budget_fraction)
    final_metrics = compute_metrics(y_val, val_scores, threshold, budget_fraction)
    LOGGER.info(
        "Budget threshold=%.4f, val recall@k=%.4f, val AP=%.4f",
        threshold,
        float(final_metrics["recall_at_k"]),
        float(final_metrics["average_precision"]),
    )

    registered = False
    model_version: str = "local"
    if beats_dummy:
        with mlflow.start_run(run_id=run_ids[selected]):
            signature = infer_signature(
                X_train.head(200), champion.predict_proba(X_train.head(200))[:, 1]
            )
            result = mlflow.sklearn.log_model(
                sk_model=champion,
                artifact_path="model",
                signature=signature,
                input_example=X_train.head(5),
                registered_model_name=registered_model_name,
            )
            model_version = str(result.registered_model_version)
            registered = True
            mlflow.log_param("threshold", threshold)
            mlflow.log_metrics(_flatten_metrics("selected_", final_metrics))
        client = make_client(tracking_uri)
        client.set_registered_model_alias(registered_model_name, "champion", model_version)
        LOGGER.info(
            "Registered '%s' version %s with alias 'champion'",
            registered_model_name,
            model_version,
        )
    else:
        LOGGER.warning(
            "Selected model does not beat the dummy baseline on AP; skipping registration"
        )

    trained_at = datetime.now(UTC).isoformat(timespec="seconds")
    meta = {
        "model_name": registered_model_name,
        "model_version": model_version,
        "algorithm": selected,
        "threshold": threshold,
        "budget_fraction": budget_fraction,
        "features": list(config.FEATURES),
        "val_metrics": final_metrics,
        "trained_at": trained_at,
        "git_commit": git_commit if git_commit != "unknown" else None,
        "mlflow_run_id": run_ids.get(selected),
    }
    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion, model_file)
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_file, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)

    val_report = {
        "selected": selected,
        "registered": registered,
        "threshold": threshold,
        "budget_fraction": budget_fraction,
        "selection_rule": "average_precision, ties broken by recall_at_k",
        "champion_gate": {
            "selected_average_precision": selected_ap,
            "dummy_average_precision": dummy_ap,
            "beats_dummy": beats_dummy,
        },
        "candidates": candidate_metrics,
        "selected_at_threshold": final_metrics,
    }
    val_metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(val_metrics_file, "w", encoding="utf-8") as handle:
        json.dump(val_report, handle, indent=2, ensure_ascii=False)

    return {
        "selected": selected,
        "threshold": threshold,
        "registered": registered,
        "model_version": model_version,
        "candidates": candidate_metrics,
        "mlflow_run_id": run_ids.get(selected),
    }


def main() -> None:
    """Run the training stage with default config paths."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    summary = run_training()
    LOGGER.info(
        "Training done: selected=%s threshold=%.4f registered=%s version=%s",
        summary["selected"],
        summary["threshold"],
        summary["registered"],
        summary["model_version"],
    )


if __name__ == "__main__":
    main()
