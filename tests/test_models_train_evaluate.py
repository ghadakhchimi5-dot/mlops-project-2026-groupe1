"""End-to-end tests for the train and evaluate stages on synthetic data.

Runs the real stage functions against temporary parquet splits and a
temporary file-based MLflow backend (sqlite), checking artifacts, registry
promotion rules and MLflow logging.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from carereturn.config import FEATURES
from carereturn.models.evaluate import age_band_metrics, run_evaluation
from carereturn.models.train import run_training
from tests.test_models_fixtures import build_dataset, write_splits

TEST_MODELS: dict[str, dict[str, Any]] = {
    "dummy": {"strategy": "prior"},
    "logreg": {"C": 1.0, "class_weight": "balanced", "max_iter": 1000},
    "random_forest": {
        "n_estimators": 30,
        "max_depth": 8,
        "min_samples_leaf": 5,
        "class_weight": "balanced_subsample",
        "n_jobs": -1,
    },
}
PARAMS = {"seed": 42, "train": {"budget_fraction": 0.2, "models": TEST_MODELS}}


class TrainedEnv:
    """Container for the paths and summary of one trained environment."""

    def __init__(self, root: Path, params: dict[str, Any], n_rows: int = 600) -> None:
        """Train a model into ``root`` and record every artifact path.

        Args:
            root: Temporary directory used for all artifacts.
            params: Params dict passed to the training stage.
            n_rows: Size of the synthetic dataset.
        """
        self.root = root
        self.splits = write_splits(build_dataset(n_rows, seed=7), root / "data" / "processed")
        self.models_dir = root / "models"
        self.reports_dir = root / "reports"
        self.tracking_uri = f"sqlite:///{root / 'mlflow.db'}"
        self.summary = run_training(
            train_file=self.splits["train"],
            val_file=self.splits["val"],
            model_file=self.models_dir / "model.joblib",
            meta_file=self.models_dir / "model_meta.json",
            val_metrics_file=self.reports_dir / "val_metrics.json",
            figures_dir=self.reports_dir / "figures",
            tracking_uri=self.tracking_uri,
            experiment_name="carereturn-tests",
            registered_model_name="carereturn-test-model",
            params=params,
        )

    @property
    def model_file(self) -> Path:
        """Path to the saved pipeline."""
        return self.models_dir / "model.joblib"

    @property
    def meta_file(self) -> Path:
        """Path to the saved metadata JSON."""
        return self.models_dir / "model_meta.json"


@pytest.fixture(scope="module")
def trained_env(tmp_path_factory: pytest.TempPathFactory) -> TrainedEnv:
    """One trained environment shared by the evaluation tests."""
    return TrainedEnv(tmp_path_factory.mktemp("trained"), PARAMS)


@pytest.fixture(scope="module")
def meta(trained_env: TrainedEnv) -> dict[str, Any]:
    """Model metadata of the trained environment."""
    return json.loads(trained_env.meta_file.read_text(encoding="utf-8"))


def test_train_writes_artifacts(trained_env: TrainedEnv, meta: dict[str, Any]) -> None:
    assert trained_env.model_file.exists()
    assert trained_env.meta_file.exists()
    required = {
        "model_name",
        "model_version",
        "threshold",
        "budget_fraction",
        "features",
        "val_metrics",
        "trained_at",
        "git_commit",
        "mlflow_run_id",
    }
    assert required <= set(meta)
    assert meta["features"] == FEATURES
    assert 0.0 <= meta["threshold"] <= 1.0
    assert meta["budget_fraction"] == pytest.approx(0.2)
    assert meta["val_metrics"]["threshold"] == pytest.approx(meta["threshold"])
    assert meta["trained_at"].endswith("Z") or "+00:00" in meta["trained_at"]


def test_train_registers_champion(trained_env: TrainedEnv, meta: dict[str, Any]) -> None:
    assert trained_env.summary["registered"] is True
    assert meta["model_version"] != "local"
    client = MlflowClient(tracking_uri=trained_env.tracking_uri)
    registered = client.get_registered_model("carereturn-test-model")
    assert str(registered.aliases.get("champion")) == meta["model_version"]


def test_train_logged_candidate_runs(trained_env: TrainedEnv, meta: dict[str, Any]) -> None:
    client = MlflowClient(tracking_uri=trained_env.tracking_uri)
    experiment = client.get_experiment_by_name("carereturn-tests")
    runs = client.search_runs([experiment.experiment_id])
    candidates = {run.data.tags.get("candidate") for run in runs}
    assert candidates == set(TEST_MODELS)
    champion_run = client.get_run(meta["mlflow_run_id"])
    assert "average_precision" in champion_run.data.metrics
    assert "data_md5_train" in champion_run.data.params
    assert champion_run.data.params["git_commit"] != ""


def test_val_metrics_report(trained_env: TrainedEnv) -> None:
    report = json.loads((trained_env.reports_dir / "val_metrics.json").read_text(encoding="utf-8"))
    assert report["selected"] in TEST_MODELS
    assert set(report["candidates"]) == set(TEST_MODELS)
    for candidate_metrics in report["candidates"].values():
        assert "average_precision" in candidate_metrics
        assert "recall_at_k" in candidate_metrics
    # The champion gate: the selected model must beat the dummy on AP.
    assert (
        report["candidates"][report["selected"]]["average_precision"]
        > report["candidates"]["dummy"]["average_precision"]
    )


def test_pipeline_roundtrip(meta: dict[str, Any], trained_env: TrainedEnv) -> None:
    pipeline = joblib.load(trained_env.model_file)
    frame = pd.read_parquet(trained_env.splits["test"])
    probas = pipeline.predict_proba(frame[FEATURES])
    assert probas.shape == (len(frame), 2)
    assert np.isfinite(probas).all()


def test_evaluate_writes_report_and_figures(trained_env: TrainedEnv, meta: dict[str, Any]) -> None:
    report = run_evaluation(
        test_file=trained_env.splits["test"],
        model_file=trained_env.model_file,
        meta_file=trained_env.meta_file,
        test_metrics_file=trained_env.reports_dir / "test_metrics.json",
        figures_dir=trained_env.reports_dir / "figures",
        tracking_uri=trained_env.tracking_uri,
        experiment_name="carereturn-tests",
    )
    assert (trained_env.reports_dir / "test_metrics.json").exists()
    for figure in ("pr_curve.png", "roc_curve.png", "calibration.png", "confusion_matrix.png"):
        assert (trained_env.reports_dir / "figures" / figure).exists()
    metrics = report["metrics"]
    assert 0.0 <= metrics["average_precision"] <= 1.0
    assert report["n_test"] == 120  # 20% of 600
    assert metrics["threshold"] == pytest.approx(meta["threshold"])
    # Test split is small: no age band reaches the 200-row gate.
    assert report["age_bands"] == {}


def test_evaluate_logs_onto_champion_run(trained_env: TrainedEnv, meta: dict[str, Any]) -> None:
    run_evaluation(
        test_file=trained_env.splits["test"],
        model_file=trained_env.model_file,
        meta_file=trained_env.meta_file,
        test_metrics_file=trained_env.reports_dir / "test_metrics.json",
        figures_dir=trained_env.reports_dir / "figures",
        tracking_uri=trained_env.tracking_uri,
        experiment_name="carereturn-tests",
    )
    client = MlflowClient(tracking_uri=trained_env.tracking_uri)
    data = client.get_run(meta["mlflow_run_id"]).data
    assert "test_average_precision" in data.metrics
    assert "test_recall_at_k" in data.metrics
    assert "test_cm_tp" in data.metrics


def test_evaluate_requires_artifacts(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_evaluation(
            test_file=tmp_path / "missing.parquet",
            model_file=tmp_path / "missing.joblib",
            meta_file=tmp_path / "missing.json",
            figures_dir=tmp_path / "figures",
            tracking_uri="file:///tmp/unused-mlruns",
            experiment_name="unused",
        )


def test_age_band_metrics_gate_and_values() -> None:
    rng = np.random.default_rng(0)
    n_big, n_small = 250, 50
    y = rng.binomial(1, 0.2, n_big + n_small)
    scores = rng.random(n_big + n_small)
    ages = ["[70-80)"] * n_big + ["[80-90)"] * n_small
    bands = age_band_metrics(y, scores, pd.Series(ages), k_fraction=0.2)
    assert set(bands) == {"[70-80)"}  # only bands with n >= 200 are reported
    assert bands["[70-80)"]["n"] == n_big
    assert 0.0 <= bands["[70-80)"]["recall_at_k"] <= 1.0


def test_train_skips_registry_when_dummy_wins(tmp_path: Path) -> None:
    env = TrainedEnv(
        tmp_path,
        {"seed": 42, "train": {"budget_fraction": 0.2, "models": {"dummy": {"strategy": "prior"}}}},
        n_rows=200,
    )
    assert env.summary["registered"] is False
    meta_local = json.loads(env.meta_file.read_text(encoding="utf-8"))
    assert meta_local["model_version"] == "local"
    client = MlflowClient(tracking_uri=env.tracking_uri)
    assert client.search_registered_models() == []
