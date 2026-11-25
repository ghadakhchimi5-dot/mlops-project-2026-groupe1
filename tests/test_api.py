"""Tests for the FastAPI serving layer (unit tests, synthetic model artifacts).

A tiny sklearn Pipeline matching the artifact contract (cast-to-str +
OneHotEncoder(handle_unknown="ignore") + LogisticRegression) is fitted on
synthetic data, dumped to tmp_path and served through the app via
MODEL_PATH / MODEL_META_PATH environment overrides.
"""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from carereturn import config
from carereturn.api.main import app
from carereturn.api.schemas import MAX_BATCH_SIZE, Encounter

AGE_BANDS = [f"[{lo}-{lo + 10})" for lo in range(0, 100, 10)]
A1C_VALUES = ["None", "Norm", ">7", ">8"]
MAX_GLU_VALUES = ["None", "Norm", ">200", ">300"]
INSULIN_VALUES = ["No", "Steady", "Up", "Down"]
CHANGE_VALUES = ["No", "Ch"]
DIABETESMED_VALUES = ["No", "Yes"]

MODEL_NAME = "readmission-logreg"
MODEL_VERSION = "test-0.1.0"
THRESHOLD = 0.3

VALID_PAYLOAD: dict[str, int | str] = {
    "time_in_hospital": 5,
    "num_lab_procedures": 41,
    "num_procedures": 1,
    "num_medications": 12,
    "number_outpatient": 2,
    "number_emergency": 0,
    "number_inpatient": 1,
    "number_diagnoses": 7,
    "age": "[70-80)",
    "admission_type_id": 1,
    "admission_source_id": 7,
    "discharge_disposition_id": 3,
    "A1Cresult": ">7",
    "max_glu_serum": "None",
    "insulin": "Up",
    "change": "Ch",
    "diabetesMed": "Yes",
}


def cast_features_to_str(frame: pd.DataFrame) -> pd.DataFrame:
    """Cast the categorical block to str (module-level so joblib can pickle it)."""
    return frame.astype(str)


def make_synthetic_frame(n_rows: int = 240, seed: int = 0) -> pd.DataFrame:
    """Synthetic encounters respecting the contract value domains."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "time_in_hospital": rng.integers(1, 15, n_rows),
            "num_lab_procedures": rng.integers(0, 100, n_rows),
            "num_procedures": rng.integers(0, 7, n_rows),
            "num_medications": rng.integers(1, 40, n_rows),
            "number_outpatient": rng.integers(0, 10, n_rows),
            "number_emergency": rng.integers(0, 10, n_rows),
            "number_inpatient": rng.integers(0, 10, n_rows),
            "number_diagnoses": rng.integers(1, 17, n_rows),
            "age": rng.choice(AGE_BANDS, n_rows),
            "admission_type_id": rng.integers(1, 9, n_rows),
            "admission_source_id": rng.integers(1, 22, n_rows),
            "discharge_disposition_id": rng.integers(1, 29, n_rows),
            "A1Cresult": rng.choice(A1C_VALUES, n_rows),
            "max_glu_serum": rng.choice(MAX_GLU_VALUES, n_rows),
            "insulin": rng.choice(INSULIN_VALUES, n_rows),
            "change": rng.choice(CHANGE_VALUES, n_rows),
            "diabetesMed": rng.choice(DIABETESMED_VALUES, n_rows),
        }
    )
    return frame[config.FEATURES]


def make_targets(frame: pd.DataFrame, seed: int = 1) -> np.ndarray:
    """Binary target correlated with a few features so the model can learn."""
    rng = np.random.default_rng(seed)
    signal = (
        0.4 * frame["number_inpatient"] / 10.0
        + 0.3 * frame["time_in_hospital"] / 14.0
        + 0.2 * frame["age"].map(AGE_BANDS.index)
        + rng.normal(0.0, 0.5, len(frame))
    )
    return (signal > signal.median()).astype(int).to_numpy()


def make_pipeline() -> Pipeline:
    """Tiny pipeline matching the artifact contract."""
    categorical = Pipeline(
        steps=[
            ("to_str", FunctionTransformer(cast_features_to_str)),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("num", "passthrough", config.NUMERIC_FEATURES),
            ("cat", categorical, config.CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[("preprocessing", preprocessing), ("clf", LogisticRegression(max_iter=500))]
    )


def make_meta() -> dict:
    """Metadata payload matching the artifact contract keys."""
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "threshold": THRESHOLD,
        "budget_fraction": 0.2,
        "features": config.FEATURES,
        "val_metrics": {"roc_auc": 0.71, "average_precision": 0.24},
        "trained_at": "2026-01-01T00:00:00Z",
        "git_commit": "deadbeef",
        "mlflow_run_id": "run-123",
    }


@pytest.fixture()
def model_dir(tmp_path):
    """Fit the tiny pipeline, dump artifacts to tmp_path, return the directory."""
    frame = make_synthetic_frame()
    targets = make_targets(frame)
    pipeline = make_pipeline().fit(frame, targets)
    joblib.dump(pipeline, tmp_path / "model.joblib")
    (tmp_path / "model_meta.json").write_text(json.dumps(make_meta()), encoding="utf-8")
    return tmp_path


@pytest.fixture()
def client(model_dir, monkeypatch):
    """Test client with the synthetic model loaded via env overrides."""
    monkeypatch.setenv("MODEL_PATH", str(model_dir / "model.joblib"))
    monkeypatch.setenv("MODEL_META_PATH", str(model_dir / "model_meta.json"))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def missing_model_client(monkeypatch, tmp_path):
    """Test client whose env points at nonexistent model files."""
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "nope" / "model.joblib"))
    monkeypatch.setenv("MODEL_META_PATH", str(tmp_path / "nope" / "model_meta.json"))
    with TestClient(app) as test_client:
        yield test_client


# --- Contract guards --------------------------------------------------------


def test_encounter_fields_match_config_contract():
    """The Encounter schema must mirror config.FEATURES exactly (in order)."""
    assert list(Encounter.model_fields) == list(config.FEATURES)


# --- Health -----------------------------------------------------------------


def test_health_ok_when_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == MODEL_VERSION


def test_health_503_when_model_missing(missing_model_client):
    response = missing_model_client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "unavailable"
    assert body["model_loaded"] is False
    assert body["model_version"] is None


# --- Single predict ---------------------------------------------------------


def test_predict_valid_encounter(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["risk_score"] <= 1.0
    assert isinstance(body["flagged"], bool)
    assert body["flagged"] == (body["risk_score"] >= body["threshold"])
    assert body["threshold"] == pytest.approx(THRESHOLD)
    assert body["model_name"] == MODEL_NAME
    assert body["model_version"] == MODEL_VERSION


def test_predict_accepts_code_id_outside_training_levels(client):
    """Pipeline ignores unknown levels, so unseen positive code ids still work."""
    payload = VALID_PAYLOAD | {"admission_type_id": 99}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


def test_predict_missing_field_returns_422(client):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "num_medications"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_unknown_extra_field_returns_422(client):
    payload = VALID_PAYLOAD | {"patient_nbr": 123}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_negative_count_returns_422(client):
    payload = VALID_PAYLOAD | {"number_inpatient": -1}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_zero_time_in_hospital_returns_422(client):
    payload = VALID_PAYLOAD | {"time_in_hospital": 0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_age_band_returns_422(client):
    payload = VALID_PAYLOAD | {"age": "60-70"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_categorical_value_returns_422(client):
    payload = VALID_PAYLOAD | {"insulin": "Sometimes"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_string_count_returns_422(client):
    payload = VALID_PAYLOAD | {"num_medications": "12"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_503_when_model_missing(missing_model_client):
    response = missing_model_client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 503


# --- Batch predict ----------------------------------------------------------


def test_predict_batch_valid(client):
    payloads = [VALID_PAYLOAD | {"time_in_hospital": days} for days in (2, 5, 9)]
    response = client.post("/predict/batch", json={"encounters": payloads})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 3
    assert len(body["predictions"]) == 3
    for prediction in body["predictions"]:
        assert 0.0 <= prediction["risk_score"] <= 1.0
        assert isinstance(prediction["flagged"], bool)
        assert prediction["model_version"] == MODEL_VERSION


def test_predict_batch_too_large_returns_422(client):
    payloads = [VALID_PAYLOAD | {"time_in_hospital": 2}] * (MAX_BATCH_SIZE + 1)
    response = client.post("/predict/batch", json={"encounters": payloads})
    assert response.status_code == 422


def test_predict_batch_empty_returns_422(client):
    response = client.post("/predict/batch", json={"encounters": []})
    assert response.status_code == 422


def test_predict_batch_invalid_encounter_returns_422(client):
    payloads = [VALID_PAYLOAD, VALID_PAYLOAD | {"age": "not-a-band"}]
    response = client.post("/predict/batch", json={"encounters": payloads})
    assert response.status_code == 422
