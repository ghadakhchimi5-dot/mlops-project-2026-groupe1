"""Shared constants and paths. This module is the contract between data, model and serving code."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(os.environ.get("CARERETURN_ROOT", Path(__file__).resolve().parents[2]))

# --- Paths -----------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DATA_FILE = RAW_DIR / "diabetic_data.csv"
RAW_MAPPING_FILE = RAW_DIR / "IDS_mapping.csv"
COHORT_FILE = INTERIM_DIR / "cohort.parquet"
SPLIT_FILES = {
    "train": PROCESSED_DIR / "train.parquet",
    "val": PROCESSED_DIR / "val.parquet",
    "test": PROCESSED_DIR / "test.parquet",
}
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODELS_DIR / "model.joblib"
MODEL_META_FILE = MODELS_DIR / "model_meta.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
PARAMS_FILE = PROJECT_ROOT / "params.yaml"

UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)

# --- Schema ----------------------------------------------------------------
ID_COLUMNS = ["encounter_id", "patient_nbr"]
TARGET_SOURCE = "readmitted"
TARGET = "target"  # 1 if readmitted == "<30" else 0
POSITIVE_LABEL = "<30"
TARGET_VALUES = {"<30", ">30", "NO"}

NUMERIC_FEATURES = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]
CATEGORICAL_FEATURES = [
    "age",
    "admission_type_id",
    "admission_source_id",
    "discharge_disposition_id",
    "A1Cresult",
    "max_glu_serum",
    "insulin",
    "change",
    "diabetesMed",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Administrative code columns are categorical even though stored as integers.
CODE_COLUMNS = ["admission_type_id", "admission_source_id", "discharge_disposition_id"]

# Discharge dispositions meaning death or hospice (IDS_mapping.csv): excluded from cohort.
EXCLUDED_DISCHARGE_CODES = {11, 13, 14, 19, 20, 21}

MISSING_TOKEN = "?"

VAL_METRICS_FILE = REPORTS_DIR / "val_metrics.json"
TEST_METRICS_FILE = REPORTS_DIR / "test_metrics.json"
DATA_VALIDATION_FILE = REPORTS_DIR / "data_validation.json"
SPLIT_SUMMARY_FILE = REPORTS_DIR / "split_summary.json"

MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
)
MLFLOW_EXPERIMENT = "carereturn-readmission"
REGISTERED_MODEL_NAME = "carereturn-readmission"


def load_params(path: Path | None = None) -> dict[str, Any]:
    """Load `params.yaml` (DVC parameters)."""
    with open(path or PARAMS_FILE, encoding="utf-8") as fh:
        return yaml.safe_load(fh)
