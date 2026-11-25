"""Shared fixtures for the data-pipeline tests.

The synthetic frame uses the real 50 raw column names but tiny deterministic
values, so every test runs without the DVC-tracked real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from carereturn.data.io import read_code_mappings, read_raw
from carereturn.data.validate import EXPECTED_RAW_COLUMNS

_MEDICINE_COLUMNS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]


def _make_synthetic_raw(n_patients: int = 24, encounters_per_patient: int = 3) -> pd.DataFrame:
    """Build a small fake raw frame with the real 50 raw column names.

    Args:
        n_patients: Number of distinct patients.
        encounters_per_patient: Encounters per patient.

    Returns:
        A DataFrame whose columns are exactly ``EXPECTED_RAW_COLUMNS``, with
        literal ``?`` missing tokens and ``None`` lab results as in the raw CSV.
    """
    rows = []
    encounter_id = 100_000
    for patient in range(1, n_patients + 1):
        for _ in range(encounters_per_patient):
            encounter_id += 1
            row: dict[str, object] = {
                "encounter_id": encounter_id,
                "patient_nbr": 10_000 + patient,
                "race": ["Caucasian", "AfricanAmerican", "?", "Hispanic"][encounter_id % 4],
                "gender": ["Female", "Male"][encounter_id % 2],
                "age": ["[40-50)", "[60-70)", "[70-80)", "?"][encounter_id % 4],
                "weight": "?" if encounter_id % 3 else "[75-100)",
                "admission_type_id": 1 + (encounter_id % 3),
                "discharge_disposition_id": [1, 2, 3, 6, 11, 13][encounter_id % 6],
                "admission_source_id": [1, 7, 7, 1, 2, 7][encounter_id % 6],
                "time_in_hospital": 1 + (encounter_id % 10),
                "payer_code": "?" if encounter_id % 2 else "MC",
                "medical_specialty": "?" if encounter_id % 3 else "InternalMedicine",
                "num_lab_procedures": (encounter_id * 7) % 100,
                "num_procedures": encounter_id % 6,
                "num_medications": 1 + (encounter_id % 40),
                "number_outpatient": encounter_id % 5,
                "number_emergency": encounter_id % 3,
                "number_inpatient": encounter_id % 4,
                "diag_1": ["250.83", "428.0", "?", "780.99"][encounter_id % 4],
                "diag_2": "648.8",
                "diag_3": "?",
                "number_diagnoses": 1 + (encounter_id % 9),
                "max_glu_serum": ["None", "Norm", "None", ">200"][encounter_id % 4],
                "A1Cresult": ["None", ">8", "Norm", "None"][encounter_id % 4],
                "change": ["No", "Ch"][encounter_id % 2],
                "diabetesMed": ["Yes", "No"][encounter_id % 2],
                "readmitted": ["<30", ">30", "NO"][encounter_id % 3],
            }
            for medicine in _MEDICINE_COLUMNS:
                row[medicine] = ["No", "Steady", "No", "Up"][encounter_id % 4]
            rows.append(row)
    frame = pd.DataFrame(rows, columns=EXPECTED_RAW_COLUMNS)
    numeric_columns = [
        "encounter_id",
        "patient_nbr",
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
    ]
    frame[numeric_columns] = frame[numeric_columns].astype("int64")
    return frame


@pytest.fixture()
def synthetic_raw_df() -> pd.DataFrame:
    """Small fake raw frame with the real 50 column names (literal strings)."""
    return _make_synthetic_raw()


@pytest.fixture()
def raw_df(synthetic_raw_df: pd.DataFrame, tmp_path) -> pd.DataFrame:
    """Synthetic raw frame as actually read from disk by ``read_raw``.

    The fixture round-trips ``synthetic_raw_df`` through a CSV so that ``?``
    became NaN and ``"None"`` stayed a string, mirroring production semantics.
    """
    path = tmp_path / "synthetic_raw.csv"
    synthetic_raw_df.to_csv(path, index=False)
    return read_raw(path)


@pytest.fixture()
def mapping_codes(tmp_path) -> dict[str, set[int]]:
    """Valid code sets per column, parsed from a synthetic IDS_mapping file."""
    lines = ["admission_type_id,description"]
    lines += [f"{code},type-{code}" for code in range(1, 9)]
    lines += [",", "discharge_disposition_id,description"]
    lines += [f"{code},disp-{code}" for code in range(1, 31)]
    lines += [",", "admission_source_id,description"]
    lines += [f"{code},source-{code}" for code in range(1, 27)]
    path = tmp_path / "IDS_mapping.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return read_code_mappings(path)


@pytest.fixture()
def cohort_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Prepared cohort (exclusions applied, target mapped) from the raw fixture."""
    from carereturn.data.prepare import prepare_cohort

    return prepare_cohort(raw_df)


@pytest.fixture()
def rng() -> np.random.Generator:
    """Deterministic random generator for tests that need extra noise."""
    return np.random.default_rng(42)
