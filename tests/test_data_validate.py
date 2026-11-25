"""Tests for carereturn.data.validate: checks and report structure."""

from __future__ import annotations

import pandas as pd
import pytest

from carereturn.config import FEATURES, ID_COLUMNS, TARGET_SOURCE
from carereturn.data.io import read_code_mappings, read_raw
from carereturn.data.validate import (
    EXPECTED_RAW_COLUMNS,
    NUMERIC_FEATURES,
    NUMERIC_RANGES,
    validate_dataframe,
)


def test_expected_columns_cover_config_contract():
    assert len(EXPECTED_RAW_COLUMNS) == 50
    assert len(set(EXPECTED_RAW_COLUMNS)) == 50
    assert set(ID_COLUMNS + FEATURES + [TARGET_SOURCE]) <= set(EXPECTED_RAW_COLUMNS)


def test_numeric_ranges_cover_all_numeric_features():
    assert set(NUMERIC_RANGES) == set(NUMERIC_FEATURES)
    for low, high in NUMERIC_RANGES.values():
        assert 0 <= low < high


def test_validation_passes_on_clean_frame(raw_df, mapping_codes):
    report = validate_dataframe(raw_df, mapping_codes)
    assert report["passed"] is True
    assert all(check["passed"] for check in report["checks"].values())
    assert set(report) == {
        "n_rows",
        "n_cols",
        "n_patients",
        "n_positive",
        "positive_rate",
        "checks",
        "passed",
    }
    assert report["n_rows"] == len(raw_df)
    assert report["n_cols"] == 50
    assert report["n_patients"] == raw_df["patient_nbr"].nunique()
    positives = int((raw_df[TARGET_SOURCE] == "<30").sum())
    assert report["n_positive"] == positives
    assert report["positive_rate"] == pytest.approx(positives / len(raw_df))


def test_validation_detects_missing_column(raw_df, mapping_codes):
    broken = raw_df.drop(columns=["age"])
    report = validate_dataframe(broken, mapping_codes)
    assert report["passed"] is False
    assert report["checks"]["schema_columns"]["passed"] is False
    assert "age" in report["checks"]["schema_columns"]["detail"]


def test_validation_detects_duplicate_encounter(raw_df, mapping_codes):
    broken = pd.concat([raw_df, raw_df.iloc[[0]]], ignore_index=True)
    report = validate_dataframe(broken, mapping_codes)
    assert report["passed"] is False
    assert report["checks"]["encounter_id_unique"]["passed"] is False


def test_validation_detects_bad_target_value(raw_df, mapping_codes):
    broken = raw_df.copy()
    broken.loc[broken.index[0], TARGET_SOURCE] = "soon"
    report = validate_dataframe(broken, mapping_codes)
    assert report["passed"] is False
    assert report["checks"]["target_values"]["passed"] is False
    assert "soon" in report["checks"]["target_values"]["detail"]


def test_validation_detects_negative_numeric(raw_df, mapping_codes):
    broken = raw_df.copy()
    broken.loc[broken.index[0], "time_in_hospital"] = -3
    report = validate_dataframe(broken, mapping_codes)
    assert report["passed"] is False
    assert report["checks"]["numeric_features"]["passed"] is False
    assert "time_in_hospital" in report["checks"]["numeric_features"]["detail"]


def test_validation_detects_out_of_range_numeric(raw_df, mapping_codes):
    broken = raw_df.copy()
    broken.loc[broken.index[0], "num_lab_procedures"] = 99999
    report = validate_dataframe(broken, mapping_codes)
    assert report["checks"]["numeric_features"]["passed"] is False


def test_validation_detects_non_integer_numeric(raw_df, mapping_codes):
    broken = raw_df.copy()
    broken["time_in_hospital"] = broken["time_in_hospital"].astype(float)
    report = validate_dataframe(broken, mapping_codes)
    assert report["checks"]["numeric_features"]["passed"] is False


def test_validation_detects_unknown_code(raw_df, mapping_codes):
    broken = raw_df.copy()
    broken.loc[broken.index[0], "admission_type_id"] = 99
    report = validate_dataframe(broken, mapping_codes)
    assert report["passed"] is False
    assert report["checks"]["code_columns"]["passed"] is False
    assert "99" in report["checks"]["code_columns"]["detail"]


@pytest.mark.integration
def test_validation_on_real_data():
    from carereturn.config import RAW_DATA_FILE, RAW_MAPPING_FILE

    if not RAW_DATA_FILE.exists():
        pytest.skip("Real raw data not available")
    frame = read_raw(RAW_DATA_FILE)
    mappings = read_code_mappings(RAW_MAPPING_FILE)
    report = validate_dataframe(frame, mappings)
    assert report["passed"] is True, report["checks"]
    assert report["n_rows"] == 101766
    assert report["n_cols"] == 50
    assert report["n_patients"] == 71518
    assert report["n_positive"] == 11357
