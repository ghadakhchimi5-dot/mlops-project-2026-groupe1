"""Tests for carereturn.data.io: NA handling, parquet and JSON helpers."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from carereturn.data.io import (
    read_code_mappings,
    read_json_report,
    read_parquet,
    read_raw,
    write_json_report,
    write_parquet,
)


def test_read_raw_converts_question_mark_to_nan(raw_df):
    assert raw_df["race"].isna().any()
    assert raw_df["weight"].isna().any()
    assert raw_df["diag_3"].isna().all()


def test_read_raw_preserves_none_string(raw_df):
    assert raw_df["A1Cresult"].eq("None").any()
    assert raw_df["max_glu_serum"].eq("None").any()
    # The only NA tokens allowed are "?": lab results are never NaN unless "?"
    assert not raw_df["A1Cresult"].isna().any()


def test_read_raw_on_real_header_columns(synthetic_raw_df, tmp_path):
    path = tmp_path / "raw.csv"
    synthetic_raw_df.to_csv(path, index=False)
    frame = read_raw(path)
    assert list(frame.columns) == list(synthetic_raw_df.columns)


def test_parquet_roundtrip(tmp_path):
    frame = pd.DataFrame(
        {
            "encounter_id": [1, 2, 3],
            "patient_nbr": [10, 10, 20],
            "label": ["a", None, "c"],
            "value": [1.5, 2.5, 3.5],
        }
    )
    path = write_parquet(frame, tmp_path / "nested" / "frame.parquet")
    assert path.exists()
    result = read_parquet(path)
    pd.testing.assert_frame_equal(result, frame)


def test_json_report_roundtrip(tmp_path):
    payload = {"n_rows": 3, "checks": {"a": {"passed": True, "detail": "ok"}}, "passed": True}
    path = write_json_report(payload, tmp_path / "nested" / "report.json")
    with open(path, encoding="utf-8") as handle:
        assert json.load(handle) == payload
    assert read_json_report(path) == payload


def test_read_code_mappings_parses_sections(mapping_codes):
    assert mapping_codes["admission_type_id"] == set(range(1, 9))
    assert mapping_codes["discharge_disposition_id"] == set(range(1, 31))
    assert mapping_codes["admission_source_id"] == set(range(1, 27))


def test_read_code_mappings_skips_bad_rows(tmp_path):
    path = tmp_path / "bad_mapping.csv"
    path.write_text(
        "admission_type_id,description\n1,Emergency\nnotanint,oops\n,\n",
        encoding="utf-8",
    )
    assert read_code_mappings(path) == {"admission_type_id": {1}}


@pytest.mark.integration
def test_read_raw_real_data():
    from carereturn.config import RAW_DATA_FILE

    if not RAW_DATA_FILE.exists():
        pytest.skip("Real raw data not available")
    frame = read_raw(RAW_DATA_FILE)
    assert frame.shape == (101766, 50)
    assert frame["A1Cresult"].eq("None").any()
