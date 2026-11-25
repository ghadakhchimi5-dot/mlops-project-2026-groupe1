"""Tests for carereturn.data.split: groups, proportions, determinism."""

from __future__ import annotations

import pandas as pd
import pytest

from carereturn.config import TARGET
from carereturn.data.split import (
    DEFAULT_PROPORTIONS,
    assert_no_patient_overlap,
    group_split,
    split_summary,
)

ROW_TOLERANCE = 0.15
RATE_TOLERANCE = 0.15


def test_group_split_has_no_patient_overlap(cohort_df):
    splits = group_split(cohort_df, seed=42)
    overlap = assert_no_patient_overlap(splits)
    assert all(count == 0 for count in overlap.values())


def test_group_split_covers_all_rows_once(cohort_df):
    splits = group_split(cohort_df, seed=42)
    total = sum(len(part) for part in splits.values())
    assert total == len(cohort_df)
    encounter_ids = pd.concat([part["encounter_id"] for part in splits.values()], ignore_index=True)
    assert encounter_ids.nunique() == cohort_df["encounter_id"].nunique()


def test_group_split_approximate_proportions(cohort_df):
    splits = group_split(cohort_df, seed=42)
    total = len(cohort_df)
    for name, expected in DEFAULT_PROPORTIONS.items():
        fraction = len(splits[name]) / total
        assert abs(fraction - expected) <= ROW_TOLERANCE, f"{name}: {fraction} vs {expected}"


def test_group_split_approximately_stratified(cohort_df):
    splits = group_split(cohort_df, seed=42)
    overall_rate = cohort_df[TARGET].mean()
    for name, part in splits.items():
        assert abs(part[TARGET].mean() - overall_rate) <= RATE_TOLERANCE, name


def test_group_split_is_deterministic(cohort_df):
    first = group_split(cohort_df, seed=42)
    second = group_split(cohort_df, seed=42)
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])


def test_assert_no_overlap_raises_on_leak():
    frame = pd.DataFrame({"patient_nbr": [1, 2, 3], "encounter_id": [10, 11, 12]})
    with pytest.raises(ValueError, match="leakage"):
        assert_no_patient_overlap({"train": frame, "val": frame.iloc[:2]})


def test_invalid_fractions_raise(cohort_df):
    with pytest.raises(ValueError, match="sum to 1"):
        group_split(cohort_df, seed=42, proportions={"train": 0.5, "val": 0.2, "test": 0.2})
    with pytest.raises(ValueError, match="positive"):
        group_split(cohort_df, seed=42, proportions={"train": -1, "val": 1.0, "test": 1.0})


def test_split_summary_payload(cohort_df):
    splits = group_split(cohort_df, seed=42)
    overlap = assert_no_patient_overlap(splits)
    summary = split_summary(splits, seed=42, proportions=DEFAULT_PROPORTIONS, overlap=overlap)
    assert summary["seed"] == 42
    assert set(summary["splits"]) == {"train", "val", "test"}
    for stats in summary["splits"].values():
        assert set(stats) == {"rows", "patients", "positive_rate", "row_fraction"}
        assert stats["rows"] > 0 and stats["patients"] > 0
        assert 0.0 <= stats["positive_rate"] <= 1.0
    assert all(count == 0 for count in summary["patient_overlap"].values())
