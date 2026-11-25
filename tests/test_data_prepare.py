"""Tests for carereturn.data.prepare: exclusions, target mapping, columns."""

from __future__ import annotations

import pandas as pd
import pytest

from carereturn.config import EXCLUDED_DISCHARGE_CODES, FEATURES, ID_COLUMNS, TARGET, TARGET_SOURCE
from carereturn.data.prepare import prepare_cohort


def test_prepare_excludes_death_and_hospice(raw_df):
    cohort = prepare_cohort(raw_df)
    excluded_mask = raw_df["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_CODES)
    assert 11 in EXCLUDED_DISCHARGE_CODES and 13 in EXCLUDED_DISCHARGE_CODES
    assert len(cohort) == int((~excluded_mask).sum())
    assert not cohort["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_CODES).any()


def test_prepare_target_mapping(raw_df):
    cohort = prepare_cohort(raw_df)
    expected = (raw_df[TARGET_SOURCE] == "<30").astype(int)
    expected = expected.loc[~raw_df["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_CODES)]
    expected = expected.reset_index(drop=True)
    assert cohort[TARGET].tolist() == expected.tolist()
    assert set(cohort[TARGET].unique()) <= {0, 1}
    assert pd.api.types.is_integer_dtype(cohort[TARGET])


def test_prepare_keeps_only_ids_features_target(cohort_df):
    assert list(cohort_df.columns) == ID_COLUMNS + FEATURES + [TARGET]


def test_prepare_positive_rows_match_readmitted_lt30(cohort_df):
    assert (cohort_df[TARGET] == 1).sum() > 0
    assert (cohort_df[TARGET] == 0).sum() > 0


def test_prepare_raises_on_missing_columns():
    frame = pd.DataFrame({"encounter_id": [1], "readmitted": ["NO"]})
    with pytest.raises(ValueError, match="missing required column"):
        prepare_cohort(frame)
