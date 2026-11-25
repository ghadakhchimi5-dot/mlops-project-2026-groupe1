"""Shared synthetic-data builders for the model tests (contains no tests).

Builds small DataFrames that follow the split contract of
``data/processed/{train,val,test}.parquet``: columns ``ID_COLUMNS + FEATURES
+ TARGET`` with int numeric features, mixed-type categorical features (int
codes, strings, the literal ``"None"``, and NaNs) and a binary target with a
learnable logistic signal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from carereturn.config import FEATURES, ID_COLUMNS, TARGET

AGE_BANDS = ["[30-60)", "[60-70)", "[70-80)", "[80-90)"]
AGE_EFFECTS = {"[30-60)": 0.0, "[60-70)": 0.4, "[70-80)": 0.8, "[80-90)": 1.2}
ADMISSION_TYPES = [1, 2, 3, 6, 8]  # 1=Emergency (+effect), 2=Urgent, 3=Elective
ADMISSION_EFFECTS = {1: 0.5, 2: 0.2, 3: 0.0, 6: 0.0, 8: 0.0}
A1C_VALUES = ["None", "Norm", ">7"]  # literal "None" = test not done
INSULIN_VALUES = ["No", "Up", "Down"]  # "Steady" only injected in the test split
SPLIT_FRACTIONS = {"train": 0.6, "val": 0.2, "test": 0.2}


def build_dataset(n_rows: int = 600, seed: int = 0) -> pd.DataFrame:
    """Build a synthetic dataset following the model input contract.

    Roughly 15% prevalence with a logistic signal driven by
    ``number_inpatient``, ``num_medications``, the age band and the admission
    type. The last 20% of rows (the test split) receives categories unseen
    during training (``insulin="Steady"``, ``A1Cresult=">8"``) plus NaNs, to
    exercise unseen-category handling.

    Args:
        n_rows: Number of encounter rows to generate.
        seed: Numpy seed for reproducibility.

    Returns:
        DataFrame with ``ID_COLUMNS + FEATURES + TARGET``; ``TARGET`` is int 0/1.
    """
    rng = np.random.default_rng(seed)

    age = rng.choice(AGE_BANDS, size=n_rows)
    admission_type_id = rng.choice(ADMISSION_TYPES, size=n_rows)
    number_inpatient = rng.poisson(0.6, size=n_rows)
    num_medications = rng.integers(1, 30, size=n_rows)

    logits = np.full(n_rows, -3.8, dtype=float)
    logits += 1.2 * number_inpatient
    logits += 0.04 * num_medications
    logits += np.vectorize(AGE_EFFECTS.get)(age)
    logits += np.vectorize(ADMISSION_EFFECTS.get)(admission_type_id)
    target = rng.binomial(1, 1.0 / (1.0 + np.exp(-logits)))

    frame = pd.DataFrame(
        {
            "encounter_id": np.arange(100_000, 100_000 + n_rows),
            "patient_nbr": np.arange(100_000, 100_000 + n_rows) // 2,
            "time_in_hospital": rng.integers(1, 14, size=n_rows),
            "num_lab_procedures": rng.integers(1, 100, size=n_rows),
            "num_procedures": rng.integers(0, 6, size=n_rows),
            "num_medications": num_medications,
            "number_outpatient": rng.poisson(0.4, size=n_rows),
            "number_emergency": rng.poisson(0.3, size=n_rows),
            "number_inpatient": number_inpatient,
            "number_diagnoses": rng.integers(1, 10, size=n_rows),
            "age": age,
            "admission_type_id": admission_type_id,
            "admission_source_id": rng.choice([1, 7, 9], size=n_rows),
            "discharge_disposition_id": rng.choice([1, 2, 3, 6], size=n_rows),
            "A1Cresult": rng.choice(A1C_VALUES, size=n_rows, p=[0.5, 0.3, 0.2]),
            "max_glu_serum": rng.choice(["None", "Norm"], size=n_rows, p=[0.8, 0.2]),
            "insulin": rng.choice(INSULIN_VALUES, size=n_rows),
            "change": rng.choice(["No", "Ch"], size=n_rows),
            "diabetesMed": rng.choice(["Yes", "No"], size=n_rows, p=[0.7, 0.3]),
            "target": target.astype(int),
        }
    )

    # Inject NaNs in numerics (~3%) and categoricals (~5%).
    numeric_columns = set(frame.select_dtypes(include=[np.number]).columns)
    for column in frame.columns:
        if column in (*ID_COLUMNS, TARGET):
            continue
        rate = 0.03 if column in numeric_columns else 0.05
        mask = rng.random(n_rows) < rate
        frame.loc[mask, column] = np.nan

    # Unseen-at-training categories, placed only in the test slice.
    test_start = int(n_rows * (SPLIT_FRACTIONS["train"] + SPLIT_FRACTIONS["val"]))
    frame.loc[test_start:, "insulin"] = rng.choice(["Steady"], size=n_rows - test_start)
    frame.loc[test_start:, "A1Cresult"] = rng.choice([">8", "None"], size=n_rows - test_start)
    return frame


def split_frames(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split a dataset positionally into train/val/test (60/20/20).

    Args:
        frame: Dataset built by :func:`build_dataset`.

    Returns:
        Dict with keys ``train``, ``val``, ``test``.
    """
    bounds = [0.0, SPLIT_FRACTIONS["train"], SPLIT_FRACTIONS["train"] + SPLIT_FRACTIONS["val"], 1.0]
    cuts = [int(fraction * len(frame)) for fraction in bounds]
    return {
        "train": frame.iloc[cuts[0] : cuts[1]].reset_index(drop=True),
        "val": frame.iloc[cuts[1] : cuts[2]].reset_index(drop=True),
        "test": frame.iloc[cuts[2] : cuts[3]].reset_index(drop=True),
    }


def write_splits(frame: pd.DataFrame, directory: Path) -> dict[str, Path]:
    """Write the three splits as parquet files named after each split.

    Args:
        frame: Dataset built by :func:`build_dataset`.
        directory: Destination directory (created if needed).

    Returns:
        Mapping from split name to the written parquet path.
    """
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, split in split_frames(frame).items():
        paths[name] = directory / f"{name}.parquet"
        split.to_parquet(paths[name], index=False)
    return paths


def assert_contract_columns(frame: pd.DataFrame) -> None:
    """Check that a frame holds exactly the contract columns of a split file.

    Args:
        frame: Dataset to check.

    Raises:
        AssertionError: If columns differ from ``ID_COLUMNS + FEATURES + TARGET``.
    """
    expected = [*ID_COLUMNS, *FEATURES, TARGET]
    assert list(frame.columns) == expected
