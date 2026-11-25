"""Group-wise train/val/test split by patient.

Run as ``python -m carereturn.data.split``: reads ``data/interim/cohort.parquet``,
splits stays so that every patient belongs to exactly one partition (with
approximately stratified classes), writes the three parquet files declared in
``config.SPLIT_FILES`` and ``reports/split_summary.json``.
"""

from __future__ import annotations

import itertools
import logging
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from carereturn.config import COHORT_FILE, SPLIT_FILES, SPLIT_SUMMARY_FILE, TARGET
from carereturn.data.io import read_parquet, write_json_report, write_parquet

logger = logging.getLogger(__name__)

#: Default 60/20/20 proportions, overridden by ``split`` in params.yaml.
DEFAULT_PROPORTIONS: dict[str, float] = {"train": 0.6, "val": 0.2, "test": 0.2}

#: Allowed deviation (in fraction points) before split proportions are flagged.
FRACTION_TOLERANCE = 0.05

#: Allowed deviation between a split positive rate and the overall rate.
RATE_TOLERANCE = 0.02

#: Patient-overlap pair keys, in report order.
_OVERLAP_PAIRS = [("train", "val"), ("train", "test"), ("val", "test")]


def _validate_fractions(fractions: dict[str, float]) -> None:
    """Check that split fractions are well-formed.

    Args:
        fractions: Requested ``train``/``val``/``test`` fractions.

    Raises:
        ValueError: If keys are missing, non-positive, or do not sum to 1.

    """
    for name in ("train", "val", "test"):
        value = fractions.get(name)
        if value is None or value <= 0:
            raise ValueError(f"Split fraction '{name}' must be a positive float, got {value}")
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1, got {total}")


def assert_no_patient_overlap(splits: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Assert that patients are disjoint across partitions.

    Args:
        splits: Mapping from split name to DataFrame with a ``patient_nbr`` column.

    Returns:
        Overlap counts per pair (all zeros when the assertion holds).

    Raises:
        ValueError: If any two partitions share at least one patient.

    """
    patients = {name: set(frame["patient_nbr"]) for name, frame in splits.items()}
    pairs = [
        (left, right) for left, right in _OVERLAP_PAIRS if left in patients and right in patients
    ]
    if not pairs:
        pairs = list(itertools.combinations(sorted(patients), 2))
    counts = {f"{left}_{right}": len(patients[left] & patients[right]) for left, right in pairs}
    leaked = {pair: count for pair, count in counts.items() if count}
    if leaked:
        raise ValueError(f"Patient leakage detected across splits: {leaked}")
    return counts


def group_split(
    cohort: pd.DataFrame,
    seed: int,
    proportions: dict[str, float] | None = None,
) -> dict[str, pd.DataFrame]:
    """Split stays into train/val/test without sharing any patient.

    A :class:`~sklearn.model_selection.StratifiedGroupKFold` over
    ``patient_nbr`` groups produces approximately stratified folds; folds are
    assigned to ``test`` and ``val`` and the remainder becomes ``train``,
    matching the requested proportions.

    Args:
        cohort: Cohort DataFrame with ``patient_nbr`` and ``target`` columns.
        seed: Random seed for reproducibility.
        proportions: Requested ``train``/``val``/``test`` fractions; defaults
            to ``DEFAULT_PROPORTIONS``.

    Returns:
        Mapping from split name to the corresponding DataFrame.

    Raises:
        ValueError: If fractions are invalid or patients leak across splits.

    """
    fractions = dict(proportions or DEFAULT_PROPORTIONS)
    _validate_fractions(fractions)
    n_splits = max(2, round(1.0 / fractions["val"]))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(cohort, cohort[TARGET], cohort["patient_nbr"]))
    indices = {
        "test": folds[0][1],
        "val": folds[1][1],
        "train": np.concatenate([fold[1] for fold in folds[2:]]),
    }
    splits = {
        name: cohort.iloc[np.sort(index)].reset_index(drop=True) for name, index in indices.items()
    }
    assert_no_patient_overlap(splits)
    _log_and_warn_balance(splits, fractions)
    return splits


def _log_and_warn_balance(splits: dict[str, pd.DataFrame], fractions: dict[str, float]) -> None:
    """Log split sizes and warn when proportions or class balance drift.

    Args:
        splits: Mapping from split name to DataFrame.
        fractions: Requested ``train``/``val``/``test`` fractions.

    """
    total_rows = sum(len(frame) for frame in splits.values())
    overall_rate = float(pd.concat(splits.values(), ignore_index=True)[TARGET].mean())
    for name, frame in splits.items():
        row_fraction = len(frame) / total_rows
        rate = float(frame[TARGET].mean())
        logger.info(
            "Split %s: rows=%d (%.1f%%, target %.1f%%) patients=%d positive_rate=%.4f",
            name,
            len(frame),
            100 * row_fraction,
            100 * fractions[name],
            frame["patient_nbr"].nunique(),
            rate,
        )
        if abs(row_fraction - fractions[name]) > FRACTION_TOLERANCE:
            logger.warning(
                "Split %s row fraction %.3f deviates from requested %.3f",
                name,
                row_fraction,
                fractions[name],
            )
        if abs(rate - overall_rate) > RATE_TOLERANCE:
            logger.warning(
                "Split %s positive rate %.4f deviates from overall %.4f", name, rate, overall_rate
            )


def split_summary(
    splits: dict[str, pd.DataFrame],
    seed: int,
    proportions: dict[str, float],
    overlap: dict[str, int],
) -> dict[str, Any]:
    """Build the JSON-serializable split summary report.

    Args:
        splits: Mapping from split name to DataFrame.
        seed: Random seed used for the split.
        proportions: Requested ``train``/``val``/``test`` fractions.
        overlap: Patient-overlap counts from :func:`assert_no_patient_overlap`.

    Returns:
        Report payload with per-split rows/patients/positive rate and overlaps.

    """
    total_rows = sum(len(frame) for frame in splits.values())
    return {
        "seed": int(seed),
        "proportions": {name: float(value) for name, value in proportions.items()},
        "splits": {
            name: {
                "rows": int(len(frame)),
                "patients": int(frame["patient_nbr"].nunique()),
                "positive_rate": round(float(frame[TARGET].mean()), 6),
                "row_fraction": round(len(frame) / total_rows, 4),
            }
            for name, frame in splits.items()
        },
        "patient_overlap": {
            f"{left}_{right}": int(overlap[f"{left}_{right}"]) for left, right in _OVERLAP_PAIRS
        },
    }


def main() -> int:
    """Run the group split and write parquet parts plus the summary report.

    Returns:
        ``0`` on success, ``1`` when the split cannot be produced.

    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    from carereturn.config import load_params

    params = load_params()
    seed = int(params["seed"])
    fractions = {
        name: float(value) for name, value in (params.get("split") or DEFAULT_PROPORTIONS).items()
    }
    cohort = read_parquet(COHORT_FILE)
    splits = group_split(cohort, seed, fractions)
    overlap = assert_no_patient_overlap(splits)
    for name, path in SPLIT_FILES.items():
        write_parquet(splits[name], path)
    summary = split_summary(splits, seed, fractions, overlap)
    write_json_report(summary, SPLIT_SUMMARY_FILE)
    logger.info("Split summary written to %s (seed=%d)", SPLIT_SUMMARY_FILE, seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
