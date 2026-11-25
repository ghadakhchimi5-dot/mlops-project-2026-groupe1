"""Cohort preparation: exclude death/hospice discharges and map the target.

Run as ``python -m carereturn.data.prepare``: reads the raw CSV, builds the
analysis cohort and writes ``data/interim/cohort.parquet``.
"""

from __future__ import annotations

import logging
import sys

import pandas as pd

from carereturn.config import (
    COHORT_FILE,
    DATA_VALIDATION_FILE,
    EXCLUDED_DISCHARGE_CODES,
    FEATURES,
    ID_COLUMNS,
    POSITIVE_LABEL,
    RAW_DATA_FILE,
    TARGET,
    TARGET_SOURCE,
)
from carereturn.data.io import read_json_report, read_raw, write_parquet

logger = logging.getLogger(__name__)


def prepare_cohort(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the analysis cohort from raw encounters.

    Steps: drop stays whose discharge disposition means death or hospice
    (codes in ``EXCLUDED_DISCHARGE_CODES``, since those patients cannot be
    readmitted), map ``readmitted == "<30"`` to the binary ``target`` column,
    and keep only identifiers, features and target.

    Args:
        frame: Raw encounters DataFrame, as produced by
            :func:`carereturn.data.io.read_raw`.

    Returns:
        The cohort DataFrame with columns ``ID_COLUMNS + FEATURES + [TARGET]``.

    Raises:
        ValueError: If a required column is missing from the input.

    """
    keep = ID_COLUMNS + FEATURES + [TARGET]
    missing = [column for column in keep if column != TARGET and column not in frame.columns]
    if missing:
        raise ValueError(f"Input frame is missing required column(s): {missing}")

    initial_rows = len(frame)
    excluded = frame["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_CODES)
    cohort = frame.loc[~excluded].copy()
    cohort[TARGET] = (cohort[TARGET_SOURCE] == POSITIVE_LABEL).astype(int)
    cohort = cohort[keep].reset_index(drop=True)
    logger.info(
        "Cohort: %d -> %d rows (%d excluded via discharge codes %s), %d positive target(s)",
        initial_rows,
        len(cohort),
        int(excluded.sum()),
        sorted(EXCLUDED_DISCHARGE_CODES),
        int(cohort[TARGET].sum()),
    )
    return cohort


def main() -> int:
    """Prepare the cohort and write it to ``data/interim/cohort.parquet``.

    Returns:
        ``0`` on success, ``1`` when the upstream validation failed.

    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if DATA_VALIDATION_FILE.exists():
        report = read_json_report(DATA_VALIDATION_FILE)
        if not report.get("passed", False):
            logger.error("Upstream data validation failed; aborting cohort preparation.")
            return 1
    else:
        logger.warning(
            "No validation report at %s; proceeding without the gate.", DATA_VALIDATION_FILE
        )
    cohort = prepare_cohort(read_raw(RAW_DATA_FILE))
    write_parquet(cohort, COHORT_FILE)
    logger.info("Cohort written to %s", COHORT_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
