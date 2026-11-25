"""Schema and quality validation of the raw encounters data.

Run as ``python -m carereturn.data.validate``: reads the raw CSV, runs every
check, writes ``reports/data_validation.json`` and exits non-zero if any check
fails.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import pandas as pd

from carereturn.config import (
    CODE_COLUMNS,
    DATA_VALIDATION_FILE,
    NUMERIC_FEATURES,
    POSITIVE_LABEL,
    RAW_DATA_FILE,
    RAW_MAPPING_FILE,
    TARGET_SOURCE,
    TARGET_VALUES,
)
from carereturn.data.io import read_code_mappings, read_raw, write_json_report

logger = logging.getLogger(__name__)

#: Exact column contract of the raw UCI file (50 columns).
EXPECTED_RAW_COLUMNS: list[str] = [
    "encounter_id",
    "patient_nbr",
    "race",
    "gender",
    "age",
    "weight",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "payer_code",
    "medical_specialty",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "diag_1",
    "diag_2",
    "diag_3",
    "number_diagnoses",
    "max_glu_serum",
    "A1Cresult",
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
    "change",
    "diabetesMed",
    "readmitted",
]

#: Inclusive plausible [min, max] ranges for the numeric features.
NUMERIC_RANGES: dict[str, tuple[int, int]] = {
    "time_in_hospital": (1, 14),
    "num_lab_procedures": (0, 132),
    "num_procedures": (0, 6),
    "num_medications": (1, 81),
    "number_outpatient": (0, 70),
    "number_emergency": (0, 76),
    "number_inpatient": (0, 21),
    "number_diagnoses": (1, 16),
}


def _check_schema(frame: pd.DataFrame) -> tuple[bool, str]:
    """Check that all expected raw columns are present.

    Args:
        frame: Raw encounters DataFrame.

    Returns:
        A ``(passed, detail)`` tuple describing the schema check.

    """
    expected = set(EXPECTED_RAW_COLUMNS)
    actual = set(frame.columns)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    passed = not missing
    detail = (
        f"{len(actual)} column(s) present; missing={missing or 'none'}; extra={extra or 'none'}"
    )
    return passed, detail


def _check_unique_encounters(frame: pd.DataFrame) -> tuple[bool, str]:
    """Check that ``encounter_id`` identifies stays uniquely.

    Args:
        frame: Raw encounters DataFrame.

    Returns:
        A ``(passed, detail)`` tuple describing the uniqueness check.

    """
    if "encounter_id" not in frame.columns:
        return False, "column encounter_id is missing"
    n_duplicate = int(frame["encounter_id"].duplicated().sum())
    passed = n_duplicate == 0
    return passed, f"{n_duplicate} duplicated encounter_id value(s)"


def _check_target_values(frame: pd.DataFrame) -> tuple[bool, str]:
    """Check that ``readmitted`` only contains the expected labels.

    Args:
        frame: Raw encounters DataFrame.

    Returns:
        A ``(passed, detail)`` tuple describing the target check.

    """
    if TARGET_SOURCE not in frame.columns:
        return False, f"column {TARGET_SOURCE} is missing"
    unexpected = sorted(set(frame[TARGET_SOURCE].dropna().unique()) - TARGET_VALUES)
    passed = not unexpected
    return passed, f"unexpected target value(s): {unexpected or 'none'}"


def _check_numeric_features(frame: pd.DataFrame) -> tuple[bool, str]:
    """Check numeric features are non-negative integers in plausible ranges.

    Args:
        frame: Raw encounters DataFrame.

    Returns:
        A ``(passed, detail)`` tuple describing the numeric check.

    """
    violations: list[str] = []
    for column in NUMERIC_FEATURES:
        if column not in frame.columns:
            violations.append(f"{column}: column missing")
            continue
        series = frame[column]
        if not pd.api.types.is_integer_dtype(series):
            violations.append(f"{column}: dtype {series.dtype} is not integer")
            continue
        low, high = NUMERIC_RANGES[column]
        out_of_range = ((series < low) | (series > high)).sum()
        if out_of_range:
            violations.append(f"{column}: {int(out_of_range)} value(s) outside [{low}, {high}]")
    passed = not violations
    return passed, "; ".join(
        violations
    ) if violations else "all numeric features are integers in range"


def _check_code_columns(
    frame: pd.DataFrame, mapping_codes: dict[str, set[int]]
) -> tuple[bool, str]:
    """Check administrative code columns are known integers from the dictionary.

    Args:
        frame: Raw encounters DataFrame.
        mapping_codes: Valid code sets per column, parsed from IDS_mapping.csv.

    Returns:
        A ``(passed, detail)`` tuple describing the code check.

    """
    violations: list[str] = []
    for column in CODE_COLUMNS:
        if column not in frame.columns:
            violations.append(f"{column}: column missing")
            continue
        series = frame[column]
        if not pd.api.types.is_integer_dtype(series):
            violations.append(f"{column}: dtype {series.dtype} is not integer")
            continue
        valid = mapping_codes.get(column)
        if valid is None:
            violations.append(f"{column}: no mapping section found")
            continue
        unknown = sorted(set(series.dropna().unique()) - valid)
        if unknown:
            violations.append(f"{column}: unknown code(s) {unknown}")
    passed = not violations
    return passed, "; ".join(violations) if violations else "all code columns match the dictionary"


def validate_dataframe(frame: pd.DataFrame, mapping_codes: dict[str, set[int]]) -> dict[str, Any]:
    """Run every validation check and build the report payload.

    Args:
        frame: Raw encounters DataFrame, as produced by
            :func:`carereturn.data.io.read_raw`.
        mapping_codes: Valid code sets per column, parsed from IDS_mapping.csv.

    Returns:
        Report with row/patient/target statistics, per-check results and an
        overall ``passed`` flag.

    """
    checks = {
        "schema_columns": _check_schema(frame),
        "encounter_id_unique": _check_unique_encounters(frame),
        "target_values": _check_target_values(frame),
        "numeric_features": _check_numeric_features(frame),
        "code_columns": _check_code_columns(frame, mapping_codes),
    }
    n_rows = len(frame)
    n_positive = (
        int((frame[TARGET_SOURCE] == POSITIVE_LABEL).sum()) if TARGET_SOURCE in frame.columns else 0
    )
    report: dict[str, Any] = {
        "n_rows": n_rows,
        "n_cols": int(frame.shape[1]),
        "n_patients": int(frame["patient_nbr"].nunique()) if "patient_nbr" in frame.columns else 0,
        "n_positive": n_positive,
        "positive_rate": round(n_positive / n_rows, 6) if n_rows else 0.0,
        "checks": {
            name: {"passed": passed, "detail": detail} for name, (passed, detail) in checks.items()
        },
        "passed": all(passed for passed, _ in checks.values()),
    }
    return report


def main() -> int:
    """Validate the raw data, write the report and return an exit code.

    Returns:
        ``0`` when every check passes, ``1`` otherwise.

    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    frame = read_raw(RAW_DATA_FILE)
    mapping_codes = read_code_mappings(RAW_MAPPING_FILE)
    report = validate_dataframe(frame, mapping_codes)
    write_json_report(report, DATA_VALIDATION_FILE)
    for name, check in report["checks"].items():
        log = logger.info if check["passed"] else logger.error
        log("check %s -> %s: %s", name, "PASS" if check["passed"] else "FAIL", check["detail"])
    logger.info(
        "Raw data: rows=%d cols=%d patients=%d positives=%d rate=%.4f",
        report["n_rows"],
        report["n_cols"],
        report["n_patients"],
        report["n_positive"],
        report["positive_rate"],
    )
    if not report["passed"]:
        logger.error("Data validation FAILED, see %s", DATA_VALIDATION_FILE)
        return 1
    logger.info("Data validation passed, report written to %s", DATA_VALIDATION_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
