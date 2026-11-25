"""Reading and writing helpers shared by every data-pipeline stage.

The raw UCI CSV uses the literal token ``?`` for missing values and the literal
string ``None`` inside ``A1Cresult`` / ``max_glu_serum`` to mean "test not
done". Because ``None`` is part of pandas' default NA-token list, the raw
reader disables the defaults and only accepts ``?`` as NA.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from carereturn.config import MISSING_TOKEN

logger = logging.getLogger(__name__)


def read_raw(path: Path | str) -> pd.DataFrame:
    """Read the raw encounters CSV with project-specific NA handling.

    ``?`` is converted to ``NaN`` while every other default pandas NA token is
    disabled, so the string ``"None"`` in ``A1Cresult`` and ``max_glu_serum``
    is preserved as a category meaning "test not done".

    Args:
        path: Path to the raw CSV file.

    Returns:
        The raw encounters as a DataFrame.

    """
    frame = pd.read_csv(
        Path(path), keep_default_na=False, na_values=[MISSING_TOKEN], low_memory=False
    )
    logger.info("Read raw data from %s (%d rows, %d columns)", path, len(frame), frame.shape[1])
    return frame


def read_code_mappings(path: Path | str) -> dict[str, set[int]]:
    """Parse the IDS_mapping dictionary into per-column sets of valid codes.

    The file contains three sections (``admission_type_id``,
    ``discharge_disposition_id``, ``admission_source_id``), each introduced by
    a header row ``<column>,description`` and separated by blank lines.

    Args:
        path: Path to ``IDS_mapping.csv``.

    Returns:
        Mapping from column name to the set of integer codes it may take.

    """
    sections: dict[str, set[int]] = {}
    current: str | None = None
    with open(Path(path), encoding="utf-8") as handle:
        for row in csv.reader(handle):
            cells = [cell.strip() for cell in row]
            if not cells or all(cell == "" for cell in cells):
                current = None
                continue
            if cells[0].endswith("_id"):
                current = cells[0]
                sections.setdefault(current, set())
                continue
            if current is None or not cells[0]:
                continue
            try:
                sections[current].add(int(cells[0]))
            except ValueError:
                logger.debug("Ignoring non-integer mapping row in %s: %s", path, row)
    logger.info("Parsed code mappings from %s: %s", path, {k: len(v) for k, v in sections.items()})
    return sections


def write_parquet(frame: pd.DataFrame, path: Path | str) -> Path:
    """Write a DataFrame to parquet, creating parent directories as needed.

    Args:
        frame: DataFrame to persist.
        path: Destination parquet file.

    Returns:
        The path that was written.

    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
    logger.info(
        "Wrote parquet to %s (%d rows, %d columns)", destination, len(frame), frame.shape[1]
    )
    return destination


def read_parquet(path: Path | str) -> pd.DataFrame:
    """Read a parquet file into a DataFrame.

    Args:
        path: Parquet file to read.

    Returns:
        The DataFrame stored at ``path``.

    """
    frame = pd.read_parquet(Path(path))
    logger.info("Read parquet from %s (%d rows, %d columns)", path, len(frame), frame.shape[1])
    return frame


def write_json_report(payload: dict[str, Any], path: Path | str) -> Path:
    """Serialize a report dictionary to a pretty-printed JSON file.

    Args:
        payload: JSON-serializable report content.
        path: Destination JSON file.

    Returns:
        The path that was written.

    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    logger.info("Wrote JSON report to %s", destination)
    return destination


def read_json_report(path: Path | str) -> dict[str, Any]:
    """Load a JSON report written by :func:`write_json_report`.

    Args:
        path: JSON file to read.

    Returns:
        The deserialized report content.

    """
    with open(Path(path), encoding="utf-8") as handle:
        payload = json.load(handle)
    logger.info("Read JSON report from %s", path)
    return payload
