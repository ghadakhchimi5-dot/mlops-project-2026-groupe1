"""Load the trained sklearn pipeline and its metadata for serving.

The loader resolves artifact paths from the `MODEL_PATH` / `MODEL_META_PATH`
environment variables, falling back to the shared defaults in
`carereturn.config` (`models/model.joblib`, `models/model_meta.json`).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from carereturn import config

logger = logging.getLogger(__name__)


class ModelNotFoundError(RuntimeError):
    """Raised when model artifacts are missing, unreadable or malformed."""


@dataclass
class LoadedModel:
    """A fitted pipeline plus its serving metadata."""

    pipeline: Any
    meta: dict[str, Any]

    @property
    def model_name(self) -> str:
        """Name recorded in the model metadata."""
        return str(self.meta["model_name"])

    @property
    def model_version(self) -> str:
        """Version recorded in the model metadata."""
        return str(self.meta["model_version"])

    @property
    def threshold(self) -> float:
        """Decision threshold used to flag high-risk encounters."""
        return float(self.meta["threshold"])

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Return positive-class probabilities for a feature frame.

        Args:
            frame: DataFrame with (at least) the `config.FEATURES` columns.

        Returns:
            Array of float probabilities in `[0, 1]`, one per row.

        Raises:
            ValueError: If required feature columns are missing.

        """
        missing = [col for col in config.FEATURES if col not in frame.columns]
        if missing:
            msg = f"Input frame is missing required feature columns: {missing}"
            raise ValueError(msg)
        ordered = frame.reindex(columns=config.FEATURES)
        proba = self.pipeline.predict_proba(ordered)
        return np.asarray(proba, dtype=float)[:, 1]


class ModelLoader:
    """Load a pipeline + metadata pair from disk, with clear failure messages."""

    def __init__(self, model_path: str | Path, meta_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self.meta_path = Path(meta_path)

    def load(self) -> LoadedModel:
        """Load and validate both artifacts.

        Raises:
            ModelNotFoundError: If either file is missing, the pipeline cannot
                be unpickled, or the metadata JSON is malformed/incomplete.

        """
        missing = [str(p) for p in (self.model_path, self.meta_path) if not p.is_file()]
        if missing:
            msg = (
                f"Model artifact(s) not found: {', '.join(missing)}. "
                "Train the model first or point MODEL_PATH / MODEL_META_PATH "
                "at existing artifacts."
            )
            raise ModelNotFoundError(msg)

        try:
            pipeline = joblib.load(self.model_path)
        except Exception as exc:
            msg = f"Failed to load model pipeline from {self.model_path}: {exc!r}"
            raise ModelNotFoundError(msg) from exc

        try:
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            msg = f"Failed to read model metadata from {self.meta_path}: {exc}"
            raise ModelNotFoundError(msg) from exc

        if not isinstance(meta, dict):
            msg = f"Model metadata at {self.meta_path} must be a JSON object."
            raise ModelNotFoundError(msg)
        missing_keys = [
            key for key in ("model_name", "model_version", "threshold") if key not in meta
        ]
        if missing_keys:
            msg = f"Model metadata at {self.meta_path} is missing keys: {missing_keys}"
            raise ModelNotFoundError(msg)

        if meta.get("features") != list(config.FEATURES):
            logger.warning(
                "Model metadata 'features' differ from config.FEATURES "
                "(meta=%s); the pipeline will still receive config.FEATURES columns.",
                meta.get("features"),
            )

        logger.info(
            "Loaded model '%s' v%s (threshold=%.3f) from %s",
            meta["model_name"],
            meta["model_version"],
            float(meta["threshold"]),
            self.model_path,
        )
        return LoadedModel(pipeline=pipeline, meta=meta)


def loader_from_env() -> ModelLoader:
    """Build a `ModelLoader` from MODEL_PATH / MODEL_META_PATH (config defaults)."""
    model_path = os.environ.get("MODEL_PATH", str(config.MODEL_FILE))
    meta_path = os.environ.get("MODEL_META_PATH", str(config.MODEL_META_FILE))
    return ModelLoader(model_path, meta_path)
