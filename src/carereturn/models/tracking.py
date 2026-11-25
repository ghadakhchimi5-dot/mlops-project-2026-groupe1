"""Thin helpers shared by the train and evaluate stages.

Keeps MLflow experiment setup, provenance capture (git commit) and data
checksums in one place so the stage modules stay readable.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from carereturn import config

LOGGER = logging.getLogger(__name__)


def setup_mlflow(tracking_uri: str, experiment_name: str) -> str:
    """Point MLflow at ``tracking_uri`` and select (or create) the experiment.

    Args:
        tracking_uri: MLflow tracking URI (sqlite file or http server).
        experiment_name: Experiment to use for subsequent runs.

    Returns:
        The experiment id.

    """
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    return experiment.experiment_id


def get_git_commit(repo: Path | None = None) -> str | None:
    """Return the current git commit SHA, or None when git is unavailable.

    Args:
        repo: Repository root (defaults to the project root).

    Returns:
        Full commit SHA, or None if the subprocess fails for any reason.

    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo or config.PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        LOGGER.warning("Could not read git commit: %s", exc)
        return None
    return result.stdout.strip() or None


def file_md5(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute the MD5 hex digest of a file, chunk by chunk.

    Args:
        path: File to hash.
        chunk_size: Read chunk size in bytes.

    Returns:
        The MD5 digest as a lowercase hex string.

    Raises:
        FileNotFoundError: If ``path`` does not exist.

    """
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_params(paths: dict[str, Path]) -> dict[str, str]:
    """Build MLflow params documenting data file checksums and git commit.

    Missing files are reported as ``"absent"`` so tracking never crashes on a
    partially produced dataset.

    Args:
        paths: Mapping from param name suffix (e.g. ``"train"``) to file path.

    Returns:
        Params dict such as ``{"data_md5_train": ..., "git_commit": ...}``.

    """
    params: dict[str, str] = {"git_commit": get_git_commit() or "unknown"}
    for name, path in paths.items():
        try:
            params[f"data_md5_{name}"] = file_md5(path)
        except OSError as exc:
            LOGGER.warning("Could not hash %s: %s", path, exc)
            params[f"data_md5_{name}"] = "absent"
    return params


def log_figure(path: Path, subdir: str | None = None) -> None:
    """Log a single figure file as an MLflow artifact.

    Args:
        path: Local file to log.
        subdir: Optional artifact subdirectory inside the run.

    """
    mlflow.log_artifact(str(path), artifact_path=subdir)


def make_client(tracking_uri: str) -> MlflowClient:
    """Create an MLflow client bound to ``tracking_uri``.

    Args:
        tracking_uri: MLflow tracking URI.

    Returns:
        A configured :class:`mlflow.tracking.MlflowClient`.

    """
    return MlflowClient(tracking_uri=tracking_uri)
