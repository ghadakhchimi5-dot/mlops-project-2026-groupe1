"""FastAPI application exposing the CareReturn readmission risk model.

Run locally with:
    uvicorn carereturn.api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from carereturn import config
from carereturn.api.model_loader import (
    LoadedModel,
    ModelNotFoundError,
    loader_from_env,
)
from carereturn.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    Encounter,
    PredictionResponse,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once at startup; keep serving (503 on /health) if it fails."""
    try:
        app.state.model = loader_from_env().load()
    except ModelNotFoundError as exc:
        app.state.model = None
        logger.error("Model failed to load, API will answer 503: %s", exc)
    yield
    app.state.model = None


app = FastAPI(
    title="CareReturn Readmission API",
    description=(
        "Risk scoring of 30-day hospital readmission for diabetic patients, "
        "to prioritise post-discharge follow-up."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_request_latency(request: Request, call_next):  # noqa: ANN201
    """Log every request's path, status and latency in milliseconds."""
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000
    response.headers["x-process-time-ms"] = f"{latency_ms:.2f}"
    logger.info(
        "%s %s -> %s (%.2f ms)",
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
    )
    return response


def _require_model(request: Request) -> LoadedModel:
    """Return the loaded model or raise 503 if it is unavailable."""
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Check server logs and the "
            "MODEL_PATH / MODEL_META_PATH configuration.",
        )
    return model


def encounters_to_frame(encounters: list[Encounter]) -> pd.DataFrame:
    """Convert validated encounters to a DataFrame with exactly config.FEATURES."""
    rows = [encounter.model_dump() for encounter in encounters]
    return pd.DataFrame(rows).reindex(columns=config.FEATURES)


def _prediction(model: LoadedModel, risk_score: float) -> PredictionResponse:
    """Build a single PredictionResponse from a score and the model metadata."""
    return PredictionResponse(
        risk_score=float(risk_score),
        flagged=bool(risk_score >= model.threshold),
        threshold=model.threshold,
        model_name=model.model_name,
        model_version=model.model_version,
    )


@app.get("/health")
def health(request: Request) -> JSONResponse:
    """Liveness/readiness probe: 200 when the model is loaded, 503 otherwise."""
    model = getattr(request.app.state, "model", None)
    if model is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unavailable", "model_loaded": False, "model_version": None},
        )
    return JSONResponse(
        content={
            "status": "ok",
            "model_loaded": True,
            "model_version": model.model_version,
        }
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(encounter: Encounter, request: Request) -> PredictionResponse:
    """Score a single encounter and flag it against the served threshold."""
    model = _require_model(request)
    frame = encounters_to_frame([encounter])
    risk_score = model.predict(frame)[0]
    return _prediction(model, risk_score)


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(batch: BatchPredictRequest, request: Request) -> BatchPredictResponse:
    """Score a batch of encounters (max 1000) and return per-encounter results."""
    model = _require_model(request)
    frame = encounters_to_frame(batch.encounters)
    scores = model.predict(frame)
    predictions = [_prediction(model, score) for score in scores]
    return BatchPredictResponse(
        predictions=predictions,
        count=len(predictions),
        model_name=model.model_name,
        model_version=model.model_version,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("carereturn.api.main:app", host="0.0.0.0", port=8000)
