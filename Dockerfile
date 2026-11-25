# CareReturn API — multi-stage build (deps in builder, slim runtime).
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VERSION=2.5.2 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1

# Poetry 2.5.2 matches the poetry.lock (lock-version 2.1).
RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

WORKDIR /app

# Dependency layer: only metadata needed to resolve the lock file.
COPY pyproject.toml poetry.lock README.md ./
RUN poetry install --only main --no-root --no-ansi

# Project layer: install the carereturn package into the same venv.
COPY src ./src
RUN poetry install --only main --no-ansi

# --- Runtime ----------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CARERETURN_ROOT=/app

RUN groupadd --system app \
    && useradd --system --gid app --create-home --shell /usr/sbin/nologin app

COPY --from=builder /app/.venv /app/.venv

WORKDIR /app
COPY --chown=app:app src ./src
COPY --chown=app:app models ./models

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"]

CMD ["uvicorn", "carereturn.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
