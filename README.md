# mlops-project-2026-groupe1

CareReturn — Prédiction du risque de réadmission hospitalière (< 30 jours) des patients diabétiques.
Voir [SYNOPSIS.md](SYNOPSIS.md) pour le cadrage.

Données : *Diabetes 130-US Hospitals for Years 1999–2008*, UCI, CC BY 4.0 — https://doi.org/10.24432/C5230J

## Démarrage

```bash
poetry install                      # Python 3.11
poetry run pre-commit install
dvc pull                            # ou télécharger le zip UCI dans data/raw/ (diabetic_data.csv, IDS_mapping.csv)
poetry run dvc repro                # validate → prepare → split → train → evaluate
poetry run dvc metrics show
```

MLflow (tracking + registry, backend SQLite local) :

```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## API

```bash
poetry run uvicorn carereturn.api.main:app --port 8000   # ou : docker compose up --build
curl localhost:8000/health
```

Endpoints : `GET /health`, `POST /predict`, `POST /predict/batch` (≤ 1000 séjours). Documentation OpenAPI sur `/docs`.

## Structure

| Chemin | Contenu |
|---|---|
| `src/carereturn/config.py` | Chemins, schéma des variables, constantes partagées |
| `src/carereturn/data/` | Lecture, validation, cohorte, séparation par patient |
| `src/carereturn/models/` | Prétraitement, métriques (AP, recall@k), entraînement, évaluation, MLflow |
| `src/carereturn/api/` | Service FastAPI |
| `dvc.yaml`, `params.yaml` | Pipeline et paramètres |
| `.github/workflows/ci.yml` | Lint, tests, build Docker |

## Tests

```bash
poetry run ruff check . && poetry run pytest
```
