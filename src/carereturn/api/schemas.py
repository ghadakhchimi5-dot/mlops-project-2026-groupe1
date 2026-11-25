"""Pydantic v2 request/response schemas for the CareReturn prediction API.

The `Encounter` schema mirrors `config.FEATURES` exactly: it is the input
contract of the served model. Any drift between this schema and
`config.FEATURES` is a bug, so the test-suite asserts field parity.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# --- Controlled vocabularies (must match the training pipeline contract) ----

AgeBand = Literal[
    "[0-10)",
    "[10-20)",
    "[20-30)",
    "[30-40)",
    "[40-50)",
    "[50-60)",
    "[60-70)",
    "[70-80)",
    "[80-90)",
    "[90-100)",
]
A1CResult = Literal["None", "Norm", ">7", ">8"]
MaxGluSerum = Literal["None", "Norm", ">200", ">300"]
Insulin = Literal["No", "Steady", "Up", "Down"]
Change = Literal["No", "Ch"]
DiabetesMed = Literal["No", "Yes"]

# Administrative code ids are categorical despite being stored as integers.
CodeId = Field(ge=1, le=100, description="Administrative code id (positive integer).")

# Maximum number of encounters accepted by the batch endpoint.
MAX_BATCH_SIZE = 1000

EXAMPLE_ENCOUNTER: dict[str, int | str] = {
    "time_in_hospital": 5,
    "num_lab_procedures": 41,
    "num_procedures": 1,
    "num_medications": 12,
    "number_outpatient": 2,
    "number_emergency": 0,
    "number_inpatient": 1,
    "number_diagnoses": 7,
    "age": "[70-80)",
    "admission_type_id": 1,
    "admission_source_id": 7,
    "discharge_disposition_id": 3,
    "A1Cresult": ">7",
    "max_glu_serum": "None",
    "insulin": "Up",
    "change": "Ch",
    "diabetesMed": "Yes",
}


class Encounter(BaseModel):
    """A single hospital stay, with exactly the `config.FEATURES` columns.

    Numeric features are strict non-negative integers (bounded by values seen
    in the dataset, with headroom); categorical features are restricted to the
    value sets the training pipeline was built on. Unknown extra fields are
    rejected so that silent typos cannot degrade predictions.
    """

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        json_schema_extra={"examples": [EXAMPLE_ENCOUNTER]},
    )

    # --- Numeric features (config.NUMERIC_FEATURES, in order) ---
    time_in_hospital: int = Field(ge=1, le=30, description="Days spent in hospital (1-30).")
    num_lab_procedures: int = Field(
        ge=0, le=200, description="Number of laboratory procedures performed."
    )
    num_procedures: int = Field(ge=0, le=20, description="Number of non-lab procedures performed.")
    num_medications: int = Field(
        ge=0, le=100, description="Number of distinct medications administered."
    )
    number_outpatient: int = Field(ge=0, le=100, description="Outpatient visits in the year prior.")
    number_emergency: int = Field(ge=0, le=100, description="Emergency visits in the year prior.")
    number_inpatient: int = Field(
        ge=0, le=100, description="Inpatient admissions in the year prior."
    )
    number_diagnoses: int = Field(
        ge=0, le=20, description="Number of diagnoses recorded for the stay."
    )

    # --- Categorical features (config.CATEGORICAL_FEATURES, in order) ---
    age: AgeBand = Field(description="Age bracket of the patient, e.g. '[70-80)'.")
    admission_type_id: int = CodeId
    admission_source_id: int = CodeId
    discharge_disposition_id: int = CodeId
    A1Cresult: A1CResult = Field(description="HbA1c test result ('None' = not taken).")
    max_glu_serum: MaxGluSerum = Field(
        description="Glucose serum test result ('None' = not taken)."
    )
    insulin: Insulin = Field(description="Insulin dosage change during the stay.")
    change: Change = Field(description="Whether any medication was changed.")
    # `diabetesMed` keeps the dataset's original camelCase column name (contract).
    diabetesMed: DiabetesMed = Field(description="Whether a diabetes medication was prescribed.")  # noqa: N815


class PredictionResponse(BaseModel):
    """Risk score for one encounter, plus the model that produced it."""

    risk_score: float = Field(
        ge=0.0, le=1.0, description="Probability of readmission within 30 days."
    )
    flagged: bool = Field(description="True if risk_score >= threshold.")
    threshold: float = Field(description="Decision threshold used to flag encounters.")
    model_name: str = Field(description="Name of the served model.")
    model_version: str = Field(description="Version of the served model.")


class BatchPredictRequest(BaseModel):
    """Batch of encounters to score, capped at `MAX_BATCH_SIZE` items."""

    model_config = ConfigDict(extra="forbid")

    encounters: list[Encounter] = Field(
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description=f"Between 1 and {MAX_BATCH_SIZE} encounters.",
    )


class BatchPredictResponse(BaseModel):
    """Per-encounter predictions for a batch, in request order."""

    predictions: list[PredictionResponse] = Field(
        description="One prediction per input encounter, in request order."
    )
    count: int = Field(ge=1, description="Number of predictions returned.")
    model_name: str = Field(description="Name of the served model.")
    model_version: str = Field(description="Version of the served model.")
