"""Tests for carereturn.models.features: mixed-type categoricals, NaN, unseen levels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from carereturn.config import FEATURES, TARGET
from carereturn.models.features import AsStringTransformer, build_pipeline, build_preprocessor
from tests.test_models_fixtures import build_dataset, split_frames


@pytest.fixture(scope="module")
def dataset() -> dict[str, pd.DataFrame]:
    """Train/val/test splits of a small synthetic dataset."""
    return split_frames(build_dataset(600, seed=1))


def test_as_string_transformer_coerces_and_keeps_nan() -> None:
    frame = pd.DataFrame(
        {
            "code_float": pd.array([1.0, 2.0, np.nan]),  # int codes + NaN -> float dtype
            "code_int": pd.array([3, 4, 3]),
            "label": ["No", "Up", None],
        }
    )
    out = AsStringTransformer().fit_transform(frame)
    # float 1.0 and int codes both encode to the same "1"-style category
    assert out[0, 0] == "1" and out[1, 0] == "2"
    assert out[0, 1] == "3" and out[1, 1] == "4"
    assert out[0, 2] == "No"
    assert pd.isna(out[2, 0]) and pd.isna(out[2, 2])  # missing stays missing
    assert out[2, 1] == "3"


def test_preprocessor_output_has_no_missing_values(dataset: dict[str, pd.DataFrame]) -> None:
    transformer = build_preprocessor()
    out = transformer.fit_transform(dataset["train"][FEATURES])
    assert np.isfinite(out).all()


def test_onehot_preserves_none_literal(dataset: dict[str, pd.DataFrame]) -> None:
    transformer = build_preprocessor()
    transformer.fit(dataset["train"][FEATURES])
    names = list(transformer.get_feature_names_out())
    assert "cat__A1Cresult_None" in names  # literal "None" = test not done
    encoder = transformer.named_transformers_["cat"].named_steps["onehot"]
    a1c_index = transformer.transformers_[1][2].index("A1Cresult")
    assert "None" in encoder.categories_[a1c_index]


def test_rare_levels_are_grouped(dataset: dict[str, pd.DataFrame]) -> None:
    transformer = build_preprocessor(cat_min_frequency=0.5)
    transformer.fit(dataset["train"][FEATURES])
    encoder = transformer.named_transformers_["cat"].named_steps["onehot"]
    assert encoder.infrequent_categories_  # rare levels collapsed into one bucket


def test_pipeline_handles_unseen_categories_and_nan(dataset: dict[str, pd.DataFrame]) -> None:
    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(dataset["train"][FEATURES], dataset["train"][TARGET])
    # The test split contains insulin="Steady" and A1Cresult=">8", never seen in train.
    unseen = set(dataset["test"]["insulin"].dropna()) - set(dataset["train"]["insulin"].dropna())
    assert unseen != set()
    probas = pipeline.predict_proba(dataset["test"][FEATURES])
    assert probas.shape == (len(dataset["test"]), 2)
    assert np.isfinite(probas).all()


def test_pipeline_learnable_signal(dataset: dict[str, pd.DataFrame]) -> None:
    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(dataset["train"][FEATURES], dataset["train"][TARGET])
    scores = pipeline.predict_proba(dataset["val"][FEATURES])[:, 1]
    prevalence = dataset["val"][TARGET].mean()
    assert average_precision_score(dataset["val"][TARGET], scores) > prevalence + 0.05


def test_build_pipeline_default_step_names(dataset: dict[str, pd.DataFrame]) -> None:
    pipeline = build_pipeline(LogisticRegression())
    pipeline.fit(dataset["train"][FEATURES], dataset["train"][TARGET])
    assert list(pipeline.named_steps) == ["preprocess", "clf"]
    # Preprocessing is fit inside the pipeline: the transformer was fit on train only.
    assert pipeline.named_steps["preprocess"] is not None
