"""Contract tests for model training and prediction."""

import numpy as np
import pandas as pd

from src.model.predict import predict_matches
from src.model.train import entity_cv_splits


def test_model_uses_entity_level_splits():
    """Model validation must split by source-one entity."""
    groups = pd.Series(["S1-1", "S1-1", "S1-2", "S1-2", "S1-3", "S1-4"])
    splits = entity_cv_splits(groups, n_splits=2)

    for train_indices, validation_indices in splits:
        assert set(groups.iloc[train_indices]).isdisjoint(
            set(groups.iloc[validation_indices])
        )


class _ProbabilityModel:
    def predict_proba(self, matrix: pd.DataFrame) -> np.ndarray:
        values = matrix["name_score"].to_numpy(dtype=np.float32)
        return np.column_stack([1.0 - values, values])


def test_prediction_contract_preserves_ids_and_probability_bounds():
    """M3 predictions retain candidate IDs and emit probabilities in [0, 1]."""
    features = pd.DataFrame(
        {
            "source1_entity_id": ["S1-1", "S1-2"],
            "candidate_entity_id": ["S2-1", "S3-2"],
            "source_type": ["S2", "S3"],
            "name_score": np.array([0.9, 0.2], dtype=np.float32),
        }
    )

    predictions = predict_matches(
        {"model": _ProbabilityModel(), "feature_columns": ["name_score"]},
        features,
    )

    assert list(predictions.columns) == [
        "source1_entity_id",
        "candidate_entity_id",
        "probability",
    ]
    assert predictions[["source1_entity_id", "candidate_entity_id"]].to_dict("records") == features[["source1_entity_id", "candidate_entity_id"]].to_dict("records")
    assert predictions["probability"].between(0.0, 1.0).all()
