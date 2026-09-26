"""Load the trained M3 model and write M4's candidate probability handoff."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.model.train import IDENTIFIER_COLUMNS
from src.features.build_feature_table import (
    _read_table,
    build_feature_table,
    enrich_candidate_pairs,
)
from src.utils.config_loader import CONFIG


PREDICTION_COLUMNS = (
    "source1_entity_id",
    "candidate_entity_id",
    "match_probability",
)


def load_model(
    path: str | Path = "artifacts/m3_model.pkl",
) -> Any:
    """Load a model artifact produced by train_model."""
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    artifact = joblib.load(model_path)
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact
    return {
        "model": artifact,
        "feature_columns": getattr(artifact, "m3_feature_columns_", None),
    }


def predict_matches(
    model: Any,
    features: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Score candidates and return only M4's required three-column schema."""
    del config
    required_ids = ("source1_entity_id", "candidate_entity_id")
    missing = [column for column in required_ids if column not in features.columns]
    if missing:
        raise ValueError("Prediction features are missing ID columns: " + ", ".join(missing))
    if features[list(required_ids)].isna().any().any():
        raise ValueError("Prediction pair IDs cannot be null")
    if features.duplicated(list(required_ids)).any():
        raise ValueError("Prediction pair IDs must be unique")

    artifact = model if isinstance(model, dict) and "model" in model else {"model": model}
    estimator = artifact["model"]
    expected_columns = artifact.get("feature_columns") or getattr(
        estimator,
        "m3_feature_columns_",
        None,
    )
    excluded = IDENTIFIER_COLUMNS | {"label"}
    matrix = features.drop(
        columns=[column for column in excluded if column in features.columns],
        errors="ignore",
    )
    if expected_columns is not None:
        missing_features = sorted(set(expected_columns) - set(matrix.columns))
        extra_features = sorted(set(matrix.columns) - set(expected_columns))
        if missing_features:
            raise ValueError(
                "Prediction data is missing model features: "
                + ", ".join(missing_features)
            )
        if extra_features:
            matrix = matrix.drop(columns=extra_features)
        matrix = matrix.reindex(columns=expected_columns)
    if not all(pd.api.types.is_numeric_dtype(matrix[column]) for column in matrix):
        raise TypeError("Prediction features must all be numeric")
    matrix = matrix.astype(np.float32)
    values = matrix.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Prediction features contain NaN or infinite values")
    probabilities = np.asarray(estimator.predict_proba(matrix)[:, 1], dtype=np.float32)
    if len(probabilities) != len(features) or not np.isfinite(probabilities).all():
        raise ValueError("Model returned invalid or non-finite match probabilities")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("Model probabilities must be within [0, 1]")

    output = features[list(required_ids)].copy().reset_index(drop=True)
    output["match_probability"] = probabilities
    return output.loc[:, PREDICTION_COLUMNS]


def write_predictions(
    predictions: pd.DataFrame,
    path: str | Path = "output/test_predictions.tsv",
) -> Path:
    """Write M4's exact TSV handoff schema."""
    if list(predictions.columns) != list(PREDICTION_COLUMNS):
        raise ValueError(
            "Prediction output columns must be exactly: "
            + ", ".join(PREDICTION_COLUMNS)
        )
    if predictions.isna().any().any():
        raise ValueError("Prediction output contains missing values")
    if not np.isfinite(
        predictions["match_probability"].to_numpy(dtype=np.float32)
    ).all():
        raise ValueError("Prediction probabilities contain NaN or infinity")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(destination, sep="\t", index=False)
    return destination


def main() -> None:
    """Build test features with training TF-IDF vocabularies and write M4 handoff."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, help="M2 test candidate-pair TSV")
    parser.add_argument("--source1", required=True, help="Normalized test Source 1 table")
    parser.add_argument("--source2", required=True, help="Normalized test Source 2 table")
    parser.add_argument("--source3", required=True, help="Normalized test Source 3 table")
    parser.add_argument("--model", default="artifacts/m3_model.pkl")
    parser.add_argument(
        "--output",
        default="output/test_predictions.tsv",
        help="M4 handoff TSV path",
    )
    parser.add_argument(
        "--features-output",
        default="output/test_feature_table.parquet",
        help="Optional persisted test feature table path",
    )
    args = parser.parse_args()

    artifact = load_model(args.model)
    candidates = _read_table(args.candidates)
    enriched = enrich_candidate_pairs(
        candidates,
        _read_table(args.source1),
        _read_table(args.source2),
        _read_table(args.source3),
    )
    feature_table = build_feature_table(
        enriched,
        config=CONFIG,
        tfidf_vectorizers=artifact.get("tfidf_vectorizers"),
    )
    feature_path = Path(args.features_output)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    feature_table.to_parquet(feature_path, index=False)
    predictions = predict_matches(artifact, feature_table, CONFIG)
    output_path = write_predictions(predictions, args.output)
    print(
        f"Wrote {len(predictions)} candidate probabilities to {output_path}; "
        f"feature table saved to {feature_path}"
    )


if __name__ == "__main__":
    main()
