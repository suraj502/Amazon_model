"""Train the baseline M3 model and write candidate probabilities."""

from __future__ import annotations

import argparse
from copy import deepcopy

import pandas as pd

from src.model.predict import predict_matches, write_predictions
from src.model.train import train_model
from src.utils.config_loader import CONFIG


def main() -> None:
    """Train M3 with entity-disjoint validation and score the candidate table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="output/feature_table.parquet")
    parser.add_argument(
        "--pair-text",
        default="output/enriched_candidate_pairs.parquet",
    )
    parser.add_argument("--ground-truth")
    parser.add_argument("--model", default="artifacts/m3_model.pkl")
    parser.add_argument("--output", default="output/m3_predictions.tsv")
    args = parser.parse_args()

    config = deepcopy(CONFIG)
    config["features"]["embedding"] = {"enabled": False}
    config["model"]["ablation"]["enabled"] = False
    config["model"]["hard_negative_mining"]["enabled"] = False
    features = pd.read_parquet(args.features)
    pair_text = pd.read_parquet(args.pair_text)
    ground_truth = (
        pd.read_csv(args.ground_truth, sep="\t", dtype=str, keep_default_na=False)
        if args.ground_truth
        else None
    )
    model = train_model(
        features,
        config=config,
        pair_text=pair_text,
        ground_truth=ground_truth,
        model_path=args.model,
    )
    predictions = predict_matches(
        {"model": model, "feature_columns": model.m3_feature_columns_},
        features,
        config,
    )
    write_predictions(predictions, args.output)
    print(f"Entity-level OOF metrics: {model.m3_cv_metrics_}")
    print(f"Wrote {len(predictions):,} probabilities to {args.output}")


if __name__ == "__main__":
    main()
