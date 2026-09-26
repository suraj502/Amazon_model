"""Run M3 feature construction for persisted candidate pairs."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd

from src.features.build_feature_table import (
    build_feature_table,
    enrich_candidate_pairs,
    _read_table,
)
from src.utils.config_loader import CONFIG


def main() -> None:
    """Build a labeled M3 feature table from the persisted M2 candidates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default="output/candidate_pairs.tsv")
    parser.add_argument("--source1", required=True)
    parser.add_argument("--source2", required=True)
    parser.add_argument("--source3", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--output", default="output/feature_table.parquet")
    parser.add_argument(
        "--enriched-output",
        default="output/enriched_candidate_pairs.parquet",
    )
    args = parser.parse_args()

    config = deepcopy(CONFIG)
    config["features"]["embedding"] = {"enabled": False}
    candidates = _read_table(args.candidates)
    source1 = _read_table(args.source1)
    source2 = _read_table(args.source2)
    source3 = _read_table(args.source3)
    ground_truth = _read_table(args.ground_truth)
    enriched = enrich_candidate_pairs(candidates, source1, source2, source3, config)
    feature_table = build_feature_table(
        enriched,
        config=config,
        ground_truth=ground_truth,
    )
    feature_path = Path(args.output)
    enriched_path = Path(args.enriched_output)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    enriched_path.parent.mkdir(parents=True, exist_ok=True)
    feature_table.to_parquet(feature_path, index=False)
    enriched.to_parquet(enriched_path, index=False)
    print(
        f"Wrote {len(feature_table):,} feature rows to {feature_path}; "
        f"enriched pairs saved to {enriched_path}"
    )


if __name__ == "__main__":
    main()
