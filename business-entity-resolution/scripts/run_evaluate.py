"""Run M4 threshold selection and write entity-level match results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluation.threshold_search import compare_decision_strategies, select_matches
from src.submission.validate_submission import validate_submission
from src.utils.config_loader import CONFIG


def main() -> None:
    """Select an M4 strategy on labeled development data and write all S1 rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default="output/m3_predictions.tsv")
    parser.add_argument("--source1", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--output", default="output/dev_matching_results.tsv")
    args = parser.parse_args()

    scores = pd.read_csv(args.predictions, sep="\t", dtype=str)
    source1 = pd.read_parquet(args.source1)
    ground_truth = pd.read_csv(args.ground_truth, sep="\t", dtype=str, keep_default_na=False)
    report = compare_decision_strategies(scores, ground_truth, CONFIG)
    winner = report.iloc[0]
    selected = select_matches(
        scores,
        strategy=str(winner["strategy"]),
        threshold=float(winner["threshold"]),
        margin=float(winner["margin"]) if pd.notna(winner["margin"]) else 0.1,
    )
    matched_by_source = {
        source_id: sorted(group["matched_entity_ids"].astype(str))
        for source_id, group in selected.groupby("source1_entity_id", sort=False)
    }
    result = pd.DataFrame(
        {
            "source1_entity_id": source1["entity_id"].astype(str),
            "matched_entity_ids": [
                ",".join(matched_by_source.get(source_id, []))
                for source_id in source1["entity_id"].astype(str)
            ],
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, sep="\t", index=False)
    validate_submission(output, CONFIG)
    print(f"Selected strategy: {winner['strategy']}")
    print(f"Selected threshold: {winner['threshold']}")
    print(f"Development F0.5: {winner['f05']:.6f}")
    print(f"Wrote {len(result):,} rows to {output}")


if __name__ == "__main__":
    main()
