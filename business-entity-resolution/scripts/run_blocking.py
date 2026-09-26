"""Run the configured A+B+C candidate blocking union."""

import argparse
from pathlib import Path

import pandas as pd

from src.blocking.candidate_generation import (
    apply_candidate_cap,
    generate_union_candidates,
    write_candidate_pairs,
)
from src.blocking.candidate_union import union_candidates
from src.utils.config_loader import CONFIG, get_config_value


def main() -> None:
    """Generate candidate pairs from normalized records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source1", default="data/processed/train_source1_normalized.parquet")
    parser.add_argument("--source2", default="data/processed/train_source2_normalized.parquet")
    parser.add_argument("--source3", default="data/processed/train_source3_normalized.parquet")
    parser.add_argument("--output", default=get_config_value(CONFIG, "paths", "candidate_pairs_out"))
    args = parser.parse_args()

    source1 = pd.read_parquet(args.source1)
    source2 = pd.read_parquet(args.source2)
    source3 = pd.read_parquet(args.source3)
    source2_candidates = generate_union_candidates(source1, source2, CONFIG)
    source3_candidates = generate_union_candidates(source1, source3, CONFIG)
    candidates = apply_candidate_cap(
        union_candidates([source2_candidates, source3_candidates], CONFIG),
        CONFIG,
    )
    write_candidate_pairs(candidates, str(Path(args.output)))


if __name__ == "__main__":
    main()
