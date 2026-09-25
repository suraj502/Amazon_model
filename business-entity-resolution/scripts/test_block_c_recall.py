from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.blocking.name_blocking import (
    build_char_ngram_index,
    generate_char_ngram_candidates,
)


def load_config():
    with open(
        ROOT / "config" / "config.yaml",
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


def evaluate_blocker(
    source1_df,
    target_df,
    gt_df,
    source_name,
    config,
):
    print("\n" + "=" * 60)
    print(f"Testing Block C on {source_name}")
    print("=" * 60)

    print(f"S1 rows: {len(source1_df):,}")
    print(f"{source_name} rows: {len(target_df):,}")

    print("\nBuilding character n-gram index...")

    index = build_char_ngram_index(
        target_df,
        config,
    )

    print("\nGenerating candidates...")

    candidates = generate_char_ngram_candidates(
        source1_df,
        index,
        config,
    )

    candidate_pairs = set(
        zip(
            candidates["source1_entity_id"].astype(str),
            candidates["candidate_entity_id"].astype(str),
        )
    )

    # Build ground-truth pairs for this source.
    truth_pairs = set()

    for _, row in gt_df.iterrows():
        source1_id = str(row["source1_entity_id"])
        matched_ids = str(
            row["matched_entity_ids"]
        ).strip()

        if (
            not matched_ids
            or matched_ids.lower() == "nan"
        ):
            continue

        for entity_id in matched_ids.split(","):
            entity_id = entity_id.strip()

            if entity_id.startswith(source_name):
                truth_pairs.add(
                    (
                        source1_id,
                        entity_id,
                    )
                )

    retrieved = len(
        truth_pairs & candidate_pairs
    )

    total_true = len(truth_pairs)

    recall = (
        retrieved / total_true
        if total_true
        else 0.0
    )

    unique_s1 = (
        candidates["source1_entity_id"]
        .nunique()
        if not candidates.empty
        else 0
    )

    print("\nResults")
    print("-" * 40)
    print(f"True matches:       {total_true:,}")
    print(f"Retrieved matches:  {retrieved:,}")
    print(f"Recall:             {recall:.4%}")
    print(
        f"Candidate pairs:    {len(candidates):,}"
    )
    print(f"Unique S1 covered:  {unique_s1:,}")

    return {
        "source": source_name,
        "true_matches": total_true,
        "retrieved": retrieved,
        "recall": recall,
        "candidate_pairs": len(candidates),
    }


def main():
    config = load_config()

    dev = ROOT / "data" / "dev"

    print("Loading small dev dataset...")

    s1 = pd.read_parquet(
        dev / "source1_sample.parquet"
    )

    s2 = pd.read_parquet(
        dev / "source2_sample.parquet"
    )

    s3 = pd.read_parquet(
        dev / "source3_sample.parquet"
    )

    gt = pd.read_csv(
        dev / "train_ground_truth_sample.tsv",
        sep="\t",
    )

    print(f"S1: {len(s1):,}")
    print(f"S2: {len(s2):,}")
    print(f"S3: {len(s3):,}")
    print(f"GT: {len(gt):,}")

    results = []

    results.append(
        evaluate_blocker(
            s1,
            s2,
            gt,
            "S2",
            config,
        )
    )

    results.append(
        evaluate_blocker(
            s1,
            s3,
            gt,
            "S3",
            config,
        )
    )

    print("\n" + "=" * 60)
    print("BLOCK C SUMMARY")
    print("=" * 60)

    for result in results:
        print(
            f"{result['source']}: "
            f"recall={result['recall']:.4%}, "
            f"candidates="
            f"{result['candidate_pairs']:,}"
        )


if __name__ == "__main__":
    main()