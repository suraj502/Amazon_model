import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from src.blocking.candidate_generation import generate_final_candidates




SOURCE1_PATH = ROOT / "data" / "dev" / "source1_sample.parquet"
SOURCE2_PATH = ROOT / "data" / "dev" / "source2_sample.parquet"
SOURCE3_PATH = ROOT / "data" / "dev" / "source3_sample.parquet"
GT_PATH = ROOT / "data" / "dev" / "train_ground_truth_sample.tsv"

CONFIG_PATH = ROOT / "config" / "config.yaml"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_ground_truth():
    gt = pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
    )

    gt["matched_entity_ids"] = (
        gt["matched_entity_ids"]
        .fillna("")
        .astype(str)
    )

    return gt


def evaluate_source(
    source_name,
    source1_df,
    target_df,
    gt,
    config,
):
    print("\n" + "=" * 60)
    print(f"Evaluating {source_name}")
    print("=" * 60)

    candidates = generate_final_candidates(
        source1_df,
        target_df,
        config,
    )

    true_pairs = set()

    prefix = source_name

    for _, row in gt.iterrows():

        source1_id = str(
            row["source1_entity_id"]
        )

        matched_ids = str(
            row["matched_entity_ids"]
        )

        if not matched_ids:
            continue

        for target_id in matched_ids.split(","):

            target_id = target_id.strip()

            if not target_id:
                continue

            if target_id.startswith(prefix):

                true_pairs.add(
                    (
                        source1_id,
                        target_id,
                    )
                )

    candidate_pairs = set(
        zip(
            candidates["source1_entity_id"],
            candidates["candidate_entity_id"],
        )
    )

    retrieved = (
        true_pairs
        & candidate_pairs
    )

    true_entity_ids = {
        source1_id
        for source1_id, _ in true_pairs
    }

    candidate_entity_ids = {
        source1_id
        for source1_id, _ in candidate_pairs
    }

    recall = (
        len(retrieved) / len(true_pairs)
        if true_pairs
        else 0.0
    )

    print("\nResults:")
    print(
        f"True matches:       {len(true_pairs):,}"
    )

    print(
        f"Retrieved matches:  {len(retrieved):,}"
    )

    print(
        f"Recall:             {recall:.4%}"
    )

    print(
        f"Candidate pairs:    {len(candidates):,}"
    )

    print(
        f"Unique S1 covered:  "
        f"{len(candidate_entity_ids):,}"
    )

    print(
        f"True S1 entities:   "
        f"{len(true_entity_ids):,}"
    )

    return {
        "source": source_name,
        "true_matches": len(true_pairs),
        "retrieved_matches": len(retrieved),
        "recall": recall,
        "candidate_pairs": len(candidates),
        "unique_s1_covered": len(candidate_entity_ids),
        "true_s1_entities": len(true_entity_ids),
    }


def main():

    print("Loading dev datasets...")

    source1 = pd.read_parquet(
        SOURCE1_PATH
    )

    source2 = pd.read_parquet(
        SOURCE2_PATH
    )

    source3 = pd.read_parquet(
        SOURCE3_PATH
    )

    gt = load_ground_truth()

    config = load_config()

    print(
        f"S1: {len(source1):,}"
    )

    print(
        f"S2: {len(source2):,}"
    )

    print(
        f"S3: {len(source3):,}"
    )

    print(
        f"GT: {len(gt):,}"
    )

    results = []

    results.append(
        evaluate_source(
            "S2",
            source1,
            source2,
            gt,
            config,
        )
    )

    results.append(
        evaluate_source(
            "S3",
            source1,
            source3,
            gt,
            config,
        )
    )

    result_df = pd.DataFrame(
        results
    )

    total_true = result_df[
        "true_matches"
    ].sum()

    total_retrieved = result_df[
        "retrieved_matches"
    ].sum()

    overall_recall = (
        total_retrieved / total_true
        if total_true
        else 0.0
    )

    print("\n" + "=" * 60)
    print("A+B+C OVERALL RESULT")
    print("=" * 60)

    print(
        f"Total true matches: "
        f"{total_true:,}"
    )

    print(
        f"Total retrieved:    "
        f"{total_retrieved:,}"
    )

    print(
        f"Overall recall:     "
        f"{overall_recall:.4%}"
    )


if __name__ == "__main__":
    main()