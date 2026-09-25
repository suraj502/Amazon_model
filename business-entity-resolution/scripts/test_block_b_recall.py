import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.blocking.name_blocking import (
    build_name_token_index,
    generate_name_token_candidates_to_file,
)


PROCESSED = Path("data/processed")
RAW = Path("data/raw")
CONFIG_PATH = Path("config/config.yaml")

ID_COL = "entity_id"
S1_SAMPLE_SIZE = 5000
DIAGNOSTIC_OUTPUTS = {
    "S2-": Path("output/diagnostic_block_b_5k_s2.tsv"),
    "S3-": Path("output/diagnostic_block_b_5k_s3.tsv"),
}
EXPECTED_COLUMNS = [
    "source1_entity_id",
    "candidate_entity_id",
    "score",
]


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_s1_sample():
    print("Loading S1 sample...")

    return pd.read_parquet(
        PROCESSED / "train_source1_normalized.parquet",
        columns=[ID_COL, "name_tokens"],
    ).head(S1_SAMPLE_SIZE)


def load_ground_truth():
    print("Loading ground truth...")

    return pd.read_csv(
        RAW / "train_ground_truth.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )


def validate_artifact(path, sample_ids, source_prefix):
    if not path.exists():
        raise ValueError(f"Artifact does not exist: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Artifact is empty: {path}")

    artifact = pd.read_csv(path, sep="\t", dtype=str)
    if list(artifact.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"Invalid header in {path}: {list(artifact.columns)}"
        )
    if artifact.empty:
        raise ValueError(f"Artifact has no candidate rows: {path}")
    if artifact[EXPECTED_COLUMNS].isna().any().any():
        raise ValueError(f"Artifact contains null values: {path}")

    artifact_s1_ids = set(artifact["source1_entity_id"])
    unexpected_s1_ids = artifact_s1_ids - sample_ids
    if unexpected_s1_ids:
        raise ValueError(
            f"Artifact contains S1 IDs outside the 5,000-row sample: "
            f"{sorted(unexpected_s1_ids)[:3]}"
        )

    invalid_target_ids = ~artifact["candidate_entity_id"].str.startswith(
        source_prefix
    )
    if invalid_target_ids.any():
        raise ValueError(
            f"Artifact contains candidate IDs without {source_prefix} prefix"
        )

    print(
        f"Validated {path}: rows={len(artifact):,}, "
        f"unique_s1_ids={artifact['source1_entity_id'].nunique():,}, "
        f"sample_size={len(sample_ids):,}"
    )
    return artifact


def generate_or_reuse_artifact(
    s1_sample,
    target_path,
    source_prefix,
    config,
    force,
):
    output_path = DIAGNOSTIC_OUTPUTS[source_prefix]
    sample_ids = set(s1_sample[ID_COL].astype(str))

    if output_path.exists():
        try:
            return validate_artifact(output_path, sample_ids, source_prefix)
        except ValueError:
            if not force:
                raise
            print(
                "Invalid artifact will be replaced because --force was set: "
                f"{output_path}"
            )

    target = pd.read_parquet(target_path, columns=[ID_COL, "name_tokens"])
    index = build_name_token_index(target, config)
    temp_path = output_path.with_suffix(".tmp.tsv")
    if temp_path.exists():
        temp_path.unlink()

    candidate_count = generate_name_token_candidates_to_file(
        s1_sample,
        index,
        config,
        str(temp_path),
    )
    artifact = validate_artifact(temp_path, sample_ids, source_prefix)
    temp_path.replace(output_path)
    print(f"Candidate rows written: {candidate_count:,}")
    return artifact


def evaluate_source(
    s1_sample,
    target_path,
    ground_truth,
    source_prefix,
    config,
    force,
):
    print(f"\n{'=' * 60}")
    print(f"Testing Block B against {source_prefix}")
    print(f"{'=' * 60}")

    candidates = generate_or_reuse_artifact(
        s1_sample,
        target_path,
        source_prefix,
        config,
        force,
    )

    candidate_lookup = {}

    for row in candidates.itertuples(index=False):
        candidate_lookup.setdefault(
            str(row.source1_entity_id),
            set(),
        ).add(str(row.candidate_entity_id))

    sample_ids = set(s1_sample[ID_COL].astype(str))

    sample_truth = ground_truth[
        ground_truth["source1_entity_id"].astype(str).isin(sample_ids)
    ]

    total_true = 0
    retrieved_true = 0

    for row in sample_truth.itertuples(index=False):
        matched_ids = str(row.matched_entity_ids).strip()

        if not matched_ids:
            continue

        true_ids = {
            x.strip()
            for x in matched_ids.split(",")
            if x.strip().startswith(source_prefix)
        }

        if not true_ids:
            continue

        total_true += len(true_ids)

        retrieved = candidate_lookup.get(
            str(row.source1_entity_id),
            set(),
        )

        retrieved_true += len(true_ids & retrieved)

    recall = (
        retrieved_true / total_true
        if total_true
        else 0.0
    )

    print(f"True matches:       {total_true:,}")
    print(f"Retrieved matches:  {retrieved_true:,}")
    print(f"Block B recall:     {recall:.4%}")

    return (
        total_true,
        retrieved_true,
        len(candidates),
        candidates["source1_entity_id"].nunique(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing invalid diagnostic artifact.",
    )
    args = parser.parse_args()

    print("Loading config...")
    config = load_config()

    s1_sample = load_s1_sample()

    print(
        f"S1 sample size: {len(s1_sample):,}"
    )

    ground_truth = load_ground_truth()

    print(
        f"Ground-truth rows: {len(ground_truth):,}"
    )

    s2_total, s2_retrieved, s2_rows, s2_s1_ids = evaluate_source(
        s1_sample,
        PROCESSED / "train_source2_normalized.parquet",
        ground_truth,
        "S2-",
        config,
        args.force,
    )

    s3_total, s3_retrieved, s3_rows, s3_s1_ids = evaluate_source(
        s1_sample,
        PROCESSED / "train_source3_normalized.parquet",
        ground_truth,
        "S3-",
        config,
        args.force,
    )

    total_true = s2_total + s3_total
    total_retrieved = s2_retrieved + s3_retrieved

    overall_recall = (
        total_retrieved / total_true
        if total_true
        else 0.0
    )

    print(f"\n{'=' * 60}")
    print("BLOCK B SAMPLE RECALL")
    print(f"{'=' * 60}")
    print(f"True matches:       {total_true:,}")
    print(f"Retrieved matches:  {total_retrieved:,}")
    print(f"Overall recall:     {overall_recall:.4%}")
    print(f"S2 artifact rows:   {s2_rows:,}")
    print(f"S2 unique S1 IDs:   {s2_s1_ids:,}")
    print(f"S3 artifact rows:   {s3_rows:,}")
    print(f"S3 unique S1 IDs:   {s3_s1_ids:,}")


if __name__ == "__main__":
    main()