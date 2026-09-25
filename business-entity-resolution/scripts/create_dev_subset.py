from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
DEV = ROOT / "data" / "dev"

S1_SIZE = 5_000
TARGET_SIZE = 50_000
SEED = 42


def extract_true_target_ids(gt_sample, source_prefix):
    """Extract all true target IDs for the sampled S1 entities."""

    true_ids = set()

    for matched_ids in gt_sample["matched_entity_ids"]:
        if pd.isna(matched_ids):
            continue

        matched_ids = str(matched_ids).strip()

        if not matched_ids:
            continue

        for entity_id in matched_ids.split(","):
            entity_id = entity_id.strip()

            if entity_id.startswith(source_prefix):
                true_ids.add(entity_id)

    return true_ids


def build_target_dev_set(
    target_df,
    true_target_ids,
    target_size,
    seed,
    source_name,
):
    """Build target dev set containing all true matches + random negatives."""

    true_target_ids = set(
        str(entity_id)
        for entity_id in true_target_ids
    )

    target_df = target_df.copy()

    target_df["entity_id"] = (
        target_df["entity_id"]
        .astype(str)
    )

    # Keep every true target row.
    true_rows = target_df[
        target_df["entity_id"].isin(true_target_ids)
    ].copy()

    actual_true_ids = set(
        true_rows["entity_id"]
    )

    missing_true_ids = (
        true_target_ids - actual_true_ids
    )

    if missing_true_ids:
        raise ValueError(
            f"{source_name}: "
            f"{len(missing_true_ids):,} true target IDs "
            f"were not found in the normalized target data."
        )

    negative_pool = target_df[
        ~target_df["entity_id"].isin(actual_true_ids)
    ]

    negative_count = target_size - len(true_rows)

    if negative_count < 0:
        raise ValueError(
            f"{source_name}: true matches "
            f"({len(true_rows):,}) exceed target dev size "
            f"({target_size:,})."
        )

    negatives = negative_pool.sample(
        n=negative_count,
        random_state=seed,
    )

    dev_target = pd.concat(
        [
            true_rows,
            negatives,
        ],
        ignore_index=True,
    )

    # Shuffle so true matches are not all grouped together.
    dev_target = (
        dev_target
        .sample(
            frac=1.0,
            random_state=seed,
        )
        .reset_index(drop=True)
    )

    return dev_target


def main():
    DEV.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Load S1
    # ---------------------------------------------------------

    print("Loading normalized S1...")

    s1 = pd.read_parquet(
        PROCESSED / "train_source1_normalized.parquet"
    )

    print(
        f"Full S1 rows: {len(s1):,}"
    )

    s1_sample = (
        s1.sample(
            n=S1_SIZE,
            random_state=SEED,
        )
        .reset_index(drop=True)
    )

    s1_sample["entity_id"] = (
        s1_sample["entity_id"]
        .astype(str)
    )

    s1_ids = set(
        s1_sample["entity_id"]
    )

    print(
        f"Sampled S1 rows: {len(s1_sample):,}"
    )

    # ---------------------------------------------------------
    # Ground truth
    # ---------------------------------------------------------

    print("\nLoading ground truth...")

    gt = pd.read_csv(
        RAW / "train_ground_truth.tsv",
        sep="\t",
    )

    gt["source1_entity_id"] = (
        gt["source1_entity_id"]
        .astype(str)
    )

    gt_sample = gt[
        gt["source1_entity_id"].isin(s1_ids)
    ].copy()

    print(
        f"Ground-truth rows: {len(gt_sample):,}"
    )

    # ---------------------------------------------------------
    # Extract true S2/S3 IDs
    # ---------------------------------------------------------

    true_s2_ids = extract_true_target_ids(
        gt_sample,
        "S2",
    )

    true_s3_ids = extract_true_target_ids(
        gt_sample,
        "S3",
    )

    print(
        f"True S2 target IDs: {len(true_s2_ids):,}"
    )

    print(
        f"True S3 target IDs: {len(true_s3_ids):,}"
    )

    # ---------------------------------------------------------
    # Load S2
    # ---------------------------------------------------------

    print("\nLoading normalized S2...")

    s2 = pd.read_parquet(
        PROCESSED / "train_source2_normalized.parquet"
    )

    print(
        f"Full S2 rows: {len(s2):,}"
    )

    s2_sample = build_target_dev_set(
        s2,
        true_s2_ids,
        TARGET_SIZE,
        SEED,
        "S2",
    )

    print(
        f"Dev S2 rows: {len(s2_sample):,}"
    )

    # ---------------------------------------------------------
    # Load S3
    # ---------------------------------------------------------

    print("\nLoading normalized S3...")

    s3 = pd.read_parquet(
        PROCESSED / "train_source3_normalized.parquet"
    )

    print(
        f"Full S3 rows: {len(s3):,}"
    )

    s3_sample = build_target_dev_set(
        s3,
        true_s3_ids,
        TARGET_SIZE,
        SEED,
        "S3",
    )

    print(
        f"Dev S3 rows: {len(s3_sample):,}"
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    print("\nSaving development files...")

    s1_sample.to_parquet(
        DEV / "source1_sample.parquet",
        index=False,
    )

    s2_sample.to_parquet(
        DEV / "source2_sample.parquet",
        index=False,
    )

    s3_sample.to_parquet(
        DEV / "source3_sample.parquet",
        index=False,
    )

    gt_sample.to_csv(
        DEV / "train_ground_truth_sample.tsv",
        sep="\t",
        index=False,
    )

    print(
        "\nDevelopment subset created successfully."
    )

    print(
        f"S1: {len(s1_sample):,} rows"
    )

    print(
        f"S2: {len(s2_sample):,} rows"
    )

    print(
        f"S3: {len(s3_sample):,} rows"
    )

    print(
        f"GT: {len(gt_sample):,} rows"
    )

    print(
        "\nAll true S2/S3 target IDs for the "
        "sampled S1 entities are included."
    )


if __name__ == "__main__":
    main()