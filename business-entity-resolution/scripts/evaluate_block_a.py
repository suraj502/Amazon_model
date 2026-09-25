from pathlib import Path
import pandas as pd

PROCESSED = Path("data/processed")
RAW = Path("data/raw")

ID_COL = "entity_id"
NAME_COL = "name_alnum_norm"


def build_name_index(path: Path):
    df = pd.read_parquet(path, columns=[ID_COL, NAME_COL])

    df = df[
        df[NAME_COL].notna()
        & df[NAME_COL].astype(str).str.strip().ne("")
    ]

    return (
        df.groupby(NAME_COL, sort=False)[ID_COL]
        .agg(set)
        .to_dict()
    )


def evaluate_source(target_path: Path, ground_truth: pd.DataFrame, source_prefix: str):
    print(f"\nBuilding index for {source_prefix}...")
    index = build_name_index(target_path)

    total_true = 0
    retrieved_true = 0
    entities_with_truth = 0
    entities_with_retrieval = 0

    for row in ground_truth.itertuples(index=False):
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

        entities_with_truth += 1
        total_true += len(true_ids)

        # Use the S1 ID to recover its normalized name.
        # This evaluator will be completed after loading S1 names.
        source1_name = s1_name_index.get(row.source1_entity_id)

        if not source1_name:
            continue

        candidates = index.get(source1_name, set())

        if candidates:
            entities_with_retrieval += 1

        retrieved_true += len(true_ids & candidates)

    recall = retrieved_true / total_true if total_true else 0.0

    print(f"{source_prefix} true matches:       {total_true:,}")
    print(f"{source_prefix} retrieved matches:  {retrieved_true:,}")
    print(f"{source_prefix} match recall:       {recall:.4%}")
    print(f"{source_prefix} entities w/ truth: {entities_with_truth:,}")
    print(f"{source_prefix} entities w/ cand.:  {entities_with_retrieval:,}")

    return total_true, retrieved_true


print("Loading S1 names...")
s1 = pd.read_parquet(
    PROCESSED / "train_source1_normalized.parquet",
    columns=[ID_COL, NAME_COL],
)

s1_name_index = dict(
    zip(
        s1[ID_COL].astype(str),
        s1[NAME_COL].fillna("").astype(str),
    )
)

print("Loading ground truth...")
ground_truth = pd.read_csv(
    RAW / "train_ground_truth.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False,
)

print(f"Ground-truth rows: {len(ground_truth):,}")

s2_total, s2_retrieved = evaluate_source(
    PROCESSED / "train_source2_normalized.parquet",
    ground_truth,
    "S2-",
)

s3_total, s3_retrieved = evaluate_source(
    PROCESSED / "train_source3_normalized.parquet",
    ground_truth,
    "S3-",
)

total_true = s2_total + s3_total
total_retrieved = s2_retrieved + s3_retrieved

print("\n" + "=" * 60)
print("BLOCK A - EXACT NORMALIZED NAME RECALL")
print("=" * 60)
print(f"Total true matches:      {total_true:,}")
print(f"Retrieved true matches:  {total_retrieved:,}")
print(
    f"Overall match recall:    "
    f"{total_retrieved / total_true:.4%}"
    if total_true
    else "Overall match recall:    0.0000%"
)

