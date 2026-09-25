from pathlib import Path

import pandas as pd
import yaml

from src.blocking.name_blocking import (
    build_name_token_index,
    generate_name_token_candidates_to_file,
)

PROCESSED = Path("data/processed")
RAW = Path("data/raw")

ID_COL = "entity_id"
CONFIG_PATH = Path("config/config.yaml")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_ground_truth():
    return pd.read_csv(
        RAW / "train_ground_truth.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )


def build_s1_lookup():
    print("Loading S1 names...")

    s1 = pd.read_parquet(
        PROCESSED / "train_source1_normalized.parquet",
        columns=[ID_COL, "name_tokens"],
    )

    return s1


def evaluate_source(
    s1: pd.DataFrame,
    target_path: Path,
    ground_truth: pd.DataFrame,
    source_prefix: str,
    config: dict,
):
    print(f"\nLoading {source_prefix} target...")

    target = pd.read_parquet(
        target_path,
        columns=[ID_COL, "name_tokens"],
    )

    print(f"Running Block B against {source_prefix}...")

    print(f"Building reusable name-token index for {source_prefix}...")

    index = build_name_token_index(
        target,
        config,
    )

    print(f"Generating candidates for {source_prefix}...")

    candidate_path = (
        Path("output")
        / f"block_b_{source_prefix.lower().replace('-', '')}.tsv"
    )

    candidate_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate_count = generate_name_token_candidates_to_file(
        s1,
        index,
        config,
        str(candidate_path),
    )

    print(f"Candidates written: {candidate_count:,}")

    candidates = pd.read_csv(
        candidate_path,
        sep="\t",
        dtype={
            "source1_entity_id": str,
            "candidate_entity_id": str,
        },
    )

    candidate_lookup = {}

    for source_id, target_id, score in candidates.itertuples(
        index=False,
        name=None,
    ):
        candidate_lookup.setdefault(
            str(source_id),
            set(),
        ).add(str(target_id))

    total_true = 0
    retrieved_true = 0
    entities_with_truth = 0
    entities_with_candidates = 0

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

        retrieved = candidate_lookup.get(
            str(row.source1_entity_id),
            set(),
        )

        if retrieved:
            entities_with_candidates += 1

        retrieved_true += len(true_ids & retrieved)

    recall = (
        retrieved_true / total_true
        if total_true
        else 0.0
    )

    print(
        f"{source_prefix} true matches:        "
        f"{total_true:,}"
    )
    print(
        f"{source_prefix} retrieved matches:   "
        f"{retrieved_true:,}"
    )
    print(
        f"{source_prefix} match recall:        "
        f"{recall:.4%}"
    )
    print(
        f"{source_prefix} entities w/ truth: "
        f"{entities_with_truth:,}"
    )
    print(
        f"{source_prefix} entities w/ cand.:   "
        f"{entities_with_candidates:,}"
    )
    print(
        f"{source_prefix} candidate pairs:    "
        f"{candidate_count:,}"
    )

    return total_true, retrieved_true


print("Loading config...")
config = load_config()

print("Loading S1...")
s1 = build_s1_lookup()

print("Loading ground truth...")
ground_truth = load_ground_truth()

print(f"Ground-truth rows: {len(ground_truth):,}")

s2_total, s2_retrieved = evaluate_source(
    s1,
    PROCESSED / "train_source2_normalized.parquet",
    ground_truth,
    "S2-",
    config,
)

s3_total, s3_retrieved = evaluate_source(
    s1,
    PROCESSED / "train_source3_normalized.parquet",
    ground_truth,
    "S3-",
    config,
)

total_true = s2_total + s3_total
total_retrieved = s2_retrieved + s3_retrieved

print("\n" + "=" * 60)
print("BLOCK B - NAME TOKEN OVERLAP RECALL")
print("=" * 60)
print(f"Total true matches:      {total_true:,}")
print(f"Retrieved true matches:  {total_retrieved:,}")
print(
    f"Overall match recall:    "
    f"{(total_retrieved / total_true):.4%}"
    if total_true
    else "Overall match recall:    0.0000%"
)