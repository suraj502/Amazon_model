from __future__ import annotations

from typing import Any

import pandas as pd

from src.blocking.name_blocking import (
    block_by_name,
    block_by_name_token_overlap,
    block_by_name_char_ngram,
)


FINAL_COLUMNS = [
    "source1_entity_id",
    "candidate_entity_id",
]


def _normalize_block_output(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert a blocker output into the common candidate schema.
    """

    if df.empty:
        return pd.DataFrame(columns=FINAL_COLUMNS)

    result = df.copy()

    # Block A and Block B use matched_entity_id.
    if "matched_entity_id" in result.columns:
        result = result.rename(
            columns={
                "matched_entity_id": "candidate_entity_id"
            }
        )

    required = set(FINAL_COLUMNS)

    missing = required - set(result.columns)

    if missing:
        raise ValueError(
            "Blocker output missing required columns: "
            f"{sorted(missing)}"
        )

    result = result[FINAL_COLUMNS].copy()

    result["source1_entity_id"] = (
        result["source1_entity_id"].astype(str)
    )

    result["candidate_entity_id"] = (
        result["candidate_entity_id"].astype(str)
    )

    return result


def generate_union_candidates(
    source1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Run Blocks A, B and C and return their union.

    The output contains unique candidate pairs using the
    common M3-facing schema:

        source1_entity_id
        candidate_entity_id

    No country hard-filtering is applied.
    """

    required_source_columns = {"entity_id"}

    missing_source = (
        required_source_columns
        - set(source1_df.columns)
    )

    if missing_source:
        raise ValueError(
            "Source1 dataframe missing required columns: "
            f"{sorted(missing_source)}"
        )

    required_target_columns = {"entity_id"}

    missing_target = (
        required_target_columns
        - set(target_df.columns)
    )

    if missing_target:
        raise ValueError(
            "Target dataframe missing required columns: "
            f"{sorted(missing_target)}"
        )

    print("\n========================================")
    print("M2 Candidate Generation: Blocks A+B+C")
    print("========================================")

    # --------------------------------------------------------
    # Block A
    # --------------------------------------------------------

    print("\n[Block A] Exact normalized-name blocking...")

    block_a = block_by_name(
        source1_df,
        target_df,
        config,
    )

    block_a = _normalize_block_output(
        block_a
    )

    print(
        f"  Candidate pairs: {len(block_a):,}"
    )

    # --------------------------------------------------------
    # Block B
    # --------------------------------------------------------

    print("\n[Block B] Name token-overlap blocking...")

    block_b = block_by_name_token_overlap(
        source1_df,
        target_df,
        config,
    )

    block_b = _normalize_block_output(
        block_b
    )

    print(
        f"  Candidate pairs: {len(block_b):,}"
    )

    # --------------------------------------------------------
    # Block C
    # --------------------------------------------------------

    print("\n[Block C] Character n-gram blocking...")

    block_c = block_by_name_char_ngram(
        source1_df,
        target_df,
        config,
    )

    block_c = _normalize_block_output(
        block_c
    )

    print(
        f"  Candidate pairs: {len(block_c):,}"
    )

    # --------------------------------------------------------
    # Union
    # --------------------------------------------------------

    union = pd.concat(
        [
            block_a,
            block_b,
            block_c,
        ],
        ignore_index=True,
    )

    before_dedup = len(union)

    union = union.drop_duplicates(
        subset=FINAL_COLUMNS
    )

    union = union.sort_values(
        by=[
            "source1_entity_id",
            "candidate_entity_id",
        ]
    ).reset_index(
        drop=True
    )

    print(
        f"\n[Union] Before deduplication: "
        f"{before_dedup:,}"
    )

    print(
        f"[Union] After deduplication: "
        f"{len(union):,}"
    )

    return union


def apply_candidate_cap(
    candidates: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Apply the configured maximum candidate count per S1 entity.

    The cap is applied only after A+B+C union.
    """

    strategy = config.get(
        "blocking",
        {}
    )

    max_candidates = int(
        strategy.get(
            "max_candidates_per_entity",
            50,
        )
    )

    if max_candidates <= 0:
        raise ValueError(
            "max_candidates_per_entity must be positive."
        )

    if candidates.empty:
        return pd.DataFrame(
            columns=FINAL_COLUMNS
        )

    capped = (
        candidates
        .sort_values(
            by=[
                "source1_entity_id",
                "candidate_entity_id",
            ]
        )
        .groupby(
            "source1_entity_id",
            sort=False,
            group_keys=False,
        )
        .head(max_candidates)
        .reset_index(drop=True)
    )

    print(
        f"\n[Cap] Maximum candidates per S1: "
        f"{max_candidates}"
    )

    print(
        f"[Cap] Candidate pairs after cap: "
        f"{len(capped):,}"
    )

    return capped


def generate_final_candidates(
    source1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Complete A+B+C candidate-generation pipeline.

    Returns the final M3-facing candidate pairs.
    """

    union = generate_union_candidates(
        source1_df,
        target_df,
        config,
    )

    final_candidates = apply_candidate_cap(
        union,
        config,
    )

    return final_candidates


def write_candidate_pairs(
    candidates: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Write final candidate pairs in TSV format.
    """

    candidates = candidates[FINAL_COLUMNS].copy()

    candidates.to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    print(
        f"\n[Output] Wrote candidate pairs to: "
        f"{output_path}"
    )

    print(
        f"[Output] Rows: {len(candidates):,}"
    )
