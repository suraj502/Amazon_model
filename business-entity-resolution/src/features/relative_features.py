"""Candidate-set rank and score context features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.config_loader import CONFIG, get_config_value


PAIR_ID_COLUMNS = (
    "source1_entity_id",
    "candidate_entity_id",
    "source_type",
)
RELATIVE_FIELDS = get_config_value(CONFIG, "features", "relative_features", "fields")


def build_relative_features(
    pairs: pd.DataFrame,
    config: dict[str, Any] = CONFIG,
) -> pd.DataFrame:
    """Build per-Source-1-candidate-set ranks, score gaps, and counts."""
    df = pairs.reset_index(drop=True).copy()
    entity_column = next(
        (
            column
            for column in ("source1_entity_id", "entity_id")
            if column in df.columns
        ),
        None,
    )
    if entity_column is None:
        raise ValueError("Relative features require source1_entity_id")

    score_column = next(
        (
            column
            for column in (
                "candidate_score",
                "blocking_score",
                "similarity_score",
                "score",
            )
            if column in df.columns
        ),
        None,
    )
    if score_column is None:
        raise ValueError(
            "Relative features require a candidate score or a derived candidate_score"
        )
    if not pd.api.types.is_numeric_dtype(df[score_column]):
        raise TypeError(f"Candidate score column {score_column!r} must be numeric")

    scores = pd.to_numeric(df[score_column], errors="raise").astype(np.float32)
    if not np.isfinite(scores.to_numpy()).all():
        raise ValueError("Candidate scores contain NaN or infinite values")
    df["_candidate_score"] = scores
    df["_row_order"] = np.arange(len(df), dtype=np.int64)
    df = df.sort_values(
        [entity_column, "_candidate_score", "_row_order"],
        ascending=[True, False, True],
        kind="stable",
    )
    grouped = df.groupby(entity_column, sort=False)["_candidate_score"]
    rank = df.groupby(entity_column, sort=False).cumcount() + 1
    top_score = grouped.transform("max")
    second_score = grouped.transform(
        lambda values: values.nlargest(2).iloc[-1] if len(values) > 1 else 0.0
    )

    output = pd.DataFrame(index=df.index)
    output["candidate_rank"] = rank.astype(np.float32)
    output["top_score"] = top_score.astype(np.float32)
    output["score_gap_from_top"] = (top_score - df["_candidate_score"]).astype(
        np.float32
    )
    output["score_gap_from_second"] = (df["_candidate_score"] - second_score).astype(
        np.float32
    )
    output["number_of_candidates"] = grouped.transform("count").astype(np.float32)
    output["number_above_0_5"] = grouped.transform(
        lambda values: int((values > 0.5).sum())
    ).astype(np.float32)
    output["number_above_0_7"] = grouped.transform(
        lambda values: int((values > 0.7).sum())
    ).astype(np.float32)

    # Restore original row order so feature rows stay aligned with candidate pairs.
    output["_row_order"] = df["_row_order"]
    output = output.sort_values("_row_order", kind="stable").drop(columns="_row_order")
    output.index = pairs.index
    numeric = output.to_numpy(dtype=np.float32)
    if not np.isfinite(numeric).all():
        raise ValueError("Relative feature generation produced NaN or infinite values")
    id_columns = [column for column in PAIR_ID_COLUMNS if column in pairs.columns]
    return pd.concat(
        [pairs[id_columns], output.astype(np.float32)],
        axis=1,
    )
