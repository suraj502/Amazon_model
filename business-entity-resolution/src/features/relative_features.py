"""Define candidate-relative rank, gap, and count feature contracts."""

from typing import Any

import pandas as pd
import numpy as np

from src.utils.config_loader import CONFIG, get_config_value


RELATIVE_FIELDS = get_config_value(
    CONFIG,
    "features",
    "relative_features",
    "fields"
)


def build_relative_features(
    pairs: pd.DataFrame,
    config: dict[str, Any] = CONFIG
) -> pd.DataFrame:
    """
    Build candidate relative features.

    Required columns:
    source entity id
    candidate score
    """


    df = pairs.copy()


    # Detect columns

    source_id_col = (
        "source1_entity_id"
        if "source1_entity_id" in df.columns
        else "entity_id"
    )


    score_col = None


    possible_scores = [

        "candidate_score",

        "blocking_score",

        "similarity_score",

        "score"

    ]


    for col in possible_scores:

        if col in df.columns:

            score_col = col
            break



    if score_col is None:

        raise ValueError(
            "No candidate score column found"
        )



    # Sort candidates by score

    df = df.sort_values(

        [
            source_id_col,
            score_col

        ],

        ascending=[
            True,
            False
        ]

    )



    # Candidate rank

    df["candidate_rank"] = (

        df.groupby(
            source_id_col
        )
        .cumcount()
        + 1

    )



    # Number of candidates per entity

    candidate_counts = (

        df.groupby(
            source_id_col
        )[score_col]
        .transform(
            "count"
        )

    )


    df["number_of_candidates"] = (
        candidate_counts
    )



    # Top score

    top_score = (

        df.groupby(
            source_id_col
        )[score_col]
        .transform(
            "max"
        )

    )


    df["top_score"] = (
        top_score
    )



    # Gap from top candidate

    df["score_gap_from_top"] = (

        df["top_score"]
        -
        df[score_col]

    )



    # Second highest score

    second_score = (

        df.groupby(
            source_id_col
        )[score_col]
        .transform(
            lambda x:
            x.nlargest(2).iloc[-1]
            if len(x) > 1
            else 0
        )

    )


    df["score_gap_from_second"] = (

        second_score
        -
        df[score_col]

    )



    # Candidates above thresholds


    df["number_above_0_5"] = (

        df.groupby(
            source_id_col
        )[score_col]
        .transform(
            lambda x:
            (x > 0.5).sum()
        )

    )


    df["number_above_0_7"] = (

        df.groupby(
            source_id_col
        )[score_col]
        .transform(
            lambda x:
            (x > 0.7).sum()
        )

    )



    output_columns = [

        source_id_col,

        "candidate_rank",

        "top_score",

        "score_gap_from_top",

        "score_gap_from_second",

        "number_of_candidates",

        "number_above_0_5",

        "number_above_0_7"

    ]



    output = df[
        output_columns
    ].copy()



    numeric_columns = [

        "candidate_rank",

        "top_score",

        "score_gap_from_top",

        "score_gap_from_second",

        "number_of_candidates",

        "number_above_0_5",

        "number_above_0_7"

    ]


    output[numeric_columns] = (

        output[numeric_columns]
        .astype(
            np.float32
        )

    )


    return output