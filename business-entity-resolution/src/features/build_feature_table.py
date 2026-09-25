"""Define orchestration for assembling the candidate feature table."""

from typing import Any

import pandas as pd
import numpy as np

from src.utils.config_loader import CONFIG, get_config_value

from src.features.string_features import build_string_features
from src.features.tfidf_features import build_tfidf_features
from src.features.embedding_features import build_embedding_features
from src.features.relative_features import build_relative_features


FEATURE_DTYPE = get_config_value(
    CONFIG,
    "features",
    "dtype"
)



def build_feature_table(
    pairs: pd.DataFrame,
    config: dict[str, Any] = CONFIG
) -> pd.DataFrame:
    """
    Assemble complete model-ready feature table.
    """



    df = pairs.copy()



    # Keep identifiers

    id_columns = [

        col for col in [

            "source1_entity_id",

            "candidate_entity_id",

            "entity_id",

            "source_type"

        ]

        if col in df.columns

    ]



    base = df[id_columns].copy()



    # -----------------------------
    # String features
    # -----------------------------

    string_features = build_string_features(
        df,
        config
    )



    # -----------------------------
    # TF-IDF features
    # -----------------------------

    tfidf_features = build_tfidf_features(
        df,
        config
    )



    # -----------------------------
    # Embedding features
    # -----------------------------

    embedding_features = build_embedding_features(
        df,
        config
    )



    # -----------------------------
    # Relative features
    # -----------------------------

    relative_features = build_relative_features(
        df,
        config
    )



    # -----------------------------
    # Merge all features
    # -----------------------------

    feature_table = base.copy()



    for feature_df in [

        string_features,

        tfidf_features,

        embedding_features,

        relative_features

    ]:


        feature_columns = [

            col for col in feature_df.columns

            if col not in feature_table.columns

        ]


        feature_table = pd.concat(

            [

                feature_table,

                feature_df[feature_columns]

            ],

            axis=1

        )



    # -----------------------------
    # Add label if available
    # -----------------------------


    if "label" in df.columns:

        feature_table["label"] = df["label"]



    # -----------------------------
    # Convert numerical columns
    # -----------------------------


    numeric_columns = feature_table.select_dtypes(

        include=[
            "float64",
            "int64"
        ]

    ).columns



    feature_table[numeric_columns] = (

        feature_table[numeric_columns]

        .astype(
            np.float32
        )

    )



    # -----------------------------
    # Validation checks
    # -----------------------------


    if feature_table.isna().any().any():

        raise ValueError(
            "Feature table contains NaN values"
        )



    if np.isinf(
        feature_table.select_dtypes(
            include=[np.number]
        )
    ).any().any():

        raise ValueError(
            "Feature table contains infinity values"
        )



    return feature_table