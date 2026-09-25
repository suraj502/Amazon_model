"""Define TF-IDF feature generation for candidate records."""

from typing import Any

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.config_loader import CONFIG, get_config_value


MAX_FEATURES = get_config_value(
    CONFIG,
    "features",
    "tfidf",
    "max_features"
)


def build_tfidf_features(
    pairs: pd.DataFrame,
    config: dict[str, Any] = CONFIG
) -> pd.DataFrame:
    """
    Build character n-gram TF-IDF cosine similarity features.
    """


    df = pairs.copy()


    features = pd.DataFrame(
        index=df.index
    )


    name_text_1 = (
        df.get(
            "name_basic_norm_s1",
            pd.Series("", index=df.index)
        )
        .fillna("")
        .astype(str)
    )


    name_text_2 = (
        df.get(
            "name_basic_norm_s2",
            pd.Series("", index=df.index)
        )
        .fillna("")
        .astype(str)
    )


    address_text_1 = (
        df.get(
            "address_basic_norm_s1",
            pd.Series("", index=df.index)
        )
        .fillna("")
        .astype(str)
    )


    address_text_2 = (
        df.get(
            "address_basic_norm_s2",
            pd.Series("", index=df.index)
        )
        .fillna("")
        .astype(str)
    )



    # Name TF-IDF

    name_corpus = pd.concat(
        [
            name_text_1,
            name_text_2
        ],
        ignore_index=True
    )


    name_vectorizer = TfidfVectorizer(

        analyzer="char",

        ngram_range=tuple(
            get_config_value(
                config,
                "features",
                "tfidf",
                "ngram_range"
            )
        ),

        max_features=MAX_FEATURES

    )


    name_matrix = name_vectorizer.fit_transform(
        name_corpus
    )


    name_1_matrix = name_matrix[
        :len(df)
    ]


    name_2_matrix = name_matrix[
        len(df):
    ]



    name_similarity = cosine_similarity(
        name_1_matrix,
        name_2_matrix
    ).diagonal()



    features["name_tfidf_cosine"] = (
        name_similarity
    )



    # Address TF-IDF

    address_corpus = pd.concat(
        [
            address_text_1,
            address_text_2
        ],
        ignore_index=True
    )


    address_vectorizer = TfidfVectorizer(

        analyzer="char",

        ngram_range=tuple(
            get_config_value(
                config,
                "features",
                "tfidf",
                "ngram_range"
            )
        ),

        max_features=MAX_FEATURES

    )


    address_matrix = address_vectorizer.fit_transform(
        address_corpus
    )


    address_1_matrix = address_matrix[
        :len(df)
    ]


    address_2_matrix = address_matrix[
        len(df):
    ]



    address_similarity = cosine_similarity(
        address_1_matrix,
        address_2_matrix
    ).diagonal()



    features["address_tfidf_cosine"] = (
        address_similarity
    )



    return pd.concat(
        [
            df[["entity_id"]]
            if "entity_id" in df.columns
            else pd.DataFrame(index=df.index),

            features.astype(
                np.float32
            )

        ],
        axis=1
    )