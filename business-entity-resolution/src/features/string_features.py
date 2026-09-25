"""Define string similarity feature construction for candidate pairs."""

from typing import Any

import pandas as pd
from rapidfuzz.fuzz import ratio, token_set_ratio
from rapidfuzz.distance import Levenshtein

from src.utils.config_loader import CONFIG, get_config_value


NAME_COLUMN = get_config_value(
    CONFIG,
    "schema",
    "name_column"
)

ADDRESS_COLUMN = get_config_value(
    CONFIG,
    "schema",
    "address_column"
)

COUNTRY_COLUMN = get_config_value(
    CONFIG,
    "schema",
    "country_column"
)

ID_COLUMN = get_config_value(
    CONFIG,
    "schema",
    "id_column"
)



def jaccard_similarity(a, b):

    if not isinstance(a, str) or not isinstance(b, str):
        return 0.0

    a = set(a.split())
    b = set(b.split())

    if not a and not b:
        return 0.0

    return len(a & b) / len(a | b)



def levenshtein_similarity(a, b):

    if not isinstance(a, str) or not isinstance(b, str):
        return 0.0

    distance = Levenshtein.distance(
        a,
        b
    )

    length = max(
        len(a),
        len(b)
    )

    if length == 0:
        return 1.0

    return 1 - (
        distance / length
    )



def jaro_winkler_similarity(a, b):

    if not isinstance(a, str) or not isinstance(b, str):
        return 0.0

    return ratio(
        a,
        b
    ) / 100.0



def build_string_features(
    pairs: pd.DataFrame,
    config: dict[str, Any] = CONFIG
) -> pd.DataFrame:
    """
    Build string similarity features.
    """


    df = pairs.copy()


    features = pd.DataFrame(
        index=df.index
    )


    # Name similarities

    features["name_jaccard"] = df.apply(
        lambda x: jaccard_similarity(
            x.get("name_basic_norm_s1", ""),
            x.get("name_basic_norm_s2", "")
        ),
        axis=1
    )


    features["name_levenshtein"] = df.apply(
        lambda x: levenshtein_similarity(
            x.get("name_basic_norm_s1", ""),
            x.get("name_basic_norm_s2", "")
        ),
        axis=1
    )


    features["name_jaro_winkler"] = df.apply(
        lambda x: jaro_winkler_similarity(
            x.get("name_transliterated_s1", ""),
            x.get("name_transliterated_s2", "")
        ),
        axis=1
    )



    # Address similarities

    features["address_jaccard"] = df.apply(
        lambda x: jaccard_similarity(
            x.get("address_basic_norm_s1", ""),
            x.get("address_basic_norm_s2", "")
        ),
        axis=1
    )


    features["address_levenshtein"] = df.apply(
        lambda x: levenshtein_similarity(
            x.get("address_basic_norm_s1", ""),
            x.get("address_basic_norm_s2", "")
        ),
        axis=1
    )


    features["address_jaro_winkler"] = df.apply(
        lambda x: jaro_winkler_similarity(
            x.get("address_basic_norm_s1", ""),
            x.get("address_basic_norm_s2", "")
        ),
        axis=1
    )



    # Country match

    features["country_match"] = df.apply(
        lambda x: int(
            x.get("country_s1", "")
            ==
            x.get("country_s2", "")
        ),
        axis=1
    )



    # Phonetic placeholder
    # will be replaced with metaphone later

    features["phonetic_match"] = df.apply(
        lambda x: int(
            x.get("name_transliterated_s1", "")
            ==
            x.get("name_transliterated_s2", "")
        ),
        axis=1
    )


    return pd.concat(
        [
            df[[ID_COLUMN]]
            if ID_COLUMN in df.columns
            else pd.DataFrame(index=df.index),

            features
        ],
        axis=1
    )