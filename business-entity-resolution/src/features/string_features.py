"""Build string, country, phonetic, and address-component pair features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler, Levenshtein

from src.utils.config_loader import CONFIG, get_config_value


PAIR_ID_COLUMNS = (
    "source1_entity_id",
    "candidate_entity_id",
    "source_type",
)

_COMPONENT_ALIASES = {
    "postal": ("postal_code", "postcode", "zip", "pin", "pincode"),
    "city": ("city", "town"),
    "state": ("state", "province", "region"),
}


def _text(value: Any) -> str:
    """Return normalized text, treating missing values as empty."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().casefold()


def jaccard_similarity(a: Any, b: Any) -> float:
    """Compute token-set Jaccard similarity."""
    tokens_a = set(_text(a).split())
    tokens_b = set(_text(b).split())
    union = tokens_a | tokens_b
    return float(len(tokens_a & tokens_b) / len(union)) if union else 0.0


def levenshtein_similarity(a: Any, b: Any) -> float:
    """Compute normalized edit similarity."""
    text_a = _text(a)
    text_b = _text(b)
    if not text_a and not text_b:
        return 0.0
    return float(1.0 - Levenshtein.normalized_distance(text_a, text_b))


def jaro_winkler_similarity(a: Any, b: Any) -> float:
    """Compute Jaro-Winkler similarity."""
    text_a = _text(a)
    text_b = _text(b)
    if not text_a and not text_b:
        return 0.0
    return float(JaroWinkler.normalized_similarity(text_a, text_b))


def _soundex(value: str) -> str:
    """Return a basic Latin Soundex code for already-transliterated text."""
    letters = [character for character in value.upper() if "A" <= character <= "Z"]
    if not letters:
        return ""

    groups = {
        **dict.fromkeys("BFPV", "1"),
        **dict.fromkeys("CGJKQSXZ", "2"),
        **dict.fromkeys("DT", "3"),
        **dict.fromkeys("L", "4"),
        **dict.fromkeys("MN", "5"),
        **dict.fromkeys("R", "6"),
    }
    first = letters[0]
    encoded = [first]
    previous = groups.get(first, "")
    for letter in letters[1:]:
        code = groups.get(letter, "")
        if code and code != previous:
            encoded.append(code)
        previous = code
    return ("".join(encoded) + "000")[:4]


def phonetic_match(a: Any, b: Any) -> int:
    """Compare Soundex codes of the first name tokens."""
    tokens_a = _text(a).split()
    tokens_b = _text(b).split()
    if not tokens_a or not tokens_b:
        return 0
    code_a = _soundex(tokens_a[0])
    code_b = _soundex(tokens_b[0])
    return int(bool(code_a and code_a == code_b))


def _first_present(row: pd.Series, column: str) -> str:
    return _text(row.get(column, ""))


def _component_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare configured address components when normalized views provide them."""
    result = pd.DataFrame(index=frame.index)
    comparisons: list[pd.Series] = []
    for output_name, aliases in _COMPONENT_ALIASES.items():
        left_col = next(
            (f"{alias}_s1" for alias in aliases if f"{alias}_s1" in frame.columns),
            None,
        )
        right_col = next(
            (f"{alias}_s2" for alias in aliases if f"{alias}_s2" in frame.columns),
            None,
        )
        if left_col is None or right_col is None:
            continue
        left = frame[left_col].map(_text)
        right = frame[right_col].map(_text)
        valid = left.ne("") & right.ne("")
        matched = valid & left.eq(right)
        result[f"address_{output_name}_match"] = matched.astype(np.float32)
        comparisons.append(matched.astype(np.float32).where(valid, np.nan))

    if comparisons:
        result["address_component_overlap"] = (
            pd.concat(comparisons, axis=1).mean(axis=1).fillna(0.0)
        )
    else:
        result["address_component_overlap"] = np.float32(0.0)
    return result


def _pair_ids(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in PAIR_ID_COLUMNS if column in frame.columns]
    return frame[columns].copy()


def build_string_features(
    pairs: pd.DataFrame,
    config: dict[str, Any] = CONFIG,
) -> pd.DataFrame:
    """Build string similarities from the normalized multi-view pair columns."""
    del config
    required = (
        "name_basic_norm_s1",
        "name_basic_norm_s2",
        "name_transliterated_s1",
        "name_transliterated_s2",
        "address_basic_norm_s1",
        "address_basic_norm_s2",
    )
    missing = [column for column in required if column not in pairs.columns]
    if missing:
        raise ValueError(
            "Candidate pairs are missing normalized feature columns: "
            + ", ".join(missing)
        )

    name_left = pairs["name_basic_norm_s1"].map(_text)
    name_right = pairs["name_basic_norm_s2"].map(_text)
    trans_left = pairs["name_transliterated_s1"].map(_text)
    trans_right = pairs["name_transliterated_s2"].map(_text)
    address_left = pairs["address_basic_norm_s1"].map(_text)
    address_right = pairs["address_basic_norm_s2"].map(_text)

    features = pd.DataFrame(index=pairs.index)
    for prefix, left, right in (
        ("name", name_left, name_right),
        ("address", address_left, address_right),
    ):
        features[f"{prefix}_jaccard"] = [
            jaccard_similarity(a, b) for a, b in zip(left, right)
        ]
        features[f"{prefix}_levenshtein"] = [
            levenshtein_similarity(a, b) for a, b in zip(left, right)
        ]
        features[f"{prefix}_jaro_winkler"] = [
            jaro_winkler_similarity(a, b) for a, b in zip(left, right)
        ]

    features["phonetic_match"] = [
        phonetic_match(a, b) for a, b in zip(trans_left, trans_right)
    ]
    features["country_match"] = [
        float(bool(a) and a == b)
        for a, b in zip(
            pairs.get("country_s1", pd.Series("", index=pairs.index)).map(_text),
            pairs.get("country_s2", pd.Series("", index=pairs.index)).map(_text),
        )
    ]
    features = pd.concat([features, _component_features(pairs)], axis=1)

    if "source_type" in pairs.columns:
        features["source_is_s3"] = pairs["source_type"].map(
            lambda value: float(str(value).strip().upper() in {"S3", "SOURCE3"})
        )
    else:
        features["source_is_s3"] = np.float32(0.0)

    numeric = features.to_numpy(dtype=np.float32)
    if not np.isfinite(numeric).all():
        raise ValueError("String feature generation produced NaN or infinite values")
    features = features.astype(np.float32)
    return pd.concat([_pair_ids(pairs), features], axis=1)
