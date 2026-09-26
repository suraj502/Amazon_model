"""Leakage-aware character n-gram TF-IDF pair features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.config_loader import CONFIG, get_config_value


PAIR_ID_COLUMNS = (
    "source1_entity_id",
    "candidate_entity_id",
    "source_type",
)
TEXT_COLUMNS = {
    "name": ("name_basic_norm_s1", "name_basic_norm_s2"),
    "address": ("address_basic_norm_s1", "address_basic_norm_s2"),
}


def _texts(frame: pd.DataFrame, column: str) -> pd.Series:
    return (
        frame.get(column, pd.Series("", index=frame.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )


def _vectorizer_config(config: dict[str, Any]) -> dict[str, Any]:
    settings = get_config_value(config, "features", "tfidf")
    ngram_range = tuple(settings.get("ngram_range", (2, 4)))
    if len(ngram_range) != 2 or ngram_range[0] < 1 or ngram_range[1] < ngram_range[0]:
        raise ValueError(f"Invalid TF-IDF ngram_range: {ngram_range}")
    max_features = settings.get("max_features", 30000)
    return {
        "analyzer": "char",
        "ngram_range": ngram_range,
        "max_features": max_features,
        "dtype": np.float32,
    }


def _fit_vectorizer(texts: pd.Series, settings: dict[str, Any]):
    usable = texts[texts.str.len() >= settings["ngram_range"][0]]
    if usable.empty:
        return None
    vectorizer = TfidfVectorizer(**settings)
    try:
        vectorizer.fit(usable.tolist())
    except ValueError as error:
        if "empty vocabulary" not in str(error).lower():
            raise
        return None
    return vectorizer


def fit_tfidf_vectorizers(
    pairs: pd.DataFrame,
    config: dict[str, Any] = CONFIG,
    *,
    fit_rows: list[int] | np.ndarray | pd.Index | None = None,
) -> dict[str, TfidfVectorizer | None]:
    """Fit one vectorizer per view, optionally using only selected row positions."""
    settings = _vectorizer_config(config)
    fit_frame = pairs if fit_rows is None else pairs.iloc[list(fit_rows)]
    fitted: dict[str, TfidfVectorizer | None] = {}
    for view, (left_column, right_column) in TEXT_COLUMNS.items():
        corpus = pd.concat(
            [_texts(fit_frame, left_column), _texts(fit_frame, right_column)],
            ignore_index=True,
        )
        fitted[view] = _fit_vectorizer(corpus, settings)
    return fitted


def _paired_cosine(
    left: pd.Series,
    right: pd.Series,
    vectorizer: TfidfVectorizer | None,
) -> np.ndarray:
    if vectorizer is None:
        return np.zeros(len(left), dtype=np.float32)
    left_matrix = vectorizer.transform(left.tolist())
    right_matrix = vectorizer.transform(right.tolist())
    return np.asarray(left_matrix.multiply(right_matrix).sum(axis=1)).ravel().astype(
        np.float32
    )


def build_tfidf_features(
    pairs: pd.DataFrame,
    config: dict[str, Any] = CONFIG,
    *,
    fit_rows: list[int] | np.ndarray | pd.Index | None = None,
    vectorizers: dict[str, TfidfVectorizer | None] | None = None,
) -> pd.DataFrame:
    """Build pairwise cosine features; fit statistics exclude rows outside fit_rows."""
    df = pairs.reset_index(drop=True)
    fitted = vectorizers or fit_tfidf_vectorizers(
        df,
        config,
        fit_rows=fit_rows,
    )
    features = pd.DataFrame(index=df.index)
    for view, (left_column, right_column) in TEXT_COLUMNS.items():
        left = _texts(df, left_column)
        right = _texts(df, right_column)
        features[f"{view}_tfidf_cosine"] = _paired_cosine(
            left,
            right,
            fitted.get(view),
        )
    if not np.isfinite(features.to_numpy(dtype=np.float32)).all():
        raise ValueError("TF-IDF feature generation produced NaN or infinite values")
    id_columns = [column for column in PAIR_ID_COLUMNS if column in df.columns]
    return pd.concat(
        [df[id_columns], features.astype(np.float32)],
        axis=1,
    )
