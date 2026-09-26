"""Batched, candidate-pair-only semantic similarity features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from src.utils.config_loader import CONFIG, get_config_value


PAIR_ID_COLUMNS = (
    "source1_entity_id",
    "candidate_entity_id",
    "source_type",
)
_MODEL_CACHE: dict[tuple[str, str], SentenceTransformer] = {}


def load_embedding_model(
    config: dict[str, Any] = CONFIG,
) -> SentenceTransformer:
    """Load the configured model; CPU is the safe default and needs no GPU."""
    settings = get_config_value(config, "features", "embedding")
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers is required when embedding features are enabled"
        )
    model_name = settings.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    device = settings.get("device", "cpu")
    key = (model_name, device)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = SentenceTransformer(model_name, device=device)
    return _MODEL_CACHE[key]


def generate_embeddings(
    texts: list[str] | pd.Series,
    model: SentenceTransformer,
    *,
    batch_size: int,
) -> np.ndarray:
    """Encode a batch as normalized float32 vectors."""
    values = list(texts)
    if not values:
        return np.empty((0, 0), dtype=np.float32)
    embeddings = model.encode(
        values,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    result = np.asarray(embeddings, dtype=np.float32)
    if result.ndim != 2 or result.shape[0] != len(values):
        raise ValueError("Embedding model returned an unexpected output shape")
    if not np.isfinite(result).all():
        raise ValueError("Embedding model returned NaN or infinite values")
    return result


def _pair_cosine(
    left: pd.Series,
    right: pd.Series,
    model: SentenceTransformer,
    batch_size: int,
) -> np.ndarray:
    left_values = left.fillna("").astype(str).str.strip().tolist()
    right_values = right.fillna("").astype(str).str.strip().tolist()
    unique_texts = list(
        dict.fromkeys(text for text in left_values + right_values if text)
    )
    encoded = generate_embeddings(unique_texts, model, batch_size=batch_size)
    embedding_by_text = dict(zip(unique_texts, encoded))
    result = np.zeros(len(left_values), dtype=np.float32)
    for index, (left_text, right_text) in enumerate(zip(left_values, right_values)):
        if left_text and right_text:
            result[index] = np.dot(
                embedding_by_text[left_text],
                embedding_by_text[right_text],
            )
    return result


def build_embedding_features(
    pairs: pd.DataFrame,
    config: dict[str, Any] = CONFIG,
) -> pd.DataFrame:
    """Compute cosine features without constructing a quadratic pair matrix."""
    settings = get_config_value(config, "features", "embedding")
    batch_size = int(settings.get("batch_size", 32))
    if batch_size < 1:
        raise ValueError("Embedding batch_size must be a positive integer")
    df = pairs.reset_index(drop=True)
    model = load_embedding_model(config)

    name_left = df.get("name_basic_norm_s1", pd.Series("", index=df.index))
    name_right = df.get("name_basic_norm_s2", pd.Series("", index=df.index))
    address_left = df.get("address_basic_norm_s1", pd.Series("", index=df.index))
    address_right = df.get("address_basic_norm_s2", pd.Series("", index=df.index))
    combined_left = name_left.fillna("").astype(str) + " " + address_left.fillna("").astype(str)
    combined_right = name_right.fillna("").astype(str) + " " + address_right.fillna("").astype(str)

    features = pd.DataFrame(
        {
            "name_embedding_cosine": _pair_cosine(
                name_left, name_right, model, batch_size
            ),
            "address_embedding_cosine": _pair_cosine(
                address_left, address_right, model, batch_size
            ),
            "combined_embedding_cosine": _pair_cosine(
                combined_left, combined_right, model, batch_size
            ),
        },
        index=df.index,
        dtype=np.float32,
    )
    if not np.isfinite(features.to_numpy(dtype=np.float32)).all():
        raise ValueError("Embedding features contain NaN or infinite values")
    id_columns = [column for column in PAIR_ID_COLUMNS if column in df.columns]
    return pd.concat([df[id_columns], features], axis=1)
