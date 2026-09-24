"""Define embedding feature extraction for candidate pairs."""

from typing import Any

import pandas as pd

from src.utils.config_loader import CONFIG, get_config_value

EMBEDDING_MODEL = get_config_value(CONFIG, "features", "embedding", "model_name")


def build_embedding_features(pairs: pd.DataFrame, config: dict[str, Any] = CONFIG) -> pd.DataFrame:
    """Build embedding-derived features keyed by stable entity IDs."""
    raise NotImplementedError
