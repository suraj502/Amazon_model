"""Define TF-IDF feature generation for candidate records."""

from typing import Any

import pandas as pd

from src.utils.config_loader import CONFIG, get_config_value

MAX_FEATURES = get_config_value(CONFIG, "features", "tfidf", "max_features")


def build_tfidf_features(pairs: pd.DataFrame, config: dict[str, Any] = CONFIG) -> pd.DataFrame:
    """Build TF-IDF features while retaining candidate pair IDs."""
    raise NotImplementedError
