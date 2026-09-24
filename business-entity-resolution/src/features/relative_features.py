"""Define candidate-relative rank, gap, and count feature contracts."""

from typing import Any

import pandas as pd

from src.utils.config_loader import CONFIG, get_config_value

RELATIVE_FIELDS = get_config_value(CONFIG, "features", "relative_features", "fields")


def build_relative_features(pairs: pd.DataFrame, config: dict[str, Any] = CONFIG) -> pd.DataFrame:
    """Build relative candidate features keyed by source and target IDs."""
    raise NotImplementedError
