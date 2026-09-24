"""Define string similarity feature construction for candidate pairs."""

from typing import Any

import pandas as pd

from src.utils.config_loader import CONFIG, get_config_value

NAME_COLUMN = get_config_value(CONFIG, "schema", "name_column")


def build_string_features(pairs: pd.DataFrame, config: dict[str, Any] = CONFIG) -> pd.DataFrame:
    """Build typed string feature columns keyed by candidate IDs."""
    raise NotImplementedError
