"""Define name-based candidate generation for source pairs."""

from typing import Any

import pandas as pd

from src.utils.config_loader import CONFIG, get_config_value

ID_COLUMN = get_config_value(CONFIG, "schema", "id_column")


def block_by_name(source1: pd.DataFrame, target: pd.DataFrame, config: dict[str, Any] = CONFIG) -> pd.DataFrame:
    """Generate candidate pairs keyed by source entity IDs."""
    raise NotImplementedError
