"""Define address-based candidate generation without scoring candidates."""

from typing import Any

import pandas as pd

from src.utils.config_loader import CONFIG, get_config_value

ID_COLUMN = get_config_value(CONFIG, "schema", "id_column")


def block_by_address(source1: pd.DataFrame, target: pd.DataFrame, config: dict[str, Any] = CONFIG) -> pd.DataFrame:
    """Generate address candidate pairs using configured entity IDs."""
    raise NotImplementedError
