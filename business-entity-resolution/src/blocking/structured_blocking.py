"""Define blocking on configured structured fields such as country."""

from typing import Any

import pandas as pd

from src.utils.config_loader import CONFIG, get_config_value

ID_COLUMN = get_config_value(CONFIG, "schema", "id_column")


def block_by_structure(source1: pd.DataFrame, target: pd.DataFrame, config: dict[str, Any] = CONFIG) -> pd.DataFrame:
    """Generate structured candidate pairs keyed by stable entity IDs."""
    raise NotImplementedError
