"""Define the name normalization stage without implementing matching logic."""

from typing import Any

import pandas as pd

from src.utils.config_loader import CONFIG, get_config_value

NAME_COLUMN = get_config_value(CONFIG, "schema", "name_column")


def normalize_names(frame: pd.DataFrame, config: dict[str, Any] = CONFIG) -> pd.DataFrame:
    """Return name-normalized records while retaining their configured IDs."""
    raise NotImplementedError
