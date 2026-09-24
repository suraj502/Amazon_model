"""Define prediction and match selection contracts for trained models."""

from typing import Any

import pandas as pd

from src.utils.config_loader import CONFIG, get_config_value

ID_COLUMN = get_config_value(CONFIG, "schema", "id_column")


def predict_matches(model: Any, features: pd.DataFrame, config: dict[str, Any] = CONFIG) -> pd.DataFrame:
    """Predict candidate scores while preserving source entity identifiers."""
    raise NotImplementedError
