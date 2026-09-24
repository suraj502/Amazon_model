"""Define model training and entity-level cross-validation contracts."""

from typing import Any

import pandas as pd

from src.utils.config_loader import CONFIG, get_config_value

MODEL_TYPE = get_config_value(CONFIG, "model", "type")


def train_model(features: pd.DataFrame, labels: pd.Series, config: dict[str, Any] = CONFIG) -> Any:
    """Train a configured model using entity-aware splits, without implementation."""
    raise NotImplementedError
