"""Define entity-level macro F0.5 evaluation contracts."""

from typing import Any

import pandas as pd

from src.utils.config_loader import CONFIG, get_config_value

METRIC = get_config_value(CONFIG, "evaluation", "metric")


def score_f05(predictions: pd.DataFrame, ground_truth: pd.DataFrame, config: dict[str, Any] = CONFIG) -> float:
    """Compute the configured precision-weighted score per source entity."""
    raise NotImplementedError
