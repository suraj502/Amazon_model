"""Define threshold selection against the configured entity-level metric."""

from typing import Any

import pandas as pd

from src.utils.config_loader import CONFIG, get_config_value

SEARCH_RANGE = get_config_value(CONFIG, "threshold", "search_range")


def find_best_threshold(scores: pd.DataFrame, ground_truth: pd.DataFrame, config: dict[str, Any] = CONFIG) -> float:
    """Search configured thresholds without assuming a fixed country set."""
    raise NotImplementedError
