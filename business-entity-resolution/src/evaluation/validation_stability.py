"""Define multi-seed and public-versus-local stability tracking contracts."""

from typing import Any

import pandas as pd

from src.utils.config_loader import CONFIG, get_config_value

N_SEEDS = get_config_value(CONFIG, "validation_stability", "n_seeds")


def track_validation_stability(results: pd.DataFrame, config: dict[str, Any] = CONFIG) -> pd.DataFrame:
    """Summarize configured validation stability experiments."""
    raise NotImplementedError
