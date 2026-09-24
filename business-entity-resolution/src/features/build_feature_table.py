"""Define orchestration for assembling the candidate feature table."""

from typing import Any

import pandas as pd

from src.utils.config_loader import CONFIG, get_config_value

FEATURE_DTYPE = get_config_value(CONFIG, "features", "dtype")


def build_feature_table(pairs: pd.DataFrame, config: dict[str, Any] = CONFIG) -> pd.DataFrame:
    """Assemble model-ready features without changing candidate identifiers."""
    raise NotImplementedError
