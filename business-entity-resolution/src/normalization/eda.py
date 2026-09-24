"""Define exploratory data checks for source quality and schema drift."""

from typing import Any

import pandas as pd

from src.utils.config_loader import CONFIG, get_config_value

ID_COLUMN = get_config_value(CONFIG, "schema", "id_column")


def profile_sources(frames: dict[str, pd.DataFrame], config: dict[str, Any] = CONFIG) -> dict[str, Any]:
    """Produce a source profile keyed by source names and entity IDs."""
    raise NotImplementedError
