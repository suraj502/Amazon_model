"""Define union and deduplication of candidate pairs from blocking strategies."""

from typing import Any

import pandas as pd

from src.utils.config_loader import CONFIG, get_config_value

ID_COLUMN = get_config_value(CONFIG, "schema", "id_column")


def union_candidates(candidate_frames: list[pd.DataFrame], config: dict[str, Any] = CONFIG) -> pd.DataFrame:
    """Union candidate frames while preserving source and target IDs."""
    raise NotImplementedError
