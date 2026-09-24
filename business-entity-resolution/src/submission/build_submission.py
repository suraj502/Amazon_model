"""Define submission serialization from ID-based matching results."""

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config_loader import CONFIG, get_config_value

OUTPUT_COLUMNS = get_config_value(CONFIG, "submission", "output_columns", "matching_results")


def build_submission(matches: pd.DataFrame, output_path: Path, config: dict[str, Any] = CONFIG) -> None:
    """Write the configured submission columns keyed by source-one IDs."""
    raise NotImplementedError
