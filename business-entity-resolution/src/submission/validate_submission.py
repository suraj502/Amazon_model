"""Define schema and identifier validation for competition submissions."""

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config_loader import CONFIG, get_config_value

OUTPUT_COLUMNS = get_config_value(CONFIG, "submission", "output_columns", "matching_results")


def validate_submission(path: Path, config: dict[str, Any] = CONFIG) -> bool:
    """Validate submission columns and ID-based row contracts."""
    raise NotImplementedError
