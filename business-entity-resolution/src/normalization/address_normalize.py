"""Define the address normalization stage for multi-country records."""

from typing import Any

import pandas as pd

from src.utils.config_loader import CONFIG, get_config_value

ADDRESS_COLUMN = get_config_value(CONFIG, "schema", "address_column")


def normalize_addresses(frame: pd.DataFrame, config: dict[str, Any] = CONFIG) -> pd.DataFrame:
    """Return address-normalized records keyed by entity ID."""
    raise NotImplementedError
