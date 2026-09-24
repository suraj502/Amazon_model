"""Provide ID-preserving tabular and parquet I/O helpers."""

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config_loader import CONFIG, get_config_value

ID_COLUMN = get_config_value(CONFIG, "schema", "id_column")


def read_tsv(path: Path, *, dtype: dict[str, str] | None = None) -> pd.DataFrame:
    """Read a TSV while preserving identifier values and declared dtypes."""
    raise NotImplementedError


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    """Write a TSV keyed by the configured identifier column."""
    raise NotImplementedError


def read_parquet(path: Path, *, columns: list[str] | None = None) -> pd.DataFrame:
    """Read parquet data with optional column selection."""
    raise NotImplementedError


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write a dataframe to parquet without changing its row identifiers."""
    raise NotImplementedError
