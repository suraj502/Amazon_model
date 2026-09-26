"""Provide ID-preserving tabular and parquet I/O helpers."""

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config_loader import CONFIG, get_config_value

ID_COLUMN = get_config_value(CONFIG, "schema", "id_column")


def read_tsv(path: Path, *, dtype: dict[str, str] | None = None) -> pd.DataFrame:
    """Read a TSV while preserving identifier values and declared dtypes."""
    frame = pd.read_csv(path, sep="\t", dtype=dtype or {ID_COLUMN: "string"})
    if ID_COLUMN in frame.columns:
        frame[ID_COLUMN] = frame[ID_COLUMN].astype("string")
    return frame


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    """Write a TSV keyed by the configured identifier column."""
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    if ID_COLUMN in output.columns:
        output[ID_COLUMN] = output[ID_COLUMN].astype("string")
    output.to_csv(path, sep="\t", index=False)


def read_parquet(path: Path, *, columns: list[str] | None = None) -> pd.DataFrame:
    """Read parquet data with optional column selection."""
    frame = pd.read_parquet(path, columns=columns)
    if ID_COLUMN in frame.columns:
        frame[ID_COLUMN] = frame[ID_COLUMN].astype("string")
    return frame


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write a dataframe to parquet without changing its row identifiers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    if ID_COLUMN in output.columns:
        output[ID_COLUMN] = output[ID_COLUMN].astype("string")
    output.to_parquet(path, index=False)
