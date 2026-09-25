"""Define schema and identifier validation for competition submissions."""

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config_loader import CONFIG, get_config_value

OUTPUT_COLUMNS = get_config_value(CONFIG, "submission", "output_columns", "matching_results")


def _ids(value: Any) -> set[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()
    return {item.strip() for item in str(value).replace(",", "|").split("|") if item.strip()}


def validate_submission(path: Path, config: dict[str, Any] = CONFIG) -> bool:
    """Validate the organizer-facing file and its candidate-set contract."""
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if list(frame.columns) != list(OUTPUT_COLUMNS):
        raise ValueError(f"Expected columns {OUTPUT_COLUMNS}, found {list(frame.columns)}")
    source_column, matched_column = OUTPUT_COLUMNS
    if frame[source_column].eq("").any() or frame[source_column].duplicated().any():
        raise ValueError("Every Source1 entity must appear exactly once with a non-empty ID")
    if frame[matched_column].isna().any():
        raise ValueError("Matched entity sets must use an empty string for no-match entities")

    candidate_path = Path(get_config_value(config, "paths", "candidate_pairs_out"))
    if candidate_path.exists():
        candidate_columns = get_config_value(config, "submission", "output_columns", "candidate_pairs")
        candidates = pd.read_csv(candidate_path, sep="\t", dtype=str, keep_default_na=False)
        if list(candidates.columns) != list(candidate_columns):
            raise ValueError("Candidate-pair output has an invalid schema")
        candidate_map = dict(zip(candidates[candidate_columns[0]], candidates[candidate_columns[1]], strict=True))
        for row in frame.itertuples(index=False):
            source_id = getattr(row, source_column)
            if source_id not in candidate_map:
                raise ValueError(f"Missing candidate set for {source_id}")
            matched_ids = _ids(getattr(row, matched_column))
            if not matched_ids.issubset(_ids(candidate_map[source_id])):
                raise ValueError(f"Matched IDs are outside candidates for {source_id}")
    return True
