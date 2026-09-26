"""Define union and deduplication of candidate pairs from blocking strategies."""

from typing import Any

import pandas as pd

from src.utils.config_loader import CONFIG, get_config_value

ID_COLUMN = get_config_value(CONFIG, "schema", "id_column")


def union_candidates(candidate_frames: list[pd.DataFrame], config: dict[str, Any] = CONFIG) -> pd.DataFrame:
    """Union candidate frames while preserving source and target IDs."""
    del config
    required = ["source1_entity_id", "candidate_entity_id"]
    if not candidate_frames:
        return pd.DataFrame(columns=required)
    normalized: list[pd.DataFrame] = []
    for frame in candidate_frames:
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError("Candidate frame missing columns: " + ", ".join(sorted(missing)))
        normalized.append(frame[required].astype("string"))
    return (
        pd.concat(normalized, ignore_index=True)
        .drop_duplicates(required)
        .sort_values(required, kind="stable")
        .reset_index(drop=True)
    )
