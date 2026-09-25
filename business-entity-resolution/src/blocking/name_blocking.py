"""Define name-based candidate generation for source pairs."""

from typing import Any

import pandas as pd

from src.utils.config_loader import CONFIG, get_config_value

ID_COLUMN = get_config_value(CONFIG, "schema", "id_column")
NAME_KEY_COLUMN = "name_alnum_norm"


def block_by_name(
    source1: pd.DataFrame,
    target: pd.DataFrame,
    config: dict[str, Any] = CONFIG,
) -> pd.DataFrame:
    """
    Generate exact normalized-name candidate pairs.

    For every source1 entity, return target entities whose
    `name_alnum_norm` value is exactly the same.

    Empty normalized names are ignored.
    Country is intentionally not used as a hard filter.
    """
    required_source1 = {ID_COLUMN, NAME_KEY_COLUMN}
    required_target = {ID_COLUMN, NAME_KEY_COLUMN}

    missing_source1 = required_source1 - set(source1.columns)
    missing_target = required_target - set(target.columns)

    if missing_source1:
        raise ValueError(
            f"source1 is missing required columns: {sorted(missing_source1)}"
        )

    if missing_target:
        raise ValueError(
            f"target is missing required columns: {sorted(missing_target)}"
        )

    # Keep only the columns needed for this blocker.
    target_view = target[[ID_COLUMN, NAME_KEY_COLUMN]].copy()

    # Empty normalized names provide no useful blocking signal.
    target_view = target_view[
        target_view[NAME_KEY_COLUMN].notna()
        & target_view[NAME_KEY_COLUMN].astype(str).str.strip().ne("")
    ]

    source_view = source1[[ID_COLUMN, NAME_KEY_COLUMN]].copy()

    source_view = source_view[
        source_view[NAME_KEY_COLUMN].notna()
        & source_view[NAME_KEY_COLUMN].astype(str).str.strip().ne("")
    ]

    # Build an inverted index:
    # normalized name -> all target entity IDs having that name.
    target_index = (
        target_view.groupby(NAME_KEY_COLUMN, sort=False)[ID_COLUMN]
        .agg(list)
        .to_dict()
    )

    pairs: list[tuple[str, str]] = []

    for source_id, name_key in source_view[
        [ID_COLUMN, NAME_KEY_COLUMN]
    ].itertuples(index=False, name=None):
        target_ids = target_index.get(name_key, [])

        for target_id in target_ids:
            pairs.append((str(source_id), str(target_id)))

    return pd.DataFrame(
        pairs,
        columns=["source1_entity_id", "matched_entity_id"],
    ).drop_duplicates(
        subset=["source1_entity_id", "matched_entity_id"]
    ).reset_index(drop=True)