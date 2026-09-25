"""Define submission serialization from ID-based matching results."""

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config_loader import CONFIG, get_config_value

OUTPUT_COLUMNS = get_config_value(CONFIG, "submission", "output_columns", "matching_results")


def _ids(value: Any) -> set[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()
    if isinstance(value, str):
        return {item.strip() for item in value.replace(",", "|").split("|") if item.strip()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return {str(item).strip() for item in value if str(item).strip()}
    return {str(value).strip()} if str(value).strip() else set()


def _source_column(frame: pd.DataFrame) -> str:
    for name in ("source1_entity_id", "source_entity_id", "entity_id"):
        if name in frame.columns:
            return name
    raise ValueError("Matches must contain a source1 entity ID column")


def _group_ids(frame: pd.DataFrame, source_column: str, value_columns: tuple[str, ...]) -> pd.DataFrame:
    value_column = next((column for column in value_columns if column in frame.columns), None)
    if value_column is None:
        raise ValueError(f"Matches must contain one of: {', '.join(value_columns)}")
    grouped = []
    for source_id, rows in frame.groupby(source_column, sort=False, dropna=False):
        if pd.isna(source_id):
            raise ValueError("Source1 entity IDs cannot be null")
        values: set[str] = set()
        for value in rows[value_column]:
            values.update(_ids(value))
        grouped.append({"source1_entity_id": str(source_id), value_columns[0]: "|".join(sorted(values))})
    return pd.DataFrame(grouped, columns=["source1_entity_id", value_columns[0]])


def build_submission(matches: pd.DataFrame, output_path: Path, config: dict[str, Any] = CONFIG) -> None:
    """Write matching and candidate TSVs, then validate the matching artifact."""
    if matches.empty:
        raise ValueError("Cannot build a submission from an empty match table")
    source_column = _source_column(matches)
    matched = _group_ids(matches, source_column, ("matched_entity_ids", "matched_entity_id", "candidate_entity_id"))
    candidate = _group_ids(matches, source_column, ("candidate_entity_ids", "candidate_entity_id", "matched_entity_id"))
    matched_sets = {row.source1_entity_id: _ids(row.matched_entity_ids) for row in matched.itertuples()}
    candidate_sets = {row.source1_entity_id: _ids(row.candidate_entity_ids) for row in candidate.itertuples()}
    if set(matched_sets) != set(candidate_sets):
        raise ValueError("Matching and candidate outputs must contain the same Source1 entities")
    violations = {
        source_id: sorted(matched_sets[source_id] - candidate_sets[source_id])
        for source_id in matched_sets
        if not matched_sets[source_id].issubset(candidate_sets[source_id])
    }
    if violations:
        raise ValueError(f"Matched IDs are outside candidate sets: {violations}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(output_path, sep="\t", index=False, columns=OUTPUT_COLUMNS)
    candidate_path = Path(get_config_value(config, "paths", "candidate_pairs_out"))
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(candidate_path, sep="\t", index=False, columns=get_config_value(config, "submission", "output_columns", "candidate_pairs"))

    from src.submission.validate_submission import validate_submission

    if not validate_submission(output_path, config):
        raise ValueError(f"Submission failed validation: {output_path}")
