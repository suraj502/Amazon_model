"""Define entity-level macro F0.5 evaluation contracts."""

from collections.abc import Iterable
from typing import Any

import pandas as pd

from src.utils.config_loader import CONFIG, get_config_value

METRIC = get_config_value(CONFIG, "evaluation", "metric")


def _split_ids(value: Any) -> set[str]:
    """Normalize a scalar/list match representation into an ID set."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()
    if isinstance(value, str):
        return {item.strip() for item in value.replace(",", "|").split("|") if item.strip()}
    if isinstance(value, Iterable) and not isinstance(value, (bytes, str)):
        return {str(item).strip() for item in value if str(item).strip()}
    return {str(value).strip()} if str(value).strip() else set()


def _as_entity_sets(frame: pd.DataFrame) -> dict[str, set[str]]:
    """Read either row-wise pairs or one row containing a delimited match set."""
    if frame.empty:
        return {}
    source_column = next(
        (column for column in ("source1_entity_id", "source_entity_id", "entity_id") if column in frame),
        None,
    )
    if source_column is None:
        raise ValueError("Expected a source1 entity ID column")
    match_column = next(
        (column for column in ("matched_entity_ids", "matched_entity_id", "candidate_entity_id", "entity_id") if column in frame and column != source_column),
        None,
    )
    if match_column is None:
        raise ValueError("Expected a matched entity ID column")

    result: dict[str, set[str]] = {}
    for row in frame[[source_column, match_column]].itertuples(index=False, name=None):
        source_id, match_value = row
        if pd.isna(source_id):
            continue
        result.setdefault(str(source_id), set()).update(_split_ids(match_value))
    return result


def _f05(precision: float, recall: float) -> float:
    """Return F0.5, guarding the zero-denominator case."""
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta_squared = 0.25
    return (1 + beta_squared) * precision * recall / (beta_squared * precision + recall)


def score_f05(predictions: pd.DataFrame, ground_truth: pd.DataFrame, config: dict[str, Any] = CONFIG) -> float:
    """Compute macro F0.5 over complete predicted and expected match sets.

    An entity with no expected matches receives 1.0 only when no matches are
    predicted. Empty entities remain in the macro average, making singleton
    misses and false-positive singletons visible to the score.
    """
    predicted_sets = _as_entity_sets(predictions)
    expected_sets = _as_entity_sets(ground_truth)
    entity_ids = set(predicted_sets) | set(expected_sets)
    if not entity_ids:
        return 1.0

    scores = []
    for entity_id in entity_ids:
        predicted = predicted_sets.get(entity_id, set())
        expected = expected_sets.get(entity_id, set())
        if not predicted and not expected:
            scores.append(1.0)
            continue
        true_positives = len(predicted & expected)
        precision = true_positives / len(predicted) if predicted else 0.0
        recall = true_positives / len(expected) if expected else 0.0
        scores.append(_f05(precision, recall))
    return float(sum(scores) / len(scores))
