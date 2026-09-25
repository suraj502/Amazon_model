"""Define threshold selection against the configured entity-level metric."""

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config_loader import CONFIG, get_config_value
from src.evaluation.f05_scorer import score_f05

SEARCH_RANGE = get_config_value(CONFIG, "decision_strategy", "global_threshold", "search_range")


def _column(frame: pd.DataFrame, names: tuple[str, ...]) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise ValueError(f"Expected one of columns: {', '.join(names)}")


def _probability_column(scores: pd.DataFrame) -> str:
    return _column(scores, ("probability", "match_probability", "score", "prediction"))


def _source_column(scores: pd.DataFrame) -> str:
    return _column(scores, ("source1_entity_id", "source_entity_id", "entity_id"))


def _candidate_column(scores: pd.DataFrame) -> str:
    return _column(scores, ("candidate_entity_id", "matched_entity_id", "entity_id"))


def _selected(scores: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    source_column = _source_column(scores)
    candidate_column = _candidate_column(scores)
    selected = scores.loc[mask].copy()
    if selected.empty:
        return pd.DataFrame(columns=["source1_entity_id", "matched_entity_ids"])
    result = selected.rename(
        columns={source_column: "source1_entity_id", candidate_column: "matched_entity_ids"}
    )
    return result[["source1_entity_id", "matched_entity_ids"]]


def select_matches(
    scores: pd.DataFrame,
    strategy: str = "global_threshold",
    threshold: float = 0.5,
    margin: float = 0.1,
) -> pd.DataFrame:
    """Return one row per selected pair using a named full-set strategy."""
    if scores.empty:
        return pd.DataFrame(columns=["source1_entity_id", "matched_entity_ids"])
    probabilities = _probability_column(scores)
    source_column = _source_column(scores)
    values = pd.to_numeric(scores[probabilities], errors="coerce").fillna(0.0)
    if strategy == "global_threshold":
        mask = values > threshold
    elif strategy == "top_score_margin":
        top_scores = values.groupby(scores[source_column]).transform("max")
        mask = (values > threshold) & (values >= top_scores - margin)
    elif strategy == "entity_adaptive":
        top_scores = values.groupby(scores[source_column]).transform("max")
        ranked = scores.assign(_probability=values).sort_values(
            [source_column, "_probability"], ascending=[True, False]
        )
        second_scores = ranked.groupby(source_column)["_probability"].nth(1)
        gaps = top_scores - scores[source_column].map(second_scores).fillna(0.0)
        candidate_counts = scores.groupby(source_column)[source_column].transform("size")
        similarity_column = next(
            (column for column in ("similarity", "name_similarity", "address_similarity") if column in scores),
            None,
        )
        similarity = (
            pd.to_numeric(scores[similarity_column], errors="coerce").fillna(0.0)
            if similarity_column
            else pd.Series(1.0, index=scores.index)
        )
        # Retain close high-scoring ties, while requiring stronger evidence
        # when an entity has a crowded candidate set.
        mask = (values >= threshold) & (values >= top_scores - margin)
        mask &= (candidate_counts <= 2) | ((gaps <= 0.05) & (similarity >= 0.7))
    else:
        raise ValueError(f"Unknown decision strategy: {strategy}")
    return _selected(scores, mask)


def compare_decision_strategies(
    scores: pd.DataFrame,
    ground_truth: pd.DataFrame,
    config: dict[str, Any] = CONFIG,
) -> pd.DataFrame:
    """Score global, margin, and adaptive decisions and optionally write a report."""
    threshold_values = _threshold_values(config)
    margin_values = _margin_values(config)
    rows: list[dict[str, Any]] = []
    for threshold in threshold_values:
        selected = select_matches(scores, "global_threshold", threshold=threshold)
        rows.append({"strategy": "global_threshold", "threshold": threshold, "margin": None, "f05": score_f05(selected, ground_truth, config)})
    best_threshold = max(rows, key=lambda row: row["f05"])["threshold"] if rows else 0.5
    for margin in margin_values:
        selected = select_matches(scores, "top_score_margin", threshold=best_threshold, margin=margin)
        rows.append({"strategy": "top_score_margin", "threshold": best_threshold, "margin": margin, "f05": score_f05(selected, ground_truth, config)})
    adaptive = select_matches(scores, "entity_adaptive", threshold=best_threshold, margin=0.1)
    rows.append({"strategy": "entity_adaptive", "threshold": best_threshold, "margin": 0.1, "f05": score_f05(adaptive, ground_truth, config)})
    result = pd.DataFrame(rows).sort_values("f05", ascending=False, ignore_index=True)
    report_path = Path(get_config_value(config, "decision_strategy", "report_out"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    winner = result.iloc[0] if not result.empty else None
    lines = ["# Decision Strategy Report", "", "| Strategy | Threshold | Margin | F0.5 |", "|---|---:|---:|---:|"]
    lines.extend(f"| {row.strategy} | {row.threshold:.3f} | {'' if pd.isna(row.margin) else f'{row.margin:.3f}'} | {row.f05:.6f} |" for row in result.itertuples())
    if winner is not None:
        lines.extend(["", f"Winner: **{winner.strategy}** with F0.5 **{winner.f05:.6f}**.", "The winner is selected from entity-level validation F0.5; no strategy is preferred in advance."])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def _threshold_values(config: dict[str, Any]) -> list[float]:
    low, high = get_config_value(config, "decision_strategy", "global_threshold", "search_range")
    step = get_config_value(config, "decision_strategy", "global_threshold", "search_step")
    count = int(round((high - low) / step))
    return [round(low + index * step, 10) for index in range(count + 1)]


def _margin_values(config: dict[str, Any]) -> list[float]:
    low, high = get_config_value(config, "decision_strategy", "top_score_margin", "margin_search_range")
    return [float(low), float(high), round((low + high) / 2, 10)]


def find_best_threshold(scores: pd.DataFrame, ground_truth: pd.DataFrame, config: dict[str, Any] = CONFIG) -> float:
    """Return the best global threshold under entity-level F0.5."""
    candidates = []
    for threshold in _threshold_values(config):
        selected = select_matches(scores, "global_threshold", threshold=threshold)
        candidates.append((score_f05(selected, ground_truth, config), threshold))
    return max(candidates, key=lambda item: (item[0], -item[1]))[1]
