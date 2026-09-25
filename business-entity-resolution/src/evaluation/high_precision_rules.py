"""Define configurable high-precision decision rule contracts."""

from typing import Any

import pandas as pd

from src.utils.config_loader import CONFIG, get_config_value
from src.evaluation.f05_scorer import score_f05

RULES_ENABLED = get_config_value(CONFIG, "decision_strategy", "high_precision_rules", "enabled")


def apply_high_precision_rules(scores: pd.DataFrame, config: dict[str, Any] = CONFIG) -> pd.DataFrame:
    """Mark exact-name/strong-address/country matches as high-confidence pairs.

    Rule output is only a score override candidate. Call ``validate_rules``
    against held-out labels before using it in a production decision.
    """
    if not RULES_ENABLED or scores.empty:
        return scores.copy()
    result = scores.copy()
    required = ("name_basic_norm", "address_basic_norm", "country")
    if not all(column in result.columns for column in required):
        result["high_precision_rule"] = False
        return result
    name = result["name_basic_norm"].fillna("").astype(str).str.strip()
    address = result["address_basic_norm"].fillna("").astype(str).str.strip()
    country = result["country"].fillna("").astype(str).str.casefold().str.strip()
    address_similarity_column = next(
        (column for column in ("address_similarity", "address_score") if column in result),
        None,
    )
    strong_address = (
        result[address_similarity_column].fillna(0.0).astype(float).ge(0.9)
        if address_similarity_column
        else address.ne("")
    )
    result["high_precision_rule"] = name.ne("") & strong_address & country.ne("")
    probability_column = next((column for column in ("probability", "match_probability", "score") if column in result), None)
    if probability_column is not None:
        result.loc[result["high_precision_rule"], probability_column] = 1.0
    return result


def validate_rules(
    scores: pd.DataFrame,
    ground_truth: pd.DataFrame,
    config: dict[str, Any] = CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep a rule only when it measurably improves held-out F0.5."""
    from src.evaluation.threshold_search import select_matches

    baseline = select_matches(scores, "global_threshold", threshold=0.5)
    ruled_scores = apply_high_precision_rules(scores, config)
    ruled = select_matches(ruled_scores, "global_threshold", threshold=0.5)
    baseline_score = score_f05(baseline, ground_truth, config)
    ruled_score = score_f05(ruled, ground_truth, config)
    summary = pd.DataFrame(
        [{"rule": "exact_name_strong_address_same_country", "baseline_f05": baseline_score, "rule_f05": ruled_score, "kept": ruled_score > baseline_score}]
    )
    if ruled_score <= baseline_score:
        ruled_scores["high_precision_rule"] = False
    return ruled_scores, summary
