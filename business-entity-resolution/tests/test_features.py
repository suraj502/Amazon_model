"""Contract tests for feature construction."""

from copy import deepcopy

import pandas as pd

from src.features.build_feature_table import build_feature_table, enrich_candidate_pairs
from src.utils.config_loader import CONFIG


def _config_without_optional_features() -> dict:
    config = deepcopy(CONFIG)
    config["features"]["tfidf"] = {"enabled": False}
    config["features"]["embedding"] = {"enabled": False}
    config["features"]["relative_features"] = {"enabled": False}
    return config


def _records(prefix: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "entity_id": f"{prefix}-1",
                "business_name": "Acme Cafe",
                "business_address": "1 Main Street",
                "country": "FR",
                "name_basic_norm": "acme cafe",
                "name_transliterated": "acme cafe",
                "address_basic_norm": "1 main street",
                "city": "Paris",
            },
            {
                "entity_id": f"{prefix}-2",
                "business_name": "Other Shop",
                "business_address": "2 Side Road",
                "country": "FR",
                "name_basic_norm": "other shop",
                "name_transliterated": "other shop",
                "address_basic_norm": "2 side road",
                "city": "Lyon",
            },
        ]
    )


def test_features_preserve_candidate_keys_and_alignment():
    """Feature rows retain the same ordered candidate pairs."""
    config = _config_without_optional_features()
    source1 = _records("S1")
    source2 = _records("S2")
    source3 = _records("S3")
    candidates = pd.DataFrame(
        {
            "source1_entity_id": ["S1-1", "S1-2"],
            "candidate_entity_id": ["S2-1", "S3-2"],
        }
    )
    enriched = enrich_candidate_pairs(candidates, source1, source2, source3, config)
    features = build_feature_table(enriched, config=config)

    assert features[["source1_entity_id", "candidate_entity_id"]].to_dict("records") == candidates.to_dict("records")
    assert {"name_jaro_winkler", "address_jaro_winkler"}.issubset(features.columns)
    assert len(features) == len(candidates)


def test_ground_truth_accepts_comma_and_pipe_match_lists():
    """The M4-style ground truth parser accepts both repository delimiters."""
    config = _config_without_optional_features()
    source1 = _records("S1")
    source2 = _records("S2")
    source3 = _records("S3")
    candidates = pd.DataFrame(
        {
            "source1_entity_id": ["S1-1", "S1-1", "S1-2"],
            "candidate_entity_id": ["S2-1", "S3-1", "S2-2"],
        }
    )
    enriched = enrich_candidate_pairs(candidates, source1, source2, source3, config)
    ground_truth = pd.DataFrame(
        {
            "source1_entity_id": ["S1-1", "S1-2"],
            "matched_entity_ids": ["S2-1,S3-1", "S2-2|S3-2"],
        }
    )
    features = build_feature_table(enriched, config=config, ground_truth=ground_truth)

    assert features["label"].tolist() == [1, 1, 1]
