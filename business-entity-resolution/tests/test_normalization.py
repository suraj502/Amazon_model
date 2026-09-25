"""Regression tests for the normalization module."""

import pandas as pd

from src.normalization.address_normalize import normalize_addresses
from src.normalization.name_normalize import normalize_names


def test_required_normalization_views_are_present():
    frame = pd.DataFrame(
        {
            "entity_id": ["S1-1", "S1-2"],
            "business_name": ["Acme Retail LLC", "Ram Market Pvt Ltd"],
            "business_address": ["123 Main St, Austin, TX", "Near Plaza, Paris"],
        }
    )

    names = normalize_names(frame)
    addresses = normalize_addresses(frame)

    expected_name_columns = {
        "name_original",
        "name_basic_norm",
        "name_token_norm",
        "name_alnum_norm",
        "name_transliterated",
        "name_tokens",
    }
    expected_address_columns = {
        "address_original",
        "address_basic_norm",
        "address_component_parsed",
    }

    assert expected_name_columns.issubset(names.columns)
    assert expected_address_columns.issubset(addresses.columns)
    assert names["entity_id"].tolist() == frame["entity_id"].tolist()


def test_name_original_is_recoverable_after_normalization():
    original = "O'Reilly & Sons, Inc"
    frame = pd.DataFrame({"entity_id": ["S1-100"], "business_name": [original]})

    result = normalize_names(frame)

    assert result["name_original"].iloc[0] == original
    assert result["name_basic_norm"].iloc[0] == "o reilly sons inc"


def test_legal_suffix_normalization_uses_configured_dictionary():
    frame = pd.DataFrame({"entity_id": ["S1-42"], "business_name": ["Acme Retail LLC"]})

    result = normalize_names(frame)

    assert result["name_token_norm"].iloc[0] == "acme retail"
    assert "llc" not in result["name_token_norm"].iloc[0]


def test_transliteration_example_for_non_latin_name():
    frame = pd.DataFrame({"entity_id": ["S2-9"], "business_name": ["राम मार्केटिंग प्राइवेट लिमिटेड"]})

    result = normalize_names(frame)

    transliterated = result["name_transliterated"].iloc[0]
    assert "ram" in transliterated.lower()
    assert "marketing" in transliterated.lower()
    assert transliterated != "राम मार्केटिंग प्राइवेट लिमिटेड"


def test_missing_address_component_handling_avoids_garbage_values():
    frame = pd.DataFrame(
        {
            "entity_id": ["A", "B", "C", "D"],
            "business_address": ["N/A", "", None, "null"],
        }
    )

    result = normalize_addresses(frame)
    parsed = result["address_component_parsed"].fillna("").astype(str)

    assert parsed.tolist() == ["", "", "", ""]
    assert not parsed.str.contains(r"n/a|null|none|nan", case=False, regex=True).any()


def test_landmark_references_are_flagged_not_resolved():
    frame = pd.DataFrame(
        {
            "entity_id": ["L1", "L2"],
            "business_address": ["Near City Plaza, Paris", "Opposite Main Station, Lyon"],
        }
    )

    result = normalize_addresses(frame)

    assert result["address_landmark_flag"].tolist() == [True, True]
    assert "near" not in result["address_component_parsed"].iloc[0].lower()
    assert "opposite" not in result["address_component_parsed"].iloc[1].lower()
