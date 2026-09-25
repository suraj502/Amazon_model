import pandas as pd

from src.blocking.name_blocking import block_by_name


def test_block_by_name_exact_match():
    source1 = pd.DataFrame(
        {
            "entity_id": ["S1-1", "S1-2", "S1-3"],
            "name_alnum_norm": [
                "acmecompany",
                "primemoney",
                "",
            ],
        }
    )

    target = pd.DataFrame(
        {
            "entity_id": ["S2-1", "S2-2", "S2-3", "S2-4"],
            "name_alnum_norm": [
                "acmecompany",
                "acmecompany",
                "otherbusiness",
                "",
            ],
        }
    )

    result = block_by_name(source1, target)

    expected = {
        ("S1-1", "S2-1"),
        ("S1-1", "S2-2"),
    }

    actual = set(
        zip(
            result["source1_entity_id"],
            result["matched_entity_id"],
        )
    )

    assert actual == expected


def test_block_by_name_does_not_use_country():
    source1 = pd.DataFrame(
        {
            "entity_id": ["S1-1"],
            "name_alnum_norm": ["acmecompany"],
            "country": ["US"],
        }
    )

    target = pd.DataFrame(
        {
            "entity_id": ["S2-1"],
            "name_alnum_norm": ["acmecompany"],
            "country": ["France"],
        }
    )

    result = block_by_name(source1, target)

    assert len(result) == 1
    assert result.iloc[0]["matched_entity_id"] == "S2-1"


def test_block_by_name_ignores_empty_names():
    source1 = pd.DataFrame(
        {
            "entity_id": ["S1-1"],
            "name_alnum_norm": [""],
        }
    )

    target = pd.DataFrame(
        {
            "entity_id": ["S2-1"],
            "name_alnum_norm": [""],
        }
    )

    result = block_by_name(source1, target)

    assert result.empty