import pandas as pd



from src.blocking.name_blocking import (
    block_by_name,
    block_by_name_token_overlap,
    build_char_ngram_index,
    generate_char_ngram_candidates,
    block_by_name_char_ngram,
)

TEST_CONFIG = {
    "blocking": {
        "strategies": [
            {
                "name": "name_token_overlap",
                "method": "jaccard",
                "top_k": 20,
            },
            {
                "name": "exact_name",
                "view": "name_alnum_norm",
                "method": "exact",
            },
        ]
    }
}

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

    result = block_by_name(source1, target, TEST_CONFIG)

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

    result = block_by_name(
    source1,
    target,
    TEST_CONFIG,
)

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

    result = block_by_name(
    source1,
    target,
    TEST_CONFIG,
)

    assert result.empty

def test_block_by_name_token_overlap():
    source1 = pd.DataFrame(
        {
            "entity_id": ["S1-1"],
            "name_tokens": [
                ["abc", "food", "services"],
            ],
        }
    )

    target = pd.DataFrame(
        {
            "entity_id": ["S2-1", "S2-2", "S2-3"],
            "name_tokens": [
                ["abc", "food", "services"],
                ["abc", "food", "service"],
                ["completely", "different"],
            ],
        }
    )

    result = block_by_name_token_overlap(
    source1,
    target,
    TEST_CONFIG,
)

    assert "S2-1" in result["matched_entity_id"].tolist()
    assert "S2-2" in result["matched_entity_id"].tolist()
    assert "S2-3" not in result["matched_entity_id"].tolist()


def test_block_by_name_token_overlap_top_k():
    source1 = pd.DataFrame(
        {
            "entity_id": ["S1-1"],
            "name_tokens": [["abc", "food"]],
        }
    )

    target = pd.DataFrame(
        {
            "entity_id": [f"S2-{i}" for i in range(1, 6)],
            "name_tokens": [
                ["abc", "food"],
                ["abc"],
                ["food"],
                ["abc", "x"],
                ["food", "y"],
            ],
        }
    )

    config = {
        "blocking": {
            "strategies": [
                {
                    "name": "name_token_overlap",
                    "method": "jaccard",
                    "top_k": 2,
                }
            ]
        }
    }

    result = block_by_name_token_overlap(
        source1,
        target,
        config=config,
    )

    assert len(result) == 2


def test_block_by_name_token_overlap_empty_tokens():
    source1 = pd.DataFrame(
        {
            "entity_id": ["S1-1"],
            "name_tokens": [[]],
        }
    )

    target = pd.DataFrame(
        {
            "entity_id": ["S2-1"],
            "name_tokens": [["abc"]],
        }
    )

    result = block_by_name_token_overlap(
    source1,
    target,
    TEST_CONFIG,
)

    assert result.empty
    
def test_build_char_ngram_index():
    target = pd.DataFrame(
        {
            "entity_id": ["S2-1", "S2-2"],
            "name_basic_norm": [
                "acme foods",
                "prime bank",
            ],
        }
    )

    config = {
        "blocking": {
            "strategies": [
                {
                    "name": "name_char_ngram",
                    "method": "char_ngram",
                    "ngram_size": 3,
                    "top_k": 20,
                    "max_ngram_frequency": 100000,
                    "max_query_ngrams": 12,
                    "max_scoring_candidates": 100,
                }
            ]
        },
        "runtime": {
            "chunk_size": 5000,
        },
    }

    index = build_char_ngram_index(
        target,
        config,
    )

    assert len(index.id_to_ngrams) == 2
    assert set(index.id_to_ngrams.keys()) == {"S2-1", "S2-2"}

    assert " ac" in index.ngram_to_ids
    assert "pri" in index.ngram_to_ids

    assert len(index.id_to_ngrams) == 2


def test_generate_char_ngram_candidates_finds_fuzzy_name():
    source1 = pd.DataFrame(
        {
            "entity_id": ["S1-1"],
            "name_basic_norm": [
                "acme foods",
            ],
        }
    )

    target = pd.DataFrame(
        {
            "entity_id": [
                "S2-1",
                "S2-2",
                "S2-3",
            ],
            "name_basic_norm": [
                "acme food",
                "completely different",
                "prime bank",
            ],
        }
    )

    config = {
        "blocking": {
            "strategies": [
                {
                    "name": "name_char_ngram",
                    "method": "char_ngram",
                    "ngram_size": 3,
                    "top_k": 20,
                }
            ]
        }
    }

    index = build_char_ngram_index(
        target,
        config,
    )

    result = generate_char_ngram_candidates(
        source1,
        index,
        config,
    )

    assert "S2-1" in result[
        "candidate_entity_id"
    ].tolist()

    assert "S2-2" not in result[
        "candidate_entity_id"
    ].tolist()


def test_generate_char_ngram_candidates_top_k():
    source1 = pd.DataFrame(
        {
            "entity_id": ["S1-1"],
            "name_basic_norm": [
                "acme foods",
            ],
        }
    )

    target = pd.DataFrame(
        {
            "entity_id": [
                "S2-1",
                "S2-2",
                "S2-3",
                "S2-4",
            ],
            "name_basic_norm": [
                "acme foods",
                "acme food",
                "acme foods company",
                "completely different",
            ],
        }
    )

    config = {
        "blocking": {
            "strategies": [
                {
                    "name": "name_char_ngram",
                    "method": "char_ngram",
                    "ngram_size": 3,
                    "top_k": 2,
                }
            ]
        }
    }

    index = build_char_ngram_index(
        target,
        config,
    )

    result = generate_char_ngram_candidates(
        source1,
        index,
        config,
    )

    assert len(result) <= 2


def test_generate_char_ngram_candidates_empty_name():
    source1 = pd.DataFrame(
        {
            "entity_id": ["S1-1"],
            "name_basic_norm": [""],
        }
    )

    target = pd.DataFrame(
        {
            "entity_id": ["S2-1"],
            "name_basic_norm": ["acme foods"],
        }
    )

    config = {
        "blocking": {
            "strategies": [
                {
                    "name": "name_char_ngram",
                    "method": "char_ngram",
                    "ngram_size": 3,
                    "top_k": 20,
                }
            ]
        }
    }

    index = build_char_ngram_index(
        target,
        config,
    )

    result = generate_char_ngram_candidates(
        source1,
        index,
        config,
    )

    assert result.empty