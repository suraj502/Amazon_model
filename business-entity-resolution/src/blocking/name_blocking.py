
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


def _get_strategy_config(
    config: dict[str, Any],
    strategy_name: str,
) -> dict[str, Any]:
    """Return configuration for a named blocking strategy."""
    for strategy in config.get("blocking", {}).get("strategies", []):
        if strategy.get("name") == strategy_name:
            return strategy

    return {}


def _clean_token_set(tokens: Any) -> set[str]:
    """
    Convert list-like token values, including numpy arrays,
    to a clean set of strings.
    """
    if tokens is None:
        return set()

    try:
        return {
            str(token).strip()
            for token in tokens
            if str(token).strip()
        }
    except TypeError:
        return set()


@dataclass
class NameTokenIndex:
    """
    Reusable target-side index for name-token blocking.

    token_frequency:
        Number of target entities containing each token.

    token_to_ids:
        Informative token -> target entity IDs.

    id_to_tokens:
        Target entity ID -> normalized token set.
    """

    token_frequency: Counter[str]
    token_to_ids: dict[str, list[str]]
    id_to_tokens: dict[str, set[str]]


def build_name_token_index(
    target_df: pd.DataFrame,
    config: dict[str, Any],
) -> NameTokenIndex:
    """
    Build a reusable target-side name-token index.

    The expensive target scan happens here once.

    The resulting index can then be reused for multiple
    source chunks without rebuilding the 5M-row target index.
    """

    required_columns = {
        "entity_id",
        "name_tokens",
    }

    missing = required_columns - set(target_df.columns)

    if missing:
        raise ValueError(
            "Target dataframe missing required columns: "
            f"{sorted(missing)}"
        )

    strategy_config = _get_strategy_config(
        config,
        "name_token_overlap",
    )

    runtime_config = config.get("runtime", {})

    chunk_size = int(
        runtime_config.get(
            "chunk_size",
            5000,
        )
    )

    max_token_frequency = int(
        strategy_config.get(
            "max_token_frequency",
            100_000,
        )
    )

    target_rows = len(target_df)

    print("\n[Block B] Building reusable target index...")
    print(f"  Target rows: {target_rows:,}")
    print(f"  Chunk size: {chunk_size:,}")
    print(f"  Max token frequency: {max_token_frequency:,}")

    # ---------------------------------------------------------
    # Phase 1: calculate token document frequencies
    # ---------------------------------------------------------
    print("\n  Phase 1/2: calculating token frequencies...")

    token_frequency: Counter[str] = Counter()

    for start in range(
        0,
        target_rows,
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            target_rows,
        )

        for tokens in target_df.iloc[start:end]["name_tokens"]:
            clean_tokens = _clean_token_set(tokens)

            # name_tokens are converted to a set, so a token
            # is counted at most once per target entity.
            token_frequency.update(clean_tokens)

        if end % 100_000 == 0 or end == target_rows:
            print(
                "    Frequency rows processed: "
                f"{end:,} / {target_rows:,}"
            )

    print(f"    Unique tokens: {len(token_frequency):,}")

    # ---------------------------------------------------------
    # Phase 2: build inverted index
    # ---------------------------------------------------------
    print("\n  Phase 2/2: building inverted index...")

    token_to_ids: dict[str, list[str]] = defaultdict(list)
    id_to_tokens: dict[str, set[str]] = {}

    skipped_common_postings = 0
    usable_target_rows = 0

    for start in range(
        0,
        target_rows,
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            target_rows,
        )

        chunk = target_df.iloc[start:end][
            [
                "entity_id",
                "name_tokens",
            ]
        ]

        for entity_id, tokens in zip(
            chunk["entity_id"],
            chunk["name_tokens"],
        ):
            clean_tokens = _clean_token_set(tokens)

            if not clean_tokens:
                continue

            target_id = str(entity_id)
            id_to_tokens[target_id] = clean_tokens
            usable_target_rows += 1

            for token in clean_tokens:
                if token_frequency[token] <= max_token_frequency:
                    token_to_ids[token].append(target_id)
                else:
                    skipped_common_postings += 1

        if end % 100_000 == 0 or end == target_rows:
            print(
                "    Index rows processed: "
                f"{end:,} / {target_rows:,}"
            )

    print(f"    Usable target rows: {usable_target_rows:,}")
    print(f"    Indexed tokens: {len(token_to_ids):,}")
    print(
        "    Common-token postings skipped: "
        f"{skipped_common_postings:,}"
    )

    return NameTokenIndex(
        token_frequency=token_frequency,
        token_to_ids=dict(token_to_ids),
        id_to_tokens=id_to_tokens,
    )


def _jaccard_similarity(
    source_tokens: set[str],
    target_tokens: set[str],
) -> float:
    """Calculate Jaccard similarity between two token sets."""
    if not source_tokens or not target_tokens:
        return 0.0

    intersection = len(source_tokens & target_tokens)
    union = len(source_tokens | target_tokens)

    if union == 0:
        return 0.0

    return intersection / union


def block_by_name(
    source1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Block candidates using exact normalized alphanumeric name.

    Country is intentionally not used as a hard filter.
    """

    required_source = {
        "entity_id",
        "name_alnum_norm",
    }

    required_target = {
        "entity_id",
        "name_alnum_norm",
    }

    missing_source = required_source - set(source1_df.columns)
    missing_target = required_target - set(target_df.columns)

    if missing_source:
        raise ValueError(
            "Source1 dataframe missing required columns: "
            f"{sorted(missing_source)}"
        )

    if missing_target:
        raise ValueError(
            "Target dataframe missing required columns: "
            f"{sorted(missing_target)}"
        )

    target_index: dict[str, list[str]] = defaultdict(list)

    for entity_id, name in zip(
        target_df["entity_id"],
        target_df["name_alnum_norm"],
    ):
        if pd.isna(name):
            continue

        normalized_name = str(name).strip()

        if not normalized_name:
            continue

        target_index[normalized_name].append(str(entity_id))

    pairs: list[dict[str, str]] = []

    for source_id, name in zip(
        source1_df["entity_id"],
        source1_df["name_alnum_norm"],
    ):
        if pd.isna(name):
            continue

        normalized_name = str(name).strip()

        if not normalized_name:
            continue

        for target_id in target_index.get(
            normalized_name,
            [],
        ):
            pairs.append(
                {
                    "source1_entity_id": str(source_id),
                    "matched_entity_id": target_id,
                }
            )

    return pd.DataFrame(
        pairs,
        columns=[
            "source1_entity_id",
            "matched_entity_id",
        ],
    )


def generate_name_token_candidates(
    source1_df: pd.DataFrame,
    index: NameTokenIndex,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Generate Block B name-token-overlap candidates using a prebuilt
    NameTokenIndex.

    The target/source2 or source3 index is built separately and reused.
    This function only processes source1 records.
    """
    required_columns = {"entity_id", "name_tokens"}
    missing = required_columns - set(source1_df.columns)

    if missing:
        raise ValueError(
            f"source1_df is missing required columns: {sorted(missing)}"
        )

    strategy_config = _get_strategy_config(
        config,
        "name_token_overlap",
    )

    top_k = int(strategy_config.get("top_k", 20))
    chunk_size = int(
        config.get("runtime", {}).get("chunk_size", 5000)
    )
    max_token_frequency = int(
        strategy_config.get("max_token_frequency", 100000)
    )
    max_scoring_candidates = int(
        strategy_config.get("max_scoring_candidates", 100)
    )

    all_pairs: list[dict[str, str | float]] = []

    total_rows = len(source1_df)

    for start in range(0, total_rows, chunk_size):
        end = min(start + chunk_size, total_rows)
        chunk = source1_df.iloc[start:end]

        for _, row in chunk.iterrows():
            source_id = str(row["entity_id"])
            source_tokens = _clean_token_set(row["name_tokens"])

            if not source_tokens:
                continue

            # Ignore extremely common tokens because their postings
            # create huge and low-quality candidate sets.
            informative_tokens = [
                token
                for token in source_tokens
                if index.token_frequency.get(token, 0)
                <= max_token_frequency
                and token in index.token_to_ids
            ]

            if not informative_tokens:
                continue

            # Start with the rarest tokens.
            informative_tokens.sort(
                key=lambda token: (
                    index.token_frequency.get(token, 0),
                    token,
                )
            )

            shared_counts: Counter[str] = Counter()

            for token in informative_tokens:
                for target_id in index.token_to_ids.get(token, []):
                    shared_counts[target_id] += 1

            if not shared_counts:
                continue

            # First reduce the candidate set using shared-token count.
            strongest_candidates = sorted(
                shared_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:max_scoring_candidates]

            scored_candidates = []

            for target_id, shared_count in strongest_candidates:
                target_tokens = index.id_to_tokens.get(target_id)

                if not target_tokens:
                    continue

                union_size = (
                    len(source_tokens)
                    + len(target_tokens)
                    - shared_count
                )

                if union_size <= 0:
                    continue

                score = shared_count / union_size

                scored_candidates.append(
                    (
                        target_id,
                        score,
                    )
                )

            scored_candidates.sort(
                key=lambda item: (-item[1], item[0])
            )

            for target_id, score in scored_candidates[:top_k]:
                all_pairs.append(
                    {
                        "source1_entity_id": source_id,
                        "candidate_entity_id": target_id,
                        "score": float(score),
                    }
                )

        print(
            f"Candidate generation: "
            f"{end:,}/{total_rows:,} S1 rows | "
            f"pairs: {len(all_pairs):,}"
        )

    return pd.DataFrame(
        all_pairs,
        columns=[
            "source1_entity_id",
            "candidate_entity_id",
            "score",
        ],
    )


def generate_name_token_candidates_to_file(
    source1_df: pd.DataFrame,
    index: NameTokenIndex,
    config: dict[str, Any],
    output_path: str,
) -> int:
    """
    Generate name-token-overlap candidates incrementally.

    Uses:
    - rare-token-first candidate generation
    - shared-token counting
    - rarity-aware scoring
    - Jaccard similarity
    - top-k candidates per S1 entity

    Candidates are written incrementally to TSV so the full candidate
    set is never kept in memory.
    """
    required_columns = {"entity_id", "name_tokens"}
    missing = required_columns - set(source1_df.columns)

    if missing:
        raise ValueError(
            f"source1_df is missing required columns: {sorted(missing)}"
        )

    strategy = _get_strategy_config(
        config,
        "name_token_overlap",
    )

    top_k = int(strategy.get("top_k", 20))
    max_token_frequency = int(
        strategy.get("max_token_frequency", 100000)
    )
    max_scoring_candidates = int(
        strategy.get("max_scoring_candidates", 100)
    )

    output = Path(output_path)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns = [
        "source1_entity_id",
        "candidate_entity_id",
        "score",
    ]

    total_pairs = 0

    source1_df = source1_df.reset_index(drop=True)

    with output.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        f.write("\t".join(columns) + "\n")

        for start in range(
            0,
            len(source1_df),
            int(config.get("runtime", {}).get("chunk_size", 5000)),
        ):
            chunk = source1_df.iloc[
                start : start
                + int(
                    config.get("runtime", {})
                    .get("chunk_size", 5000)
                )
            ]

            chunk_pairs = []

            for row in chunk.itertuples(index=False):
                source_id = str(
                    getattr(row, "entity_id")
                )

                source_tokens = _clean_token_set(
                    getattr(row, "name_tokens")
                )

                if not source_tokens:
                    continue

                usable_tokens = [
                    token
                    for token in source_tokens
                    if token in index.token_to_ids
                    and index.token_frequency.get(
                        token,
                        0,
                    )
                    <= max_token_frequency
                ]

                if not usable_tokens:
                    continue

                # Rare tokens are more informative.
                usable_tokens.sort(
                    key=lambda token: (
                        index.token_frequency.get(
                            token,
                            0,
                        ),
                        token,
                    )
                )

                shared_counts: Counter[str] = Counter()

                for token in usable_tokens:
                    for target_id in index.token_to_ids.get(
                        token,
                        [],
                    ):
                        shared_counts[target_id] += 1

                if not shared_counts:
                    continue

                # First restrict the expensive scoring stage.
                candidate_ids = [
                    target_id
                    for target_id, _ in shared_counts.most_common(
                        max_scoring_candidates
                    )
                ]

                scored = []

                source_size = len(source_tokens)

                for target_id in candidate_ids:
                    target_tokens = index.id_to_tokens.get(
                        target_id
                    )

                    if not target_tokens:
                        continue

                    shared = shared_counts[target_id]

                    union_size = (
                        source_size
                        + len(target_tokens)
                        - shared
                    )

                    if union_size <= 0:
                        continue

                    jaccard = (
                        shared / union_size
                    )

                    # Rarity-aware bonus.
                    #
                    # Rare shared tokens contribute more than
                    # extremely common business/legal tokens.
                    rarity_score = 0.0

                    for token in usable_tokens:
                        if target_id not in index.token_to_ids.get(
                            token,
                            [],
                        ):
                            continue

                        frequency = max(
                            index.token_frequency.get(
                                token,
                                1,
                            ),
                            1,
                        )

                        rarity_score += 1.0 / (
                            frequency ** 0.5
                        )

                    # Keep Jaccard as the main signal.
                    # Rarity is a small tie-breaking signal.
                    score = (
                        jaccard
                        + 0.10 * rarity_score
                    )

                    scored.append(
                        (
                            target_id,
                            score,
                            jaccard,
                        )
                    )

                scored.sort(
                    key=lambda item: (
                        item[1],
                        item[2],
                    ),
                    reverse=True,
                )

                for target_id, score, _ in scored[:top_k]:
                    chunk_pairs.append(
                        (
                            source_id,
                            str(target_id),
                            float(score),
                        )
                    )

            for source_id, target_id, score in chunk_pairs:
                f.write(
                    f"{source_id}\t"
                    f"{target_id}\t"
                    f"{score:.8f}\n"
                )

            total_pairs += len(chunk_pairs)

            print(
                f"Candidate generation: "
                f"{min(start + len(chunk), len(source1_df)):,}/"
                f"{len(source1_df):,} S1 rows | "
                f"pairs written: {total_pairs:,}"
            )

    return total_pairs


def block_by_name_token_overlap(
    source1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Temporary compatibility wrapper for Block B.

    The reusable target index is now available through
    build_name_token_index().

    Candidate-generation logic will be moved to the
    reusable-index path in the next step.
    """

    required_source = {
        "entity_id",
        "name_tokens",
    }

    required_target = {
        "entity_id",
        "name_tokens",
    }

    missing_source = required_source - set(source1_df.columns)
    missing_target = required_target - set(target_df.columns)

    if missing_source:
        raise ValueError(
            "Source1 dataframe missing required columns: "
            f"{sorted(missing_source)}"
        )

    if missing_target:
        raise ValueError(
            "Target dataframe missing required columns: "
            f"{sorted(missing_target)}"
        )

    strategy_config = _get_strategy_config(
        config,
        "name_token_overlap",
    )

    top_k = int(
        strategy_config.get(
            "top_k",
            20,
        )
    )

    runtime_config = config.get(
        "runtime",
        {}
    )

    chunk_size = int(
        runtime_config.get(
            "chunk_size",
            5000,
        )
    )

    max_token_frequency = int(
        strategy_config.get(
            "max_token_frequency",
            100_000,
        )
    )

    max_scoring_candidates = int(
        strategy_config.get(
            "max_scoring_candidates",
            100,
        )
    )

    print(f"Block B top_k: {top_k}")
    print(f"Block B max_token_frequency: {max_token_frequency}")
    print(f"Block B max_scoring_candidates: {max_scoring_candidates}")
    print(f"Block B chunk_size: {chunk_size}")

    # Build the target index using the new reusable builder.
    index = build_name_token_index(
        target_df,
        config,
    )

    print("\n[Block B] Generating candidates...")

    source_rows = len(source1_df)

    all_pairs: list[dict[str, str]] = []

    total_candidate_pairs = 0

    for start in range(
        0,
        source_rows,
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            source_rows,
        )

        print(
            f"\n  Processing S1 rows "
            f"{start + 1:,} - "
            f"{end:,} / "
            f"{source_rows:,}"
        )

        source_chunk = source1_df.iloc[start:end][
            [
                "entity_id",
                "name_tokens",
            ]
        ]

        for local_index, (
            source_id,
            tokens,
        ) in enumerate(
            zip(
                source_chunk["entity_id"],
                source_chunk["name_tokens"],
            ),
            start=1,
        ):
            source_tokens = _clean_token_set(tokens)

            if not source_tokens:
                continue

            informative_tokens = [
                token
                for token in source_tokens
                if index.token_frequency.get(
                    token,
                    0,
                )
                <= max_token_frequency
            ]

            if not informative_tokens:
                continue

            informative_tokens.sort(
                key=lambda token: (
                    index.token_frequency.get(
                        token,
                        0,
                    ),
                    token,
                )
            )

            shared_counts: Counter[str] = Counter()

            for token in informative_tokens:
                for target_id in index.token_to_ids.get(
                    token,
                    [],
                ):
                    shared_counts[target_id] += 1

            if not shared_counts:
                continue

            strongest_candidates = sorted(
                shared_counts.items(),
                key=lambda item: (
                    -item[1],
                    item[0],
                ),
            )[:max_scoring_candidates]

            scored_candidates = []

            for (
                target_id,
                shared_count,
            ) in strongest_candidates:

                target_token_set = index.id_to_tokens.get(
                    target_id
                )

                if not target_token_set:
                    continue

                union_size = (
                    len(source_tokens)
                    + len(target_token_set)
                    - shared_count
                )

                if union_size <= 0:
                    continue

                score = shared_count / union_size

                scored_candidates.append(
                    (
                        score,
                        target_id,
                    )
                )

            scored_candidates.sort(
                key=lambda item: (
                    -item[0],
                    item[1],
                )
            )

            for _, target_id in scored_candidates[:top_k]:
                all_pairs.append(
                    {
                        "source1_entity_id": str(source_id),
                        "matched_entity_id": str(target_id),
                    }
                )

                total_candidate_pairs += 1

            if local_index % 500 == 0:
                print(
                    f"    S1 chunk progress: "
                    f"{local_index:,} / "
                    f"{end - start:,}"
                    f" | total pairs: "
                    f"{total_candidate_pairs:,}"
                )

        print(
            f"  Completed S1 rows: "
            f"{end:,} / "
            f"{source_rows:,}"
        )

        print(
            f"  Total candidate pairs so far: "
            f"{total_candidate_pairs:,}"
        )

    print("\n[Block B] Complete.")

    print(
        f"  Total candidate pairs: "
        f"{total_candidate_pairs:,}"
    )

    return pd.DataFrame(
        all_pairs,
        columns=[
            "source1_entity_id",
            "matched_entity_id",
        ],
    )

