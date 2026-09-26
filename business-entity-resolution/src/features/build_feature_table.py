"""Join normalized records and assemble the M3 candidate feature table."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd

from src.features.embedding_features import build_embedding_features
from src.features.relative_features import build_relative_features
from src.features.string_features import build_string_features
from src.features.tfidf_features import (
    build_tfidf_features,
    fit_tfidf_vectorizers,
)
from src.utils.config_loader import CONFIG, get_config_value


PAIR_KEYS = ("source1_entity_id", "candidate_entity_id", "source_type")
NORMALIZED_COLUMNS = (
    "name_basic_norm",
    "name_transliterated",
    "address_basic_norm",
    "country",
    "postal_code",
    "postcode",
    "zip",
    "pin",
    "pincode",
    "city",
    "town",
    "state",
    "province",
    "region",
)
FEATURE_GROUP_PREFIXES = (
    "name_",
    "address_",
    "phonetic_",
    "country_",
    "source_",
)


def _enabled(config: dict[str, Any], name: str) -> bool:
    value = get_config_value(config, "features", name)
    return bool(value.get("enabled", True)) if isinstance(value, dict) else bool(value)


def _canonical_source_types(
    pairs: pd.DataFrame,
    config: dict[str, Any],
) -> pd.Series:
    candidate_ids = pairs["candidate_entity_id"].astype(str)
    if "source_type" in pairs.columns:
        raw_types = pairs["source_type"].astype(str).str.strip()
    else:
        raw_types = pd.Series("", index=pairs.index, dtype="object")

    prefixes = get_config_value(config, "schema", "source_prefixes")
    normalized: list[str] = []
    for candidate_id, source_type in zip(candidate_ids, raw_types):
        kind = source_type.strip().upper()
        if kind in {"S2", "SOURCE2"}:
            normalized.append("S2")
            continue
        if kind in {"S3", "SOURCE3"}:
            normalized.append("S3")
            continue
        if not kind:
            if candidate_id.startswith(prefixes["source2"]):
                normalized.append("S2")
                continue
            if candidate_id.startswith(prefixes["source3"]):
                normalized.append("S3")
                continue
        raise ValueError(
            f"Cannot determine candidate source type for ID {candidate_id!r}; "
            "provide source_type as S2 or S3"
        )
    return pd.Series(normalized, index=pairs.index, name="source_type")


def _normalized_records(
    records: pd.DataFrame,
    suffix: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    id_column = get_config_value(config, "schema", "id_column")
    name_column = get_config_value(config, "schema", "name_column")
    address_column = get_config_value(config, "schema", "address_column")
    country_column = get_config_value(config, "schema", "country_column")
    if id_column not in records.columns:
        raise ValueError(f"Normalized records are missing ID column {id_column!r}")

    def choose(columns: tuple[str, ...], fallback: str) -> pd.Series:
        for column in columns:
            if column in records.columns:
                return records[column]
        if fallback in records.columns:
            return records[fallback]
        return pd.Series("", index=records.index, dtype="object")

    if not any(
        column in records.columns
        for column in ("name_basic_norm", "name_transliterated", name_column)
    ):
        raise ValueError("Normalized records have no usable business-name column")
    if not any(
        column in records.columns
        for column in ("address_basic_norm", address_column)
    ):
        raise ValueError("Normalized records have no usable business-address column")

    view = pd.DataFrame(index=records.index)
    view["entity_id"] = records[id_column].astype(str)
    view[f"name_basic_norm_{suffix}"] = choose(
        ("name_basic_norm", "name_transliterated", name_column),
        name_column,
    )
    view[f"name_transliterated_{suffix}"] = choose(
        ("name_transliterated", "name_basic_norm", name_column),
        name_column,
    )
    view[f"address_basic_norm_{suffix}"] = choose(
        ("address_basic_norm", address_column),
        address_column,
    )
    view[f"country_{suffix}"] = (
        records[country_column]
        if country_column in records.columns
        else ""
    )

    for column in NORMALIZED_COLUMNS:
        if column not in {"name_basic_norm", "name_transliterated", "address_basic_norm", "country"}:
            if column in records.columns:
                view[f"{column}_{suffix}"] = records[column]
    duplicate_ids = view.loc[view["entity_id"].duplicated(keep=False), "entity_id"]
    if not duplicate_ids.empty:
        raise ValueError(
            f"Normalized records contain duplicate entity ID {duplicate_ids.iloc[0]!r}"
        )
    return view


def enrich_candidate_pairs(
    pairs: pd.DataFrame,
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    config: dict[str, Any] = CONFIG,
) -> pd.DataFrame:
    """Join the three normalized sources to M2's ID-keyed candidate rows."""
    required = {"source1_entity_id", "candidate_entity_id"}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError("Candidate pairs are missing columns: " + ", ".join(sorted(missing)))
    if pairs[["source1_entity_id", "candidate_entity_id"]].isna().any().any():
        raise ValueError("Candidate pair IDs cannot be null")

    frame = pairs.copy().reset_index(drop=True)
    frame["source1_entity_id"] = frame["source1_entity_id"].astype(str)
    frame["candidate_entity_id"] = frame["candidate_entity_id"].astype(str)
    frame["source_type"] = _canonical_source_types(frame, config)
    if frame.duplicated(["source1_entity_id", "candidate_entity_id"]).any():
        raise ValueError("Candidate pair IDs must be unique")
    frame["_m3_row"] = np.arange(len(frame), dtype=np.int64)

    reference_view = _normalized_records(source1, "s1", config)
    frame = frame.merge(
        reference_view,
        left_on="source1_entity_id",
        right_on="entity_id",
        how="left",
        validate="many_to_one",
        sort=False,
    ).drop(columns="entity_id")
    if frame["name_basic_norm_s1"].isna().any():
        raise ValueError("Some candidate pairs have no matching normalized Source 1 row")

    for source_type, source_records in (("S2", source2), ("S3", source3)):
        subset_mask = frame["source_type"].eq(source_type)
        if not subset_mask.any():
            continue
        target_view = _normalized_records(source_records, "s2", config)
        subset = frame.loc[subset_mask].merge(
            target_view,
            left_on="candidate_entity_id",
            right_on="entity_id",
            how="left",
            validate="many_to_one",
            sort=False,
        ).drop(columns="entity_id")
        for column in target_view.columns:
            if column != "entity_id":
                frame.loc[subset_mask, column] = subset[column].to_numpy()
    frame = frame.sort_values("_m3_row", kind="stable").drop(columns="_m3_row")
    frame = frame.reset_index(drop=True)
    if frame["name_basic_norm_s2"].isna().any():
        raise ValueError("Some candidate pairs have no matching normalized target row")
    return frame


def attach_labels(
    pairs: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> pd.DataFrame:
    """Label candidate pairs from one-to-many M4-style ground truth."""
    if "source1_entity_id" not in ground_truth.columns:
        raise ValueError("Ground truth requires source1_entity_id")
    if "candidate_entity_id" in ground_truth.columns:
        label_column = next(
            (column for column in ("label", "is_match") if column in ground_truth.columns),
            None,
        )
        if label_column is None:
            raise ValueError("Pairwise ground truth requires label or is_match")
        truth_pairs = ground_truth[
            ["source1_entity_id", "candidate_entity_id", label_column]
        ].copy()
        truth_pairs["source1_entity_id"] = truth_pairs["source1_entity_id"].astype(str)
        truth_pairs["candidate_entity_id"] = truth_pairs["candidate_entity_id"].astype(str)
        truth_pairs[label_column] = truth_pairs[label_column].astype(np.int8)
        if truth_pairs.duplicated(["source1_entity_id", "candidate_entity_id"]).any():
            raise ValueError("Ground truth contains duplicate candidate pairs")
        result = pairs.merge(
            truth_pairs.rename(columns={label_column: "label"}),
            on=["source1_entity_id", "candidate_entity_id"],
            how="left",
            validate="one_to_one",
            sort=False,
        )
        if result["label"].isna().any():
            raise ValueError("Pairwise ground truth does not cover every candidate pair")
        result["label"] = result["label"].astype(np.int8)
        return result

    if "matched_entity_ids" not in ground_truth.columns:
        raise ValueError("Ground truth requires matched_entity_ids")
    truth: dict[str, set[str]] = {}
    for row in ground_truth.itertuples(index=False):
        source_id = str(getattr(row, "source1_entity_id"))
        value = getattr(row, "matched_entity_ids")
        if isinstance(value, (list, tuple, set)):
            matched = {str(item).strip() for item in value if str(item).strip()}
        elif value is None or pd.isna(value):
            matched = set()
        else:
            matched = {
                item.strip()
                for item in str(value).split("|")
                if item.strip()
            }
        if source_id in truth:
            raise ValueError(f"Duplicate Source 1 ground-truth row for {source_id!r}")
        truth[source_id] = matched

    source_ids = pairs["source1_entity_id"].astype(str)
    missing_source_ids = sorted(set(source_ids) - set(truth))
    if missing_source_ids:
        raise ValueError(
            "Ground truth has no row for Source 1 IDs: "
            + ", ".join(missing_source_ids[:10])
        )
    labels = [
        int(candidate_id in truth[source_id])
        for source_id, candidate_id in zip(
            source_ids,
            pairs["candidate_entity_id"].astype(str),
        )
    ]
    result = pairs.copy()
    result["label"] = np.asarray(labels, dtype=np.int8)
    return result


def build_feature_table(
    pairs: pd.DataFrame,
    config: dict[str, Any] = CONFIG,
    *,
    source1: pd.DataFrame | None = None,
    source2: pd.DataFrame | None = None,
    source3: pd.DataFrame | None = None,
    ground_truth: pd.DataFrame | None = None,
    tfidf_vectorizers: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Assemble a numeric model table keyed by the original candidate pair IDs."""
    df = pairs.copy().reset_index(drop=True)
    source_frames = (source1, source2, source3)
    if any(frame is not None for frame in source_frames):
        if any(frame is None for frame in source_frames):
            raise ValueError("Provide all of source1, source2, and source3 normalized data")
        df = enrich_candidate_pairs(df, source1, source2, source3, config)

    required_ids = {"source1_entity_id", "candidate_entity_id", "source_type"}
    if not required_ids.issubset(df.columns):
        raise ValueError(
            "Feature construction requires source1_entity_id, candidate_entity_id, "
            "and source_type"
        )

    if ground_truth is not None:
        df = attach_labels(df, ground_truth)

    id_columns = ["source1_entity_id", "candidate_entity_id", "source_type"]
    feature_table = df[id_columns].copy()
    generated: list[pd.DataFrame] = []

    if _enabled(config, "string_similarity"):
        generated.append(build_string_features(df, config))
    if _enabled(config, "tfidf"):
        vectorizers = tfidf_vectorizers
        if vectorizers is None:
            vectorizers = fit_tfidf_vectorizers(df, config)
        generated.append(
            build_tfidf_features(df, config, vectorizers=vectorizers)
        )
    if _enabled(config, "embedding"):
        generated.append(build_embedding_features(df, config))
    if _enabled(config, "relative_features"):
        relative_input = df.copy()
        score_column = next(
            (
                column
                for column in (
                    "candidate_score",
                    "blocking_score",
                    "similarity_score",
                    "score",
                )
                if column in relative_input.columns
            ),
            None,
        )
        if score_column is not None:
            relative_input["candidate_score"] = pd.to_numeric(
                relative_input[score_column],
                errors="raise",
            ).astype(np.float32)
        else:
            name_similarity = (
                generated[0]["name_jaro_winkler"].to_numpy(dtype=np.float32)
                if generated and "name_jaro_winkler" in generated[0]
                else np.zeros(len(df), dtype=np.float32)
            )
            address_similarity = (
                generated[0]["address_jaro_winkler"].to_numpy(dtype=np.float32)
                if generated and "address_jaro_winkler" in generated[0]
                else np.zeros(len(df), dtype=np.float32)
            )
            relative_input["candidate_score"] = (
                0.7 * name_similarity + 0.3 * address_similarity
            ).astype(np.float32)
        generated.append(build_relative_features(relative_input, config))
        feature_table["candidate_score"] = relative_input[
            "candidate_score"
        ].to_numpy(dtype=np.float32)

    for feature_frame in generated:
        feature_columns = [
            column
            for column in feature_frame.columns
            if column not in id_columns and column not in feature_table.columns
        ]
        if feature_columns:
            feature_table = pd.concat(
                [
                    feature_table,
                    feature_frame[feature_columns].reset_index(drop=True),
                ],
                axis=1,
            )

    if "label" in df.columns:
        feature_table["label"] = pd.to_numeric(df["label"], errors="raise").astype(
            np.int8
        )

    numeric_columns = feature_table.select_dtypes(include=[np.number]).columns
    feature_table[numeric_columns] = feature_table[numeric_columns].astype(np.float32)
    if "label" in feature_table.columns:
        feature_table["label"] = feature_table["label"].astype(np.int8)
    numeric_values = feature_table.select_dtypes(include=[np.number]).to_numpy(
        dtype=np.float32
    )
    if not np.isfinite(numeric_values).all():
        raise ValueError("Feature table contains NaN or infinite numeric values")
    if feature_table.duplicated(list(PAIR_KEYS[:2])).any():
        raise ValueError("Feature table contains duplicate candidate pair IDs")
    return feature_table


def _read_table(path: str) -> pd.DataFrame:
    if path.casefold().endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    separator = "\t" if path.casefold().endswith(".tsv") else ","
    return pd.read_csv(path, sep=separator, dtype=str, keep_default_na=False)


def main() -> None:
    """Build train/test parquet artifacts from M1 normalized data and M2 candidates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, help="M2 candidate-pair TSV")
    parser.add_argument("--source1", required=True, help="Normalized Source 1 table")
    parser.add_argument("--source2", required=True, help="Normalized Source 2 table")
    parser.add_argument("--source3", required=True, help="Normalized Source 3 table")
    parser.add_argument("--ground-truth", help="Training ground-truth TSV")
    parser.add_argument(
        "--output",
        default="output/feature_table.parquet",
        help="Feature table output path",
    )
    parser.add_argument(
        "--enriched-output",
        default="output/enriched_candidate_pairs.parquet",
        help="Aligned normalized pair text for fold-specific TF-IDF fitting",
    )
    args = parser.parse_args()

    candidates = _read_table(args.candidates)
    source1 = _read_table(args.source1)
    source2 = _read_table(args.source2)
    source3 = _read_table(args.source3)
    truth = _read_table(args.ground_truth) if args.ground_truth else None
    enriched = enrich_candidate_pairs(candidates, source1, source2, source3)
    table = build_feature_table(
        enriched,
        ground_truth=truth,
    )
    feature_path = Path(args.output)
    pair_path = Path(args.enriched_output)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    pair_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(feature_path, index=False)
    enriched.to_parquet(pair_path, index=False)
    print(
        f"Wrote {len(table)} candidate feature rows to {feature_path}; "
        f"fold-safe pair text to {pair_path}"
    )


if __name__ == "__main__":
    main()
