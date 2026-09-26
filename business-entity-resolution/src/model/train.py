"""Entity-level model validation, feature ablation, and hard-negative mining."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold

from src.features.tfidf_features import build_tfidf_features
from src.features.tfidf_features import fit_tfidf_vectorizers
from src.utils.config_loader import CONFIG, get_config_value


IDENTIFIER_COLUMNS = {
    "source1_entity_id",
    "candidate_entity_id",
    "entity_id",
    "source_type",
}
TFIDF_COLUMNS = ("name_tfidf_cosine", "address_tfidf_cosine")


def _metrics(y_true: Iterable[int], probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(list(y_true), dtype=np.int8)
    predictions = (probabilities >= 0.5).astype(np.int8)
    return {
        "f0.5": float(fbeta_score(labels, predictions, beta=0.5, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
    }


def entity_cv_splits(
    groups: pd.Series,
    n_splits: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return GroupKFold splits and assert that no entity crosses a fold."""
    unique_groups = groups.nunique(dropna=False)
    if n_splits < 2 or unique_groups < n_splits:
        raise ValueError(
            f"Entity-level CV requires at least {n_splits} distinct entities; "
            f"received {unique_groups}"
        )
    splitter = GroupKFold(n_splits=n_splits)
    indices = np.arange(len(groups))
    splits = list(splitter.split(indices, groups=groups))
    for train_indices, validation_indices in splits:
        train_entities = set(groups.iloc[train_indices])
        validation_entities = set(groups.iloc[validation_indices])
        if not train_entities.isdisjoint(validation_entities):
            raise AssertionError("Entity overlap detected between train and validation")
    return splits


def _model(config: dict[str, Any]) -> lgb.LGBMClassifier:
    model_type = get_config_value(config, "model", "type").casefold()
    if model_type != "lightgbm":
        raise ValueError(
            f"Unsupported configured model type {model_type!r}; only lightgbm is installed"
        )
    params = dict(get_config_value(config, "model", "params"))
    return lgb.LGBMClassifier(**params)


def _model_matrix(
    features: pd.DataFrame,
    indices: np.ndarray,
) -> pd.DataFrame:
    excluded = IDENTIFIER_COLUMNS | {"label"}
    selected = features.iloc[indices]
    matrix = selected.drop(
        columns=[column for column in excluded if column in selected.columns],
        errors="ignore",
    )
    if matrix.empty:
        raise ValueError("No numeric feature columns are available for training")
    if not all(pd.api.types.is_numeric_dtype(matrix[column]) for column in matrix):
        raise TypeError("Model features must be numeric; remove metadata columns")
    values = matrix.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Model features contain NaN or infinite values")
    return matrix.astype(np.float32)


def _fold_features(
    features: pd.DataFrame,
    pair_text: pd.DataFrame | None,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    config: dict[str, Any],
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_matrix = _model_matrix(features, train_indices)
    validation_matrix = _model_matrix(features, validation_indices)
    if columns is not None:
        train_matrix = train_matrix.reindex(columns=columns, fill_value=0.0)
        validation_matrix = validation_matrix.reindex(columns=columns, fill_value=0.0)
    if pair_text is not None and all(
        column in pair_text.columns
        for column in (
            "name_basic_norm_s1",
            "name_basic_norm_s2",
            "address_basic_norm_s1",
            "address_basic_norm_s2",
        )
    ):
        fold_tfidf = build_tfidf_features(
            pair_text.reset_index(drop=True),
            config,
            fit_rows=train_indices,
        )
        for column in TFIDF_COLUMNS:
            if column in train_matrix.columns:
                train_matrix[column] = fold_tfidf.iloc[train_indices][
                    column
                ].to_numpy(dtype=np.float32)
                validation_matrix[column] = fold_tfidf.iloc[validation_indices][
                    column
                ].to_numpy(dtype=np.float32)
    return train_matrix, validation_matrix


def _out_of_fold_probabilities(
    features: pd.DataFrame,
    labels: pd.Series,
    groups: pd.Series,
    config: dict[str, Any],
    pair_text: pd.DataFrame | None = None,
    columns: list[str] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    folds = int(get_config_value(config, "model", "cv", "n_folds"))
    splits = entity_cv_splits(groups.reset_index(drop=True), folds)
    oof = np.full(len(features), np.nan, dtype=np.float32)
    fold_details: list[dict[str, Any]] = []
    for fold_number, (train_indices, validation_indices) in enumerate(splits, start=1):
        train_matrix, validation_matrix = _fold_features(
            features,
            pair_text,
            train_indices,
            validation_indices,
            config,
            columns,
        )
        model = _model(config)
        model.fit(train_matrix, labels.iloc[train_indices].to_numpy(dtype=np.int8))
        probabilities = model.predict_proba(validation_matrix)[:, 1].astype(
            np.float32
        )
        oof[validation_indices] = probabilities
        fold_details.append(
            {
                "fold": fold_number,
                **_metrics(labels.iloc[validation_indices], probabilities),
                "train_entities": groups.iloc[train_indices].nunique(),
                "validation_entities": groups.iloc[validation_indices].nunique(),
            }
        )
    if np.isnan(oof).any() or not np.isfinite(oof).all():
        raise AssertionError("Out-of-fold predictions are incomplete or non-finite")
    return oof, fold_details


def _feature_groups(features: pd.DataFrame) -> dict[str, list[str]]:
    numeric = [
        column
        for column in features.select_dtypes(include=[np.number]).columns
        if column not in IDENTIFIER_COLUMNS and column != "label"
    ]
    groups: dict[str, list[str]] = {
        "string": [
            column
            for column in numeric
            if column.endswith(("_jaccard", "_levenshtein", "_jaro_winkler"))
            or column in {"country_match", "phonetic_match", "source_is_s3"}
            or "component" in column
            or column.endswith("_match")
        ],
        "tfidf": [column for column in numeric if column.endswith("_tfidf_cosine")],
        "embeddings": [
            column for column in numeric if "_embedding_cosine" in column
        ],
        "relative": [
            column
            for column in numeric
            if column
            in {
                "candidate_score",
                "candidate_rank",
                "top_score",
                "score_gap_from_top",
                "score_gap_from_second",
                "number_of_candidates",
                "number_above_0_5",
                "number_above_0_7",
            }
        ],
    }
    groups = {name: columns for name, columns in groups.items() if columns}
    covered = {column for columns in groups.values() for column in columns}
    remainder = [column for column in numeric if column not in covered]
    if remainder:
        groups["other"] = remainder
    return groups


def run_feature_ablation(
    features: pd.DataFrame,
    labels: pd.Series,
    groups: pd.Series | None = None,
    *,
    config: dict[str, Any] = CONFIG,
    pair_text: pd.DataFrame | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Compare cumulative feature groups using entity-disjoint OOF predictions."""
    if groups is None:
        if "source1_entity_id" not in features.columns:
            raise ValueError("Feature ablation requires source1_entity_id groups")
        groups = features["source1_entity_id"]
    feature_groups = _feature_groups(features)
    cumulative: list[str] = []
    rows: list[dict[str, Any]] = []
    for group_name in ("string", "tfidf", "embeddings", "relative", "other"):
        selected = feature_groups.get(group_name)
        if not selected:
            continue
        cumulative.extend(selected)
        probabilities, _ = _out_of_fold_probabilities(
            features,
            labels,
            groups,
            config,
            pair_text,
            columns=cumulative,
        )
        rows.append(
            {
                "feature_set": " + ".join(
                    name
                    for name in ("string", "tfidf", "embeddings", "relative", "other")
                    if name in feature_groups and any(
                        item in cumulative for item in feature_groups[name]
                    )
                ),
                "features": len(cumulative),
                **_metrics(labels, probabilities),
            }
        )

    report = pd.DataFrame(rows)
    destination = Path(
        output_path
        or get_config_value(config, "model", "ablation", "report_out")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Feature Ablation",
        "",
        "Scores use entity-disjoint out-of-fold predictions at probability 0.5.",
        "",
        "| Feature set | Features | F0.5 | Precision | Recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['feature_set']} | {row['features']} | "
            f"{row['f0.5']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} |"
        )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _source_metrics(
    features: pd.DataFrame,
    labels: pd.Series,
    probabilities: np.ndarray,
    ground_truth: pd.DataFrame | None = None,
    config: dict[str, Any] = CONFIG,
) -> pd.DataFrame:
    if "source_type" not in features.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for source_type, indices in features.groupby("source_type", sort=True).indices.items():
        selected = np.asarray(indices, dtype=np.int64)
        row = {"source_type": source_type, **_metrics(labels.iloc[selected], probabilities[selected])}
        if ground_truth is not None and "matched_entity_ids" in ground_truth.columns:
            truth_ids: set[str] = set()
            for value in ground_truth["matched_entity_ids"].dropna():
                truth_ids.update(
                    item.strip() for item in str(value).split("|") if item.strip()
                )
            candidate_positive_ids = set(
                features.iloc[selected]
                .loc[labels.iloc[selected].to_numpy(dtype=bool), "candidate_entity_id"]
                .astype(str)
            )
            prefixes = get_config_value(config, "schema", "source_prefixes")
            source_prefix = prefixes[
                "source2" if source_type == "S2" else "source3"
            ]
            denominator = sum(item.startswith(source_prefix) for item in truth_ids)
            row["blocking_recall"] = (
                len(candidate_positive_ids & truth_ids) / denominator
                if denominator
                else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def find_hard_negatives(
    probabilities: np.ndarray,
    labels: pd.Series,
    *,
    threshold: float = 0.7,
    source1_entity_ids: pd.Series | None = None,
    candidate_entity_ids: pd.Series | None = None,
) -> pd.DataFrame:
    """Return false-positive OOF pairs above a fixed, explicit mining threshold."""
    values = np.asarray(probabilities, dtype=np.float32)
    if len(values) != len(labels) or not np.isfinite(values).all():
        raise ValueError("Hard-negative probabilities must be finite and label-aligned")
    mask = (values >= threshold) & (labels.to_numpy(dtype=np.int8) == 0)
    result = pd.DataFrame(
        {
            "oof_probability": values[mask],
            "label": np.zeros(int(mask.sum()), dtype=np.int8),
        }
    )
    if source1_entity_ids is not None:
        result.insert(
            0,
            "source1_entity_id",
            source1_entity_ids.reset_index(drop=True).astype(str).to_numpy()[mask],
        )
    if candidate_entity_ids is not None:
        result.insert(
            1 if source1_entity_ids is not None else 0,
            "candidate_entity_id",
            candidate_entity_ids.reset_index(drop=True).astype(str).to_numpy()[mask],
        )
    return result.sort_values("oof_probability", ascending=False, ignore_index=True)


def save_hard_negative_report(
    hard_negatives: pd.DataFrame,
    *,
    precision_before: float,
    precision_after: float | None,
    destination: str | Path,
) -> None:
    """Write mined false-positive counts and validation precision delta."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    delta = (
        f"{precision_after - precision_before:+.4f}"
        if precision_after is not None
        else "not measured"
    )
    destination.write_text(
        "\n".join(
            [
                "# Hard-Negative Mining Report",
                "",
                f"- OOF false positives above mining threshold: {len(hard_negatives)}",
                f"- Precision before mining: {precision_before:.4f}",
                (
                    f"- Precision after mining: {precision_after:.4f}"
                    if precision_after is not None
                    else "- Precision after mining: not measured"
                ),
                f"- Precision delta: {delta}",
                "",
                "Hard negatives are mined from entity-disjoint out-of-fold predictions.",
                "The augmented model is retrained and evaluated on the unchanged OOF folds.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def train_model(
    features: pd.DataFrame,
    labels: pd.Series | None = None,
    config: dict[str, Any] = CONFIG,
    *,
    pair_text: pd.DataFrame | None = None,
    ground_truth: pd.DataFrame | None = None,
    model_path: str | Path | None = None,
    baseline_model_path: str | Path = "artifacts/m3_model_baseline.pkl",
) -> lgb.LGBMClassifier:
    """Run OOF evaluation/ablation/mining, then fit and persist the final model."""
    if labels is None:
        if "label" not in features.columns:
            raise ValueError("Training features require labels or a label column")
        labels = features["label"]
    labels = pd.Series(
        np.asarray(labels),
        index=features.index,
    ).astype(np.int8).reset_index(drop=True)
    features = features.reset_index(drop=True)
    if len(features) != len(labels):
        raise ValueError("Labels and feature rows must be aligned")
    group_column = "source1_entity_id"
    if group_column not in features.columns:
        raise ValueError("Training requires source1_entity_id for entity-level CV")
    entity_groups = features[group_column].astype(str).reset_index(drop=True)
    if labels.nunique() < 2:
        raise ValueError("Training requires both positive and negative candidate pairs")

    raw_pairs = pair_text.reset_index(drop=True) if pair_text is not None else None
    if raw_pairs is not None and len(raw_pairs) != len(features):
        raise ValueError("pair_text and feature table have different row counts")
    tfidf_settings = get_config_value(config, "features", "tfidf")
    tfidf_enabled = (
        bool(tfidf_settings.get("enabled", True))
        if isinstance(tfidf_settings, dict)
        else bool(tfidf_settings)
    )
    if tfidf_enabled and raw_pairs is None:
        raise ValueError(
            "Pass enriched pair_text to training so TF-IDF vocabularies are fit "
            "within each entity-level training fold"
        )
    if tfidf_enabled and raw_pairs is not None:
        tfidf_text_columns = {
            "name_basic_norm_s1",
            "name_basic_norm_s2",
            "address_basic_norm_s1",
            "address_basic_norm_s2",
        }
        missing_text_columns = tfidf_text_columns - set(raw_pairs.columns)
        if missing_text_columns:
            raise ValueError(
                "Fold-safe TF-IDF training requires enriched text columns: "
                + ", ".join(sorted(missing_text_columns))
            )
    if raw_pairs is not None:
        for identifier in ("source1_entity_id", "candidate_entity_id"):
            if identifier in features.columns and identifier in raw_pairs.columns:
                if not features[identifier].astype(str).reset_index(drop=True).equals(
                    raw_pairs[identifier].astype(str).reset_index(drop=True)
                ):
                    raise ValueError(
                        f"pair_text is misaligned with feature rows by {identifier}"
                    )

    oof_probabilities, fold_scores = _out_of_fold_probabilities(
        features,
        labels,
        entity_groups,
        config,
        raw_pairs,
    )
    report_dir = Path(get_config_value(config, "paths", "reports_dir"))
    report_dir.mkdir(parents=True, exist_ok=True)
    cv_report = pd.DataFrame(fold_scores)
    cv_report.to_csv(report_dir / "model_cv_report.csv", index=False)
    mean_metrics = _metrics(labels, oof_probabilities)
    source_report = _source_metrics(
        features,
        labels,
        oof_probabilities,
        ground_truth,
        config,
    )
    if not source_report.empty:
        source_report.to_csv(report_dir / "per_source_performance.csv", index=False)

    ablation_settings = get_config_value(config, "model", "ablation")
    if ablation_settings.get("enabled", True):
        run_feature_ablation(
            features,
            labels,
            entity_groups,
            config=config,
            pair_text=raw_pairs,
            output_path=ablation_settings.get(
                "report_out", "reports/feature_ablation.md"
            ),
        )

    hard_negative_settings = get_config_value(
        config, "model", "hard_negative_mining"
    )
    hard_negative_path = Path(
        hard_negative_settings.get(
            "report_out", "reports/hard_negative_report.md"
        )
    )
    precision_before = mean_metrics["precision"]
    precision_after: float | None = None
    extra = pd.DataFrame(columns=features.columns)
    mined = pd.DataFrame()
    if hard_negative_settings.get("enabled", True):
        threshold = float(hard_negative_settings.get("threshold", 0.7))
        mined = find_hard_negatives(
            oof_probabilities,
            labels,
            threshold=threshold,
            source1_entity_ids=features[group_column],
            candidate_entity_ids=features.get("candidate_entity_id"),
        )
        if not mined.empty and "candidate_entity_id" in features.columns:
            augmented = features.copy()
            augmented["label"] = labels
            positioned_features = features.copy()
            positioned_features["_position"] = np.arange(len(features))
            selected = mined.merge(
                positioned_features,
                on=["source1_entity_id", "candidate_entity_id"],
                how="inner",
                validate="one_to_one",
            )
            extra = features.iloc[selected["_position"].to_numpy()].copy()
            extra["label"] = 0
            augmented = pd.concat([augmented, extra], ignore_index=True)
            augmented_groups = augmented[group_column].astype(str)
            augmented_labels = augmented["label"].astype(np.int8)
            augmented_pair_text = (
                pd.concat(
                    [
                        raw_pairs,
                        raw_pairs.iloc[selected["_position"].to_numpy()],
                    ],
                    ignore_index=True,
                )
                if raw_pairs is not None
                else None
            )
            augmented_probabilities, _ = _out_of_fold_probabilities(
                augmented,
                augmented_labels,
                augmented_groups,
                config,
                augmented_pair_text,
            )
            # Evaluate only the original held-out rows to measure mining impact.
            precision_after = _metrics(
                labels,
                augmented_probabilities[: len(features)],
            )["precision"]

        save_hard_negative_report(
            mined,
            precision_before=precision_before,
            precision_after=precision_after,
            destination=hard_negative_path,
        )

    final_features = (
        pd.concat([features, extra], ignore_index=True)
        if not extra.empty
        else features
    )
    final_labels = (
        np.concatenate(
            [
                labels.to_numpy(dtype=np.int8),
                np.zeros(len(extra), dtype=np.int8),
            ]
        )
        if not extra.empty
        else labels.to_numpy(dtype=np.int8)
    )
    baseline_matrix = _model_matrix(features, np.arange(len(features)))
    baseline_model = _model(config)
    baseline_model.fit(baseline_matrix, labels.to_numpy(dtype=np.int8))
    final_matrix = _model_matrix(final_features, np.arange(len(final_features)))
    final_model = _model(config)
    final_model.fit(final_matrix, final_labels)
    vectorizers = (
        fit_tfidf_vectorizers(raw_pairs, config)
        if raw_pairs is not None and tfidf_enabled
        else None
    )
    baseline_artifact = Path(baseline_model_path)
    hard_negative_artifact = Path(
        model_path or "artifacts/m3_model_hard_negative.pkl"
    )
    artifacts = (
        (baseline_artifact, baseline_model, baseline_matrix),
        (hard_negative_artifact, final_model, final_matrix),
    )
    for artifact_path, estimator, matrix in artifacts:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": estimator,
                "feature_columns": matrix.columns.tolist(),
                "metrics": mean_metrics,
                "tfidf_vectorizers": vectorizers,
            },
            artifact_path,
        )
    final_model.m3_feature_columns_ = final_matrix.columns.tolist()
    final_model.m3_cv_metrics_ = mean_metrics
    final_model.m3_source_report_ = source_report
    final_model.m3_baseline_model_path_ = str(baseline_artifact)
    final_model.m3_hard_negative_model_path_ = str(hard_negative_artifact)
    final_model.m3_hard_negatives_ = (
        mined if hard_negative_settings.get("enabled", True) else pd.DataFrame()
    )
    return final_model


def main() -> None:
    """Train, cross-validate, ablate, mine hard negatives, and save the model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        default="output/feature_table.parquet",
        help="Labeled feature table from src.features.build_feature_table",
    )
    parser.add_argument(
        "--pair-text",
        default="output/enriched_candidate_pairs.parquet",
        help="Aligned normalized candidate pairs used to refit TF-IDF per fold",
    )
    parser.add_argument(
        "--ground-truth",
        help="Optional ground-truth TSV for source-level blocking recall",
    )
    parser.add_argument(
        "--model",
        default="artifacts/m3_model.pkl",
        help="Trained model artifact path",
    )
    parser.add_argument(
        "--baseline-model",
        default="artifacts/m3_model_baseline.pkl",
        help="Baseline model artifact path",
    )
    args = parser.parse_args()

    feature_table = pd.read_parquet(args.features)
    pair_text = pd.read_parquet(args.pair_text)
    ground_truth = (
        pd.read_csv(args.ground_truth, sep="\t", dtype=str, keep_default_na=False)
        if args.ground_truth
        else None
    )
    model = train_model(
        feature_table,
        config=CONFIG,
        pair_text=pair_text,
        ground_truth=ground_truth,
        model_path=args.model,
        baseline_model_path=args.baseline_model,
    )
    print(f"Saved model to {args.model}")
    print(f"Entity-level OOF metrics: {model.m3_cv_metrics_}")


if __name__ == "__main__":
    main()
