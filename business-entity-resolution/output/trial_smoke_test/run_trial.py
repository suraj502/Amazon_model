"""Run an isolated first end-to-end submission smoke test."""

from __future__ import annotations

import shutil
import zipfile
from copy import deepcopy
from pathlib import Path

import joblib
import pandas as pd

from src.blocking.candidate_generation import apply_candidate_cap, generate_union_candidates
from src.blocking.candidate_union import union_candidates
from src.features.build_feature_table import build_feature_table, enrich_candidate_pairs
from src.model.predict import predict_matches, write_predictions
from src.normalization.address_normalize import normalize_addresses
from src.normalization.name_normalize import normalize_names
from src.submission.build_submission import build_submission
from src.submission.validate_submission import validate_submission
from src.utils.config_loader import CONFIG


ROOT = Path(__file__).resolve().parents[2]
TRIAL = Path(__file__).resolve().parent
RAW = TRIAL / "raw"
NORMALIZED = TRIAL / "normalized"


def _copy_sample(source_name: str, rows: int) -> Path:
    source = ROOT / "data" / "raw" / source_name
    destination = RAW / source_name
    frame = pd.read_csv(source, sep="\t", dtype=str, keep_default_na=False, nrows=rows)
    if list(frame.columns) != ["entity_id", "business_name", "business_address", "country"]:
        raise ValueError(f"Unexpected schema in {source_name}: {list(frame.columns)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, sep="\t", index=False)
    return destination


def _normalize(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype={"entity_id": "string"}, keep_default_na=False)
    normalized = normalize_addresses(normalize_names(frame, CONFIG), CONFIG)
    return normalized


def main() -> None:
    if TRIAL.exists():
        for child in TRIAL.iterdir():
            if child.name != Path(__file__).name:
                shutil.rmtree(child) if child.is_dir() else child.unlink()

    s1_raw = _copy_sample("test_source1.tsv", 100)
    s2_raw = _copy_sample("test_source2.tsv", 1000)
    s3_raw = _copy_sample("test_source3.tsv", 1000)
    source1 = _normalize(s1_raw)
    source2 = _normalize(s2_raw)
    source3 = _normalize(s3_raw)
    NORMALIZED.mkdir(parents=True, exist_ok=True)
    source1.to_parquet(NORMALIZED / "test_source1_normalized.parquet", index=False)
    source2.to_parquet(NORMALIZED / "test_source2_normalized.parquet", index=False)
    source3.to_parquet(NORMALIZED / "test_source3_normalized.parquet", index=False)

    config = deepcopy(CONFIG)
    config["features"]["embedding"] = {"enabled": False}
    config["blocking"]["max_candidates_per_entity"] = 50

    candidates_s2 = generate_union_candidates(source1, source2, config)
    candidates_s3 = generate_union_candidates(source1, source3, config)
    candidates = apply_candidate_cap(
        union_candidates([candidates_s2, candidates_s3], config), config
    )

    # Keep the official candidate-set contract complete even when a tiny sample
    # has no lexical blocker hit for an S1 row.
    covered = set(candidates["source1_entity_id"])
    fallback_id = str(source2["entity_id"].iloc[0])
    missing = source1.loc[~source1["entity_id"].astype(str).isin(covered), "entity_id"].astype(str)
    fallback = pd.DataFrame(
        {"source1_entity_id": missing, "candidate_entity_id": fallback_id}
    )
    candidates = pd.concat([candidates, fallback], ignore_index=True).drop_duplicates(
        ["source1_entity_id", "candidate_entity_id"]
    )
    valid_target_ids = set(source2["entity_id"].astype(str)) | set(source3["entity_id"].astype(str))
    if not set(candidates["candidate_entity_id"]).issubset(valid_target_ids):
        raise AssertionError("Blocking emitted an ID outside the copied S2/S3 tables")
    candidates = candidates.sort_values(["source1_entity_id", "candidate_entity_id"]).reset_index(drop=True)
    pair_path = TRIAL / "candidate_pairs_blocked.tsv"
    candidates.to_csv(pair_path, sep="\t", index=False)

    enriched = enrich_candidate_pairs(candidates, source1, source2, source3, config)
    artifact = joblib.load(ROOT / "artifacts" / "m3_model.pkl")
    features = build_feature_table(
        enriched,
        config=config,
        tfidf_vectorizers=artifact["tfidf_vectorizers"],
    )
    feature_path = TRIAL / "test_feature_table.parquet"
    features.to_parquet(feature_path, index=False)
    predictions = predict_matches(artifact, features, config)
    prediction_path = TRIAL / "m3_predictions.tsv"
    write_predictions(predictions, prediction_path)

    selected = predictions[predictions["probability"] > 0.25]
    matched = selected.rename(columns={"candidate_entity_id": "matched_entity_ids"})[
        ["source1_entity_id", "matched_entity_ids"]
    ]
    matched = matched.groupby("source1_entity_id", sort=False)["matched_entity_ids"].agg(
        lambda values: "|".join(sorted(set(values.astype(str))))
    ).reset_index()
    all_s1 = pd.DataFrame({"source1_entity_id": source1["entity_id"].astype(str)})
    matches = all_s1.merge(matched, on="source1_entity_id", how="left")
    matches["matched_entity_ids"] = matches["matched_entity_ids"].fillna("")
    candidate_sets = candidates.groupby("source1_entity_id", sort=False)["candidate_entity_id"].agg(
        lambda values: "|".join(sorted(set(values.astype(str))))
    ).reset_index().rename(columns={"candidate_entity_id": "candidate_entity_ids"})
    matches = matches.merge(candidate_sets, on="source1_entity_id", how="left")
    submission_config = deepcopy(config)
    submission_config["paths"]["candidate_pairs_out"] = str(TRIAL / "candidate_pairs.tsv")
    matching_path = TRIAL / "matching_results.tsv"
    build_submission(matches, matching_path, submission_config)
    validation_ok = validate_submission(matching_path, submission_config)

    zip_path = TRIAL / "trial_submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(matching_path, "matching_results.tsv")
        archive.write(TRIAL / "candidate_pairs.tsv", "candidate_pairs.tsv")

    candidate_count = sum(
        len(str(value).split("|"))
        for value in pd.read_csv(TRIAL / "candidate_pairs.tsv", sep="\t", dtype=str)["candidate_entity_ids"]
        if value
    )
    print(f"dummy_s1_entities={len(source1)}")
    print(f"candidate_pairs={candidate_count}")
    predicted_match_count = sum(
        len(str(value).split("|")) for value in matches["matched_entity_ids"] if value
    )
    print(f"predicted_matches={predicted_match_count}")
    print("m4_threshold=0.25")
    print(f"validation={validation_ok}")
    print(f"matching_results={matching_path}")
    print(f"candidate_pairs_path={TRIAL / 'candidate_pairs.tsv'}")
    print(f"zip_path={zip_path}")


if __name__ == "__main__":
    main()