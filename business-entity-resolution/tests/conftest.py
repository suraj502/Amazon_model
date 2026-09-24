"""Shared synthetic records for contract tests across all pipeline stages."""

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def source1_rows() -> pd.DataFrame:
    """Return six reference entities, including an entity with no match."""
    return pd.DataFrame(
        [
            {"entity_id": "S1-001", "business_name": "Acme Corporation", "business_address": "1 Main St, Austin, TX 78701", "country": "US"},
            {"entity_id": "S1-002", "business_name": "Tata Consultancy Services", "business_address": "5 MG Road, Bengaluru 560001", "country": "IN"},
            {"entity_id": "S1-003", "business_name": "Cafe de Paris", "business_address": "10 Rue de Lyon, Paris 75001", "country": "FR"},
            {"entity_id": "S1-004", "business_name": "Muller & Fils", "business_address": "2 Rue Victor Hugo, Lyon 69001", "country": "FR"},
            {"entity_id": "S1-005", "business_name": "Northwind Logistics", "business_address": "99 Market Street, Seattle, WA 98101", "country": "US"},
            {"entity_id": "S1-006", "business_name": "No Match Holdings", "business_address": "Unknown Road, Pune 411001", "country": "IN"},
        ]
    )


@pytest.fixture
def source2_rows() -> pd.DataFrame:
    """Return synthetic Source 2 records with noisy variants."""
    return pd.DataFrame(
        [
            {"entity_id": "S2-101", "business_name": "Acme Corp.", "business_address": "1 Main Street, Austin TX 78701", "country": "US"},
            {"entity_id": "S2-102", "business_name": "TCS", "business_address": "5 M G Rd, Bengaluru 560001", "country": "IN"},
            {"entity_id": "S2-103", "business_name": "Cafe de Paris SARL", "business_address": "10 Rue de Lyon, Paris 75001", "country": "FR"},
            {"entity_id": "S2-104", "business_name": "Northwind Logistics", "business_address": "99 Market St, Seattle WA 98101", "country": "US"},
            {"entity_id": "S2-105", "business_name": "Other Trading Co", "business_address": "8 Park Road, Delhi 110001", "country": "IN"},
        ]
    )


@pytest.fixture
def source3_rows() -> pd.DataFrame:
    """Return synthetic Source 3 records including a transliterated name."""
    return pd.DataFrame(
        [
            {"entity_id": "S3-201", "business_name": "ACME Incorporated", "business_address": "1 Main St Austin TX", "country": "US"},
            {"entity_id": "S3-202", "business_name": "Tata Konsultansi Services", "business_address": "5 MG Road Bengaluru 560001", "country": "IN"},
            {"entity_id": "S3-203", "business_name": "Café de Paris", "business_address": "10 Rue de Lyon Paris", "country": "FR"},
            {"entity_id": "S3-204", "business_name": "Mueller et Fils", "business_address": "2 Rue Victor Hugo Lyon", "country": "FR"},
            {"entity_id": "S3-205", "business_name": "Northwind Logistik", "business_address": "99 Market Street Seattle WA", "country": "US"},
        ]
    )


@pytest.fixture
def ground_truth_path(tmp_path: Path, source1_rows: pd.DataFrame) -> Path:
    """Write fake ID-based ground truth in the expected TSV contract."""
    ground_truth = pd.DataFrame(
        [
            {"source1_entity_id": "S1-001", "matched_entity_ids": "S2-101|S3-201"},
            {"source1_entity_id": "S1-002", "matched_entity_ids": "S2-102|S3-202"},
            {"source1_entity_id": "S1-003", "matched_entity_ids": "S2-103|S3-203"},
            {"source1_entity_id": "S1-004", "matched_entity_ids": "S3-204"},
            {"source1_entity_id": "S1-005", "matched_entity_ids": "S2-104|S3-205"},
            {"source1_entity_id": "S1-006", "matched_entity_ids": ""},
        ]
    )
    path = tmp_path / "ground_truth.tsv"
    ground_truth.to_csv(path, sep="\t", index=False)
    return path
