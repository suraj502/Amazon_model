"""Run name and address normalization for configured source tables."""

from pathlib import Path

import pandas as pd

from src.normalization.address_normalize import normalize_addresses
from src.normalization.name_normalize import normalize_names
from src.utils.config_loader import CONFIG, get_config_value


ROOT = Path(__file__).resolve().parents[1]


def _normalize(input_path: Path, output_path: Path) -> None:
    frame = pd.read_csv(input_path, sep="\t", dtype={"entity_id": "string"})
    normalized = normalize_addresses(normalize_names(frame, CONFIG), CONFIG)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_parquet(output_path, index=False)


def main() -> None:
    """Load configured inputs and normalize them by entity ID."""
    paths = get_config_value(CONFIG, "paths")
    for source in ("source1", "source2", "source3"):
        input_path = ROOT / paths[f"train_{source}"]
        output_path = ROOT / paths["processed_dir"] / f"train_{source}_normalized.parquet"
        _normalize(input_path, output_path)


if __name__ == "__main__":
    main()
