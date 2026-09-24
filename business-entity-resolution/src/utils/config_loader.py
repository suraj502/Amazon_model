"""Load and validate the repository's YAML configuration."""

from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.yaml"
_REQUIRED_SECTIONS = {"paths", "schema", "normalization", "blocking", "features", "model", "evaluation", "threshold", "submission", "runtime"}


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load config.yaml and validate its required top-level sections."""
    path = config_path or _CONFIG_PATH
    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict) or not _REQUIRED_SECTIONS.issubset(config):
        raise ValueError("Configuration is missing one or more required sections")
    return config


def get_config_value(config: dict[str, Any], *keys: str) -> Any:
    """Return a nested configuration value addressed by successive keys."""
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise KeyError("Missing configuration key: " + ".".join(keys))
        value = value[key]
    return value


CONFIG = load_config()
