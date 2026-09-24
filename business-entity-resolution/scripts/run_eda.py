"""Run exploratory data analysis before any downstream pipeline stage."""

from typing import Any

from src.utils.config_loader import CONFIG, get_config_value

EDA_REPORT_PATH = get_config_value(CONFIG, "eda", "report_out")


def main() -> None:
    """Profile configured raw sources and write the configured EDA report."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
