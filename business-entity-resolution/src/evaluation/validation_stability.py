"""Define multi-seed and public-versus-local stability tracking contracts."""

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config_loader import CONFIG, get_config_value

N_SEEDS = get_config_value(CONFIG, "validation_stability", "n_seeds")


def track_validation_stability(results: pd.DataFrame, config: dict[str, Any] = CONFIG) -> pd.DataFrame:
    """Summarize seed-level F0.5 results and write a variance report.

    ``results`` may contain ``seed``, ``strategy``, ``f05`` and optional
    ``public_f05`` columns. The function deliberately does not mix public
    leaderboard scores into local strategy selection.
    """
    if results.empty:
        summary = pd.DataFrame(columns=["strategy", "mean_f05", "std_f05", "min_f05", "max_f05", "n_seeds"])
    else:
        required = {"strategy", "f05"}
        missing = required - set(results.columns)
        if missing:
            raise ValueError(f"Stability results missing columns: {sorted(missing)}")
        summary = (
            results.groupby("strategy", as_index=False)["f05"]
            .agg(mean_f05="mean", std_f05="std", min_f05="min", max_f05="max", n_seeds="count")
        )
        summary["std_f05"] = summary["std_f05"].fillna(0.0)

    report_path = Path(get_config_value(config, "validation_stability", "log_out"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Validation Stability", "", "| Strategy | Mean F0.5 | Std. dev. | Min | Max | Seeds |", "|---|---:|---:|---:|---:|---:|"]
    lines.extend(
        f"| {row.strategy} | {row.mean_f05:.6f} | {row.std_f05:.6f} | {row.min_f05:.6f} | {row.max_f05:.6f} | {int(row.n_seeds)} |"
        for row in summary.itertuples()
    )
    if "public_f05" in results.columns and "f05" in results.columns:
        gap = results["public_f05"] - results["f05"]
        lines.extend(["", f"Mean public-minus-local F0.5 gap: **{gap.mean():.6f}**.", "The private/local validation score remains the decision criterion; public scores are tracked for drift and overfitting warnings."])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
