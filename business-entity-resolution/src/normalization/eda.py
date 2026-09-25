"""Streaming EDA for business entity resolution normalization."""

from pathlib import Path
from typing import Any
import re
import math

import pandas as pd

from src.utils.config_loader import CONFIG, get_config_value


PATHS = get_config_value(CONFIG, "paths")
SCHEMA = get_config_value(CONFIG, "schema")
EDA_CONFIG = get_config_value(CONFIG, "eda")

ID_COLUMN = get_config_value(SCHEMA, "id_column")
NAME_COLUMN = get_config_value(SCHEMA, "name_column")
ADDRESS_COLUMN = get_config_value(SCHEMA, "address_column")
COUNTRY_COLUMN = get_config_value(SCHEMA, "country_column")

CHUNK_SIZE = 50_000
MISSING_VALUES = {"", "na", "n/a", "null", "none", "nan"}
NON_LATIN_PATTERN = re.compile(r"[^\x00-\x7F]")


def _resolve_data_path(configured_path: str, split: str) -> str:
    """Resolve raw-data path while supporting train/test subdirectories."""
    configured = Path(configured_path)

    if configured.exists():
        return str(configured)

    fallback = configured.parent / split / configured.name

    if fallback.exists():
        return str(fallback)

    raise FileNotFoundError(
        f"Raw data file not found. Tried:\n"
        f"  {configured}\n"
        f"  {fallback}"
    )


def _read_chunks(path: str):
    """Read a large TSV incrementally."""
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
        chunksize=CHUNK_SIZE,
    )


def _is_missing_series(series: pd.Series) -> pd.Series:
    """Vectorized missing-value detection."""
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(MISSING_VALUES)
    )


def _clean_text_series(series: pd.Series) -> pd.Series:
    """Temporary case-folded representation for analysis only."""
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
    )


def _percentile(sorted_values: list[int], q: float) -> float:
    """Linear-interpolation percentile."""
    if not sorted_values:
        return 0.0

    n = len(sorted_values)

    if n == 1:
        return float(sorted_values[0])

    position = (n - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))

    if lower == upper:
        return float(sorted_values[lower])

    fraction = position - lower
    return (
        sorted_values[lower]
        + fraction * (sorted_values[upper] - sorted_values[lower])
    )


def _length_stats(lengths: list[int]) -> dict[str, float]:
    if not lengths:
        return {
            "min": 0,
            "median": 0.0,
            "mean": 0.0,
            "p95": 0.0,
            "max": 0,
        }

    values = sorted(lengths)

    return {
        "min": int(values[0]),
        "median": _percentile(values, 0.50),
        "mean": float(sum(values) / len(values)),
        "p95": _percentile(values, 0.95),
        "max": int(values[-1]),
    }


def _new_field_state() -> dict[str, Any]:
    return {
        "count": 0,
        "missing": 0,
        "lengths": [],
        "seen": set(),
        "duplicate_records": 0,
    }


def _process_source(path: str) -> dict[str, Any]:
    """Process one large source file incrementally."""
    state: dict[str, Any] = {
        "rows": 0,
        "columns": [],
        "fields": {},
        "countries": {},
        "non_latin_count": 0,
    }

    print(f"Processing: {path}")

    for chunk_number, chunk in enumerate(_read_chunks(path), start=1):
        if not state["columns"]:
            state["columns"] = list(chunk.columns)

            for column in chunk.columns:
                state["fields"][column] = _new_field_state()

        state["rows"] += len(chunk)

        for column in chunk.columns:
            field = state["fields"][column]
            series = chunk[column]

            missing_mask = _is_missing_series(series)
            field["count"] += len(series)
            field["missing"] += int(missing_mask.sum())

            if column in (NAME_COLUMN, ADDRESS_COLUMN):
                valid = series.loc[~missing_mask].astype(str).str.strip()

                if len(valid):
                    field["lengths"].extend(
                        valid.str.len().astype(int).tolist()
                    )

                    normalized = (
                        valid.str.casefold()
                        .str.strip()
                    )

                    counts = normalized.value_counts()

                    field["duplicate_records"] += int(
                        counts[counts > 1].sum()
                    )

                    field["seen"].update(normalized.tolist())

        if COUNTRY_COLUMN in chunk.columns:
            countries = (
                chunk[COUNTRY_COLUMN]
                .fillna("")
                .astype(str)
                .str.strip()
            )

            countries = countries.where(
                ~_is_missing_series(countries),
                "MISSING",
            )

            counts = countries.value_counts()

            for country, count in counts.items():
                state["countries"][str(country)] = (
                    state["countries"].get(str(country), 0)
                    + int(count)
                )

        text_columns = [
            column
            for column in (NAME_COLUMN, ADDRESS_COLUMN)
            if column in chunk.columns
        ]

        if text_columns:
            combined = chunk[text_columns[0]].fillna("").astype(str)

            for column in text_columns[1:]:
                combined = combined.str.cat(
                    chunk[column].fillna("").astype(str),
                    sep=" ",
                )

            state["non_latin_count"] += int(
                combined.str.contains(
                    NON_LATIN_PATTERN,
                    regex=True,
                    na=False,
                ).sum()
            )

        if chunk_number % 10 == 0:
            print(
                f"  processed {state['rows']:,} rows..."
            )

    for column, field in state["fields"].items():
        total = field["count"]

        field["missing_rate"] = (
            field["missing"] / total
            if total
            else 0.0
        )

        if column in (NAME_COLUMN, ADDRESS_COLUMN):
            field["length_stats"] = _length_stats(field["lengths"])

            field["duplicate_rate"] = (
                field["duplicate_records"]
                / total
                if total
                else 0.0
            )

        field.pop("lengths", None)
        field.pop("seen", None)

    state["non_latin_rate"] = (
        state["non_latin_count"] / state["rows"]
        if state["rows"]
        else 0.0
    )

    state.pop("non_latin_count", None)

    return state


def _process_ground_truth(path: str) -> tuple[float, int, int]:
    """Calculate singleton rate without loading ground truth fully."""
    total = 0
    singleton_count = 0

    print(f"Processing ground truth: {path}")

    for chunk in _read_chunks(path):
        if "matched_entity_ids" not in chunk.columns:
            total += len(chunk)
            continue

        values = chunk["matched_entity_ids"].fillna("").astype(str)

        counts = values.map(
            lambda value: len(
                [
                    item
                    for item in value.split(",")
                    if item.strip()
                ]
            )
        )

        singleton_count += int((counts == 1).sum())
        total += len(chunk)

    rate = singleton_count / total if total else 0.0

    return rate, singleton_count, total


def _fmt_rate(value: float) -> str:
    return f"{value * 100:.2f}%"


def _build_report(
    profile: dict[str, Any],
    singleton_rate: float,
    singleton_count: int,
    singleton_total: int,
) -> str:
    lines: list[str] = []

    lines.append("# Normalization EDA Report")
    lines.append("")
    lines.append(
        "Streaming EDA over the raw train/test source files. "
        f"Chunk size: {CHUNK_SIZE:,} rows."
    )
    lines.append("")
    lines.append(
        "**Raw data was not modified. Case-folding and other temporary "
        "representations are used only for analysis.**"
    )
    lines.append("")

    lines.append("## 1. Source overview")
    lines.append("")
    lines.append(
        "| Source | Rows | Columns | Estimated non-Latin rate |"
    )
    lines.append("|---|---:|---:|---:|")

    for source, values in profile["sources"].items():
        lines.append(
            f"| `{source}` | {values['rows']:,} | "
            f"{len(values['columns'])} | "
            f"{_fmt_rate(values['non_latin_rate'])} |"
        )

    lines.append("")

    lines.append("## 2. Missing/null rate")
    lines.append("")
    lines.append(
        "| Source | Column | Missing records | Missing rate |"
    )
    lines.append("|---|---|---:|---:|")

    for source, values in profile["sources"].items():
        for column, field in values["fields"].items():
            if "missing_rate" not in field:
                continue

            missing_count = field["missing"]

            lines.append(
                f"| `{source}` | `{column}` | "
                f"{missing_count:,} | "
                f"{_fmt_rate(field['missing_rate'])} |"
            )

    lines.append("")

    lines.append("## 3. Country distribution")
    lines.append("")
    lines.append("### Train")
    lines.append("")

    for source, countries in profile["country_distribution"]["train"].items():
        lines.append(f"**{source}**")
        lines.append("")
        lines.append("| Country | Count |")
        lines.append("|---|---:|")

        for country, count in sorted(
            countries.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            lines.append(f"| `{country}` | {count:,} |")

        lines.append("")

    lines.append("### Test")
    lines.append("")

    for source, countries in profile["country_distribution"]["test"].items():
        lines.append(f"**{source}**")
        lines.append("")
        lines.append("| Country | Count |")
        lines.append("|---|---:|")

        for country, count in sorted(
            countries.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            lines.append(f"| `{country}` | {count:,} |")

        lines.append("")

    lines.append("### Country shift")
    lines.append("")

    if profile["country_shift"]:
        lines.append(
            "Countries present in test but not training: "
            + ", ".join(
                f"`{country}`"
                for country in profile["country_shift"]
            )
        )
    else:
        lines.append("No test-only countries detected.")

    lines.append("")
    lines.append(
        "Country is treated as a soft signal, not a hard filter."
    )
    lines.append("")

    lines.append("## 4. Ground-truth singleton estimate")
    lines.append("")
    lines.append(
        f"- Ground-truth rows: **{singleton_total:,}**"
    )
    lines.append(
        f"- Singleton rows: **{singleton_count:,}**"
    )
    lines.append(
        f"- Estimated singleton rate: "
        f"**{_fmt_rate(singleton_rate)}**"
    )
    lines.append("")

    lines.append("## 5. Name/address length distributions")
    lines.append("")
    lines.append(
        "| Source | Field | Min | Median | Mean | P95 | Max |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")

    for source, values in profile["sources"].items():
        for field_name in (NAME_COLUMN, ADDRESS_COLUMN):
            if field_name not in values["fields"]:
                continue

            stats = values["fields"][field_name].get(
                "length_stats",
                {},
            )

            lines.append(
                f"| `{source}` | `{field_name}` | "
                f"{stats.get('min', 0)} | "
                f"{stats.get('median', 0):.1f} | "
                f"{stats.get('mean', 0):.1f} | "
                f"{stats.get('p95', 0):.1f} | "
                f"{stats.get('max', 0)} |"
            )

    lines.append("")

    lines.append("## 6. Within-source duplicate detection")
    lines.append("")
    lines.append(
        "| Source | Field | Duplicate-record rate |"
    )
    lines.append("|---|---|---:|")

    for source, values in profile["sources"].items():
        for field_name in (NAME_COLUMN, ADDRESS_COLUMN):
            if field_name not in values["fields"]:
                continue

            rate = values["fields"][field_name].get(
                "duplicate_rate",
                0.0,
            )

            lines.append(
                f"| `{source}` | `{field_name}` | "
                f"{_fmt_rate(rate)} |"
            )

    lines.append("")

    lines.append("## 7. Transliteration / non-Latin estimate")
    lines.append("")
    lines.append(
        "A record is flagged when its name or address contains at least "
        "one non-ASCII character. This is an approximation, not a "
        "linguistic classifier."
    )
    lines.append("")
    lines.append("| Source | Estimated non-Latin rate |")
    lines.append("|---|---:|")

    for source, values in profile["sources"].items():
        lines.append(
            f"| `{source}` | "
            f"{_fmt_rate(values['non_latin_rate'])} |"
        )

    lines.append("")

    lines.append(
        "## 8. Normalization decisions and information-loss risks"
    )
    lines.append("")
    lines.append(
        "Original name and address values must always remain recoverable."
    )
    lines.append("")
    lines.append(
        "- Case/punctuation/whitespace normalization can collapse "
        "distinct spellings."
    )
    lines.append(
        "- Legal-suffix removal can create collisions between "
        "different businesses."
    )
    lines.append(
        "- Token normalization can remove ordering information."
    )
    lines.append(
        "- Alphanumeric normalization can remove meaningful separators "
        "or symbols."
    )
    lines.append(
        "- Transliteration can map different scripts or spellings to "
        "the same Latin representation."
    )
    lines.append(
        "- Address parsing can produce ambiguous or incomplete "
        "components."
    )
    lines.append(
        "- Missing address components must not generate artificial "
        "tokens."
    )
    lines.append(
        "- Landmark references should be flagged but never resolved "
        "through external identity lookup."
    )
    lines.append("")
    lines.append(
        "Member 2 should check these information-loss points against "
        "blocking recall."
    )
    lines.append("")

    lines.append("## 9. Gate status")
    lines.append("")
    lines.append(
        "EDA completed. Blocking top_k and candidate caps should be "
        "tuned from the distributions in this report rather than "
        "guessed."
    )
    lines.append("")

    return "\n".join(lines)


def run_eda() -> Path:
    """Run streaming EDA over all raw sources."""
    files = {
        "train_source1": _resolve_data_path(
            PATHS["train_source1"], "train"
        ),
        "train_source2": _resolve_data_path(
            PATHS["train_source2"], "train"
        ),
        "train_source3": _resolve_data_path(
            PATHS["train_source3"], "train"
        ),
        "test_source1": _resolve_data_path(
            PATHS["test_source1"], "test"
        ),
        "test_source2": _resolve_data_path(
            PATHS["test_source2"], "test"
        ),
        "test_source3": _resolve_data_path(
            PATHS["test_source3"], "test"
        ),
    }

    profile: dict[str, Any] = {
        "sources": {},
        "country_distribution": {
            "train": {},
            "test": {},
        },
        "country_shift": [],
    }

    for source, path in files.items():
        profile["sources"][source] = _process_source(path)

        split = "train" if source.startswith("train_") else "test"

        profile["country_distribution"][split][source] = (
            profile["sources"][source]["countries"]
        )

    train_countries = set()

    for countries in profile["country_distribution"]["train"].values():
        train_countries.update(countries.keys())

    test_countries = set()

    for countries in profile["country_distribution"]["test"].values():
        test_countries.update(countries.keys())

    profile["country_shift"] = sorted(
        test_countries - train_countries
    )

    ground_truth_path = _resolve_data_path(
        PATHS["train_ground_truth"],
        "train",
    )

    singleton_rate, singleton_count, singleton_total = (
        _process_ground_truth(ground_truth_path)
    )

    report = _build_report(
        profile,
        singleton_rate,
        singleton_count,
        singleton_total,
    )

    report_path = Path(EDA_CONFIG["report_out"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    return report_path


if __name__ == "__main__":
    output = run_eda()
    print(f"EDA report written to: {output}")
