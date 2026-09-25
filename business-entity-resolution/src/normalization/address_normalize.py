"""Address normalization stage for multi-country business records."""

from typing import Any
import re
import unicodedata

import pandas as pd
import yaml

from src.utils.config_loader import CONFIG, get_config_value

ADDRESS_COLUMN = get_config_value(CONFIG, "schema", "address_column")

_ABBREVIATION_CACHE: dict[str, dict[str, str]] = {}


def _is_missing_like(value: Any) -> bool:
    """Treat common null-like placeholders as missing."""
    if value is None or pd.isna(value):
        return True

    text = str(value).strip()
    if not text:
        return True

    normalized = unicodedata.normalize("NFKC", text).casefold()
    compact = re.sub(r"[^a-z0-9]+", "", normalized)

    return compact in {"na", "null", "none", "nan"}


def _to_text(value: Any) -> str:
    """Convert a value to safe text while preserving missing values as empty."""
    if _is_missing_like(value):
        return ""
    return str(value).strip()


def _basic_normalize(value: Any) -> str:
    """Lowercase and normalize Unicode, punctuation, and whitespace."""
    text = _to_text(value)

    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()

    text = "".join(
        " " if unicodedata.category(ch).startswith(("P", "S")) else ch
        for ch in text
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text


def _component_text(value: Any) -> str:
    """
    Normalize an address while preserving commas as structural separators.
    """
    text = _to_text(value)

    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()

    text = "".join(
        ch
        if ch == ","
        else (
            " "
            if unicodedata.category(ch).startswith(("P", "S"))
            else ch
        )
        for ch in text
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text


def _load_abbreviations(config: dict[str, Any]) -> dict[str, str]:
    """Load configured address abbreviation -> canonical form mappings."""
    path = (
        config.get("normalization", {})
        .get("address_abbreviation_dictionary")
    )

    if not path:
        return {}

    cache_key = str(path)

    if cache_key in _ABBREVIATION_CACHE:
        return _ABBREVIATION_CACHE[cache_key]

    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError):
        return {}

    abbreviations: dict[str, str] = {}

    for canonical, values in data.get("abbreviations", {}).items():
        canonical_value = str(canonical).casefold().strip()

        for value in values or []:
            abbreviation = str(value).casefold().strip()

            if abbreviation:
                abbreviations[abbreviation] = canonical_value

    _ABBREVIATION_CACHE[cache_key] = abbreviations

    return abbreviations


def _apply_abbreviations(
    text: str,
    config: dict[str, Any],
) -> str:
    """Replace configured address abbreviations token-by-token."""
    if not text:
        return ""

    abbreviations = _load_abbreviations(config)

    if not abbreviations:
        return text

    tokens = text.split()

    tokens = [
        abbreviations.get(token, token)
        for token in tokens
    ]

    return " ".join(tokens)


def _extract_postal_code(text: str) -> str:
    """Extract common postal-code formats conservatively."""
    if not text:
        return ""

    match = re.search(r"\b\d{5}-\d{4}\b", text)

    if match:
        return match.group(0)

    matches = re.findall(r"\b\d{5,6}\b", text)

    if not matches:
        return ""

    return matches[-1]


def _remove_postal_from_segment(segment: str) -> str:
    """Remove postal-code tokens from a parsed address segment."""
    return re.sub(
        r"\b\d{5}(?:-\d{4})?\b|\b\d{6}\b",
        "",
        segment,
    )


def _parse_components(
    value: Any,
    config: dict[str, Any],
) -> str:
    """
    Create a conservative component representation.

    Format:
        street=<...>|city=<...>|state=<...>|postal=<...>

    Empty components are omitted.
    """
    text = _component_text(value)

    if not text:
        return ""

    postal = _extract_postal_code(text)

    parts = [
        part.strip()
        for part in text.split(",")
        if part.strip()
    ]

    if not parts:
        return ""

    parts_without_postal = [
        _remove_postal_from_segment(part).strip()
        for part in parts
    ]

    parts_without_postal = [
        re.sub(r"\s+", " ", part)
        for part in parts_without_postal
        if part
    ]

    parts_without_postal = [
        _apply_abbreviations(part, config)
        for part in parts_without_postal
    ]

    components: list[str] = []

    if len(parts_without_postal) >= 3:
        street = parts_without_postal[0]
        city = parts_without_postal[-2]
        state = parts_without_postal[-1]

        if street:
            components.append(f"street={street}")

        if city:
            components.append(f"city={city}")

        if state:
            components.append(f"state={state}")

    elif len(parts_without_postal) == 2:
        components.append(
            f"street={parts_without_postal[0]}"
        )
        components.append(
            f"state={parts_without_postal[1]}"
        )

    else:
        components.append(
            f"street={parts_without_postal[0]}"
        )

    if postal:
        components.append(f"postal={postal}")

    return "|".join(components)


def _has_landmark_reference(value: Any) -> bool:
    """Flag likely landmark references without resolving them."""
    text = _basic_normalize(value)

    if not text:
        return False

    landmark_patterns = [
        r"\bnear\b",
        r"\bopposite\b",
        r"\bbehind\b",
        r"\bnext to\b",
        r"\bbeside\b",
        r"\bclose to\b",
        r"\bnearby\b",
        r"\blandmark\b",
    ]

    return any(
        re.search(pattern, text)
        for pattern in landmark_patterns
    )


def normalize_addresses(
    frame: pd.DataFrame,
    config: dict[str, Any] = CONFIG,
) -> pd.DataFrame:
    """
    Return address-normalized records while preserving original data.

    The input DataFrame is never modified in-place.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")

    if ADDRESS_COLUMN not in frame.columns:
        raise ValueError(
            f"Required address column '{ADDRESS_COLUMN}' is missing"
        )

    result = frame.copy()

    result["address_original"] = result[ADDRESS_COLUMN]

    result["address_basic_norm"] = result[ADDRESS_COLUMN].map(
        lambda value: _apply_abbreviations(
            _basic_normalize(value),
            config,
        )
    )

    if (
        config.get("normalization", {})
        .get("address_component_parsing", True)
    ):
        result["address_component_parsed"] = result[
            ADDRESS_COLUMN
        ].map(
            lambda value: ""
            if _has_landmark_reference(value)
            else _parse_components(value, config)
        )
    else:
        result["address_component_parsed"] = ""

    result["address_landmark_flag"] = result[
        ADDRESS_COLUMN
    ].map(_has_landmark_reference)

    return result
