"""Name normalization stage for business entity resolution."""

from typing import Any
import re
import unicodedata
from pathlib import Path

import pandas as pd

from src.utils.config_loader import CONFIG, get_config_value

NAME_COLUMN = get_config_value(CONFIG, "schema", "name_column")


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
    """Convert a value to normalized-safe text."""
    if _is_missing_like(value):
        return ""
    return str(value).strip()


def _basic_normalize(value: Any) -> str:
    """
    Lowercase and normalize Unicode, punctuation, and whitespace.

    Original values are never modified.
    """
    text = _to_text(value)

    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()

    # Convert punctuation/symbols into spaces rather than deleting them.
    # This prevents accidental token concatenation.
    text = "".join(
        " " if unicodedata.category(ch).startswith(("P", "S")) else ch
        for ch in text
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text


_LEGAL_SUFFIX_CACHE: dict[str, set[str]] = {}


def _load_legal_suffixes(suffix_path: str) -> set[str]:
    if suffix_path in _LEGAL_SUFFIX_CACHE:
        return _LEGAL_SUFFIX_CACHE[suffix_path]
    try:
        import yaml
        path = Path(suffix_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        suffixes: set[str] = set()
        for values in data.get('suffix_groups', {}).values():
            if isinstance(values, list):
                suffixes.update(str(value).casefold().strip() for value in values if str(value).strip())
        _LEGAL_SUFFIX_CACHE[suffix_path] = suffixes
        return suffixes
    except (OSError, ImportError, TypeError, AttributeError):
        _LEGAL_SUFFIX_CACHE[suffix_path] = set()
        return set()


def _remove_legal_suffix(tokens: list[str], config: dict[str, Any]) -> list[str]:
    normalization = config.get('normalization', {})
    suffix_path = normalization.get('legal_suffix_dictionary')
    if not suffix_path:
        return tokens
    suffixes = _load_legal_suffixes(str(suffix_path))
    if tokens and tokens[-1] in suffixes:
        return tokens[:-1]
    return tokens


def _token_normalize(value: Any, config: dict[str, Any]) -> str:
    """Create a canonical token representation."""
    basic = _basic_normalize(value)

    if not basic:
        return ""

    tokens = basic.split()

    # Remove configured legal suffixes only from the end.
    tokens = _remove_legal_suffix(tokens, config)

    # Remove duplicates while preserving the original token order.
    ordered_tokens: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token and token not in seen:
            seen.add(token)
            ordered_tokens.append(token)

    return " ".join(ordered_tokens)


def _alnum_normalize(value: Any, config: dict[str, Any]) -> str:
    """Create a compact alphanumeric matching representation."""
    token_norm = _token_normalize(value, config)

    return re.sub(r"[^0-9a-z]+", "", token_norm)


DEVANAGARI_MAP: dict[str, str] = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ii", "उ": "u", "ऊ": "uu",
    "ऋ": "r", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh",
    "ष": "sh", "स": "s", "ह": "h", "ा": "a", "ि": "i",
    "ी": "i", "ु": "u", "ू": "u", "े": "e", "ै": "ai",
    "ो": "o", "ौ": "au", "ं": "n", "ः": "h", "्": "",
    "़": "", "ॅ": "e", "ॉ": "o", "ॆ": "e", "ॊ": "o",
}


def _transliterate(value: Any) -> str:
    """
    Produce a lowercase Latin-oriented representation.

    Uses a lightweight script-aware transliteration for Devanagari text when
    a dedicated transliteration library is unavailable. The model is only used
    for generic text representation; no external business identity lookup is
    performed.
    """
    text = _basic_normalize(value)

    if not text:
        return ""

    if any("\u0900" <= ch <= "\u097F" for ch in text):
        romanized_chars: list[str] = []
        for char in text:
            romanized_chars.append(DEVANAGARI_MAP.get(char, char))
        romanized = "".join(romanized_chars)
        return re.sub(r"\s+", " ", romanized.casefold()).strip()

    try:
        from text_unidecode import unidecode

        return re.sub(r"\s+", " ", unidecode(text).casefold()).strip()
    except ImportError:
        decomposed = unicodedata.normalize("NFKD", text)

        ascii_text = decomposed.encode(
            "ascii", errors="ignore"
        ).decode("ascii")

        return re.sub(r"\s+", " ", ascii_text.casefold()).strip()


def normalize_names(
    frame: pd.DataFrame,
    config: dict[str, Any] = CONFIG,
) -> pd.DataFrame:
    """
    Return name-normalized records while retaining original IDs and names.

    The input DataFrame is never modified in-place.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")

    if NAME_COLUMN not in frame.columns:
        raise ValueError(
            f"Required name column '{NAME_COLUMN}' is missing"
        )

    result = frame.copy()

    # Always preserve the original source value.
    result["name_original"] = result[NAME_COLUMN]

    result["name_basic_norm"] = result[NAME_COLUMN].map(
        _basic_normalize
    )

    result["name_token_norm"] = result[NAME_COLUMN].map(
        lambda value: _token_normalize(value, config)
    )

    # Set representation for blocking.
    # Uses the same suffix-aware normalization as name_token_norm.
    result["name_tokens"] = result[NAME_COLUMN].map(
        lambda value: list(
            dict.fromkeys(
                _token_normalize(value, config).split()
            )
        )
        if _token_normalize(value, config)
        else []
    )

    result["name_alnum_norm"] = result[NAME_COLUMN].map(
        lambda value: _alnum_normalize(value, config)
    )

    if (
        config.get("normalization", {})
        .get("transliteration_enabled", True)
    ):
        result["name_transliterated"] = result[NAME_COLUMN].map(
            _transliterate
        )
    else:
        result["name_transliterated"] = result["name_basic_norm"]

    return result
