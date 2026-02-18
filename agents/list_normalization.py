import ast
import json
from typing import Any, Iterable, List, Optional, Sequence, Set

import pandas as pd


_NULL_TOKENS = {"", "nan", "none", "null"}


def _try_parse_sequence(text: str) -> Optional[Sequence[Any]]:
    stripped = text.strip()
    if not stripped or stripped[0] not in "[({":
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(stripped)
        except Exception:
            continue
        if isinstance(parsed, (list, tuple, set)):
            return parsed
    return None


def _flatten_list_tokens(value: Any, depth: int = 0, max_depth: int = 6) -> List[str]:
    if depth > max_depth:
        return [str(value)]

    if value is None:
        return []

    if isinstance(value, float) and pd.isna(value):
        return []

    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for item in value:
            out.extend(_flatten_list_tokens(item, depth=depth + 1, max_depth=max_depth))
        return out

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        parsed = _try_parse_sequence(text)
        if parsed is not None:
            return _flatten_list_tokens(parsed, depth=depth + 1, max_depth=max_depth)
        return [text]

    return [str(value)]


def normalize_list_value(value: Any, dedupe: bool = True) -> List[str]:
    tokens = _flatten_list_tokens(value)
    cleaned: List[str] = []
    seen: Set[str] = set()
    for token in tokens:
        normalized = token.strip()
        if normalized.lower() in _NULL_TOKENS:
            continue
        if dedupe:
            if normalized in seen:
                continue
            seen.add(normalized)
        cleaned.append(normalized)
    return cleaned


def is_list_like_value(value: Any) -> bool:
    if isinstance(value, (list, tuple, set)):
        return True
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return False
        return _try_parse_sequence(text) is not None
    return False


def detect_list_like_columns(
    dataframes: Iterable[pd.DataFrame],
    exclude_columns: Optional[Set[str]] = None,
    sample_size: int = 200,
    min_hits: int = 3,
    min_ratio: float = 0.35,
) -> List[str]:
    exclude = {c.lower() for c in (exclude_columns or set())}
    hit_counts: dict[str, int] = {}
    sample_counts: dict[str, int] = {}

    for df in dataframes:
        for column in df.columns:
            if column.lower() in exclude:
                continue
            sample = df[column].dropna().head(sample_size)
            total = len(sample)
            if total == 0:
                continue
            hits = sum(1 for value in sample if is_list_like_value(value))
            hit_counts[column] = hit_counts.get(column, 0) + hits
            sample_counts[column] = sample_counts.get(column, 0) + total

    detected = []
    for column, total in sample_counts.items():
        hits = hit_counts.get(column, 0)
        ratio = hits / total if total else 0.0
        if hits >= min_hits and ratio >= min_ratio:
            detected.append(column)
    return sorted(detected)


def normalize_dataframe_list_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            continue
        df[column] = df[column].apply(normalize_list_value)
    return df


def normalize_list_like_columns(
    dataframes: List[pd.DataFrame],
    columns: Iterable[str],
) -> List[pd.DataFrame]:
    for df in dataframes:
        normalize_dataframe_list_columns(df, columns)
    return dataframes
