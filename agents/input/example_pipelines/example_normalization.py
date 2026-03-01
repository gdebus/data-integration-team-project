import os
import re
import sys
from pathlib import Path

import pandas as pd

from PyDI.io import load_xml
from PyDI.normalization import create_normalization_config, normalize_country, normalize_dataset

try:
    from list_normalization import detect_list_like_columns, normalize_list_like_columns
except ModuleNotFoundError:
    module_name = "list_normalization.py"
    for candidate in (Path.cwd(), Path.cwd() / "agents"):
        if (candidate / module_name).is_file():
            candidate_str = str(candidate.resolve())
            if candidate_str not in sys.path:
                sys.path.append(candidate_str)
    from list_normalization import detect_list_like_columns, normalize_list_like_columns


def load_dataset(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".xml":
        return load_xml(path, name=os.path.basename(path), nested_handling="aggregate")
    raise ValueError(f"Unsupported dataset format: {path}")


def infer_id_column(df: pd.DataFrame) -> str | None:
    for candidate in ("id", "_id"):
        if candidate in df.columns:
            return candidate
    for col in df.columns:
        if "id" in str(col).lower():
            return col
    return None


# Example inputs (replace in generated code)
datasets = [
    "input/datasets/games/dbpedia.xml",
    "input/datasets/games/metacritic.xml",
    "input/datasets/games/sales.xml",
]
validation_set = "input/datasets/games/testsets/validation_set_fusion.xml"

validation_df = load_dataset(validation_set)
validation_columns = set(validation_df.columns.tolist()) if validation_df is not None else set()


def infer_lowercase_columns(df: pd.DataFrame) -> set[str]:
    out: set[str] = set()
    if df is None or df.empty:
        return out
    for col in df.columns:
        series = df[col]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue
        alpha_values = []
        for value in series.dropna().astype(str).head(500).tolist():
            text = value.strip()
            if text and re.search(r"[A-Za-z]", text):
                alpha_values.append(text)
        if len(alpha_values) < 8:
            continue
        lower_ratio = sum(1 for v in alpha_values if v == v.lower()) / len(alpha_values)
        if lower_ratio >= 0.85:
            out.add(str(col).lower())
    return out


def infer_country_output_format(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "name"
    cols = [c for c in df.columns if "country" in str(c).lower()]
    values = []
    for col in cols:
        values.extend([str(v).strip() for v in df[col].dropna().astype(str).head(500).tolist() if str(v).strip()])
    if not values:
        return "name"
    alpha2 = sum(1 for v in values if re.fullmatch(r"[A-Z]{2}", v))
    alpha3 = sum(1 for v in values if re.fullmatch(r"[A-Z]{3}", v))
    numeric = sum(1 for v in values if re.fullmatch(r"\d{3}", v))
    official = sum(1 for v in values if (" of " in v.lower() and len(v) > 18) or " and " in v.lower())
    total = max(len(values), 1)
    if alpha2 / total >= 0.70:
        return "alpha_2"
    if alpha3 / total >= 0.70:
        return "alpha_3"
    if numeric / total >= 0.70:
        return "numeric"
    if official / total >= 0.55:
        return "official_name"
    return "name"


validation_lowercase_columns = infer_lowercase_columns(validation_df)
country_output_format = infer_country_output_format(validation_df)
validation_list_like_columns = set(
    str(c).lower()
    for c in detect_list_like_columns([validation_df], exclude_columns={"id", "_id"})
) if validation_df is not None and not validation_df.empty else set()

os.makedirs("output/normalization", exist_ok=True)

normalized_paths: list[str] = []
for dataset_path in datasets:
    df = load_dataset(dataset_path)
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    id_col = infer_id_column(df)

    # Restrict explicit transforms to validation-relevant columns to avoid over-normalizing.
    transforms = {}
    for col in df.columns:
        if id_col is not None and col == id_col:
            continue
        if validation_columns and col not in validation_columns:
            continue
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            ops = ["strip", "normalize_whitespace"]
            if str(col).lower() in validation_lowercase_columns:
                ops.append("lower")
            transforms[col] = ops

    config = create_normalization_config(
        enable_unit_conversion=True,
        enable_quantity_scaling=True,
        normalize_text=False,
        lowercase_text=False,
        remove_extra_whitespace=True,
        standardize_nulls=True,
        add_metadata_columns=False,
        column_transformations=transforms,
        missing_transform_column_policy="warn",
    )

    normalized_df, result = normalize_dataset(df, config=config)

    # Preserve identifier values exactly; normalization must never mutate IDs.
    if id_col and id_col in df.columns and id_col in normalized_df.columns:
        normalized_df[id_col] = df[id_col].copy()

    # Country canonicalization for fields with likely country semantics.
    def _normalize_country_safe(value):
        if pd.isna(value):
            return value
        text = str(value).strip()
        if not text:
            return value
        normalized = normalize_country(text, output_format=country_output_format)
        return normalized if normalized else text

    country_columns = [c for c in normalized_df.columns if "country" in str(c).lower()]
    for col in country_columns:
        if validation_columns and col not in validation_columns:
            continue
        normalized_df[col] = normalized_df[col].apply(_normalize_country_safe)

    # Normalize list-like encoding inconsistencies (JSON-like strings, nested lists, etc.).
    list_cols = set(detect_list_like_columns([normalized_df], exclude_columns={id_col.lower()} if id_col else set()))
    if validation_list_like_columns:
        for col in normalized_df.columns:
            if str(col).lower() in validation_list_like_columns:
                list_cols.add(str(col))
    list_cols = sorted(list(list_cols))
    if list_cols:
        normalize_list_like_columns([normalized_df], list_cols)

    # Safety checks
    if len(normalized_df) != len(df):
        raise RuntimeError(f"Row count changed for {dataset_name}")
    if id_col and id_col not in normalized_df.columns:
        raise RuntimeError(f"ID column missing after normalization in {dataset_name}: {id_col}")

    out_path = f"output/normalization/{dataset_name}.csv"
    normalized_df.to_csv(out_path, index=False)
    normalized_paths.append(out_path)

    print(f"[NORMALIZATION] {dataset_name}: {result.get_summary()}")
    if country_columns:
        print(f"[NORMALIZATION] {dataset_name}: country columns -> {country_columns} (format={country_output_format})")
    if list_cols:
        print(f"[NORMALIZATION] {dataset_name}: list-like columns -> {list_cols}")

print("[NORMALIZATION] normalized datasets:")
for path in normalized_paths:
    print(f"  - {path}")
