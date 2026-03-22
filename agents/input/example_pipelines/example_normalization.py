"""Example normalization pipeline using PyDI NormalizationSpec.

The LLM-driven normalization node generates specs like these automatically by
examining source and target dataset samples.  This example shows the manual
equivalent so generated pipelines have a clear reference.
"""

import os
import sys
from pathlib import Path

import pandas as pd

from PyDI.io import load_xml
from PyDI.normalization import NormalizationSpec, transform_dataframe

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


# --- Configuration (replace in generated code) ---

datasets = [
    "input/datasets/games/dbpedia.xml",
    "input/datasets/games/metacritic.xml",
    "input/datasets/games/sales.xml",
]
validation_set = "input/datasets/games/testsets/validation_set_fusion.xml"

# --- NormalizationSpec per source dataset ---
# The LLM generates these by comparing source data samples with validation
# set samples.  Each spec defines per-column transformations using PyDI's
# ColumnSpec fields.
#
# Available ColumnSpec fields:
#   output_type:            "string" | "float" | "int" | "bool" | "datetime" | "keep"
#   on_failure:             "keep" | "null" | "raise"
#   strip_whitespace:       bool
#   case:                   "lower" | "upper" | "title" | "keep"
#   country_format:         "alpha_2" | "alpha_3" | "numeric" | "name"
#   currency_format:        "alpha_3" | "name" | "keep"
#   phone_format:           "e164" | "international" | "national" | "keep"
#   phone_default_region:   str (e.g. "US")
#   normalize_email:        bool
#   stdnum_format:          bool
#   date_format:            str (e.g. "%Y-%m-%d")
#   expand_scale_modifiers: bool    (e.g. "5 million" → 5000000)
#   convert_percentage:     "to_decimal" | "to_percent" | "keep"
#   target_unit:            str (e.g. "km", "USD")

SPECS = {
    "dbpedia": {
        "name": {"strip_whitespace": True, "case": "lower", "on_failure": "keep"},
        "country": {"country_format": "alpha_2", "strip_whitespace": True, "on_failure": "keep"},
        "genre": {"strip_whitespace": True, "case": "lower", "on_failure": "keep"},
    },
    "metacritic": {
        "name": {"strip_whitespace": True, "case": "lower", "on_failure": "keep"},
        "genre": {"strip_whitespace": True, "case": "lower", "on_failure": "keep"},
    },
    "sales": {
        "name": {"strip_whitespace": True, "case": "lower", "on_failure": "keep"},
        "global_sales": {"output_type": "float", "expand_scale_modifiers": True, "on_failure": "keep"},
    },
}

# Columns that contain list-like values (e.g. '["a", "b"]')
LIST_COLUMNS = ["genre"]

# --- Normalization ---

os.makedirs("output/normalization", exist_ok=True)

normalized_paths: list[str] = []
for dataset_path in datasets:
    df = load_dataset(dataset_path)
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    id_col = infer_id_column(df)

    # Look up spec for this dataset
    spec_dict = SPECS.get(dataset_name, {})

    # Remove ID column from spec (never normalize identifiers)
    if id_col and id_col in spec_dict:
        spec_dict = {k: v for k, v in spec_dict.items() if k != id_col}

    # Remove spec entries for columns not in the DataFrame
    spec_dict = {k: v for k, v in spec_dict.items() if k in df.columns}

    # Save original ID values
    id_backup = df[id_col].copy() if id_col and id_col in df.columns else None

    # Apply NormalizationSpec via PyDI
    normalized_df = df.copy()
    if spec_dict:
        spec = NormalizationSpec.from_dict({"columns": spec_dict})
        result = transform_dataframe(normalized_df, spec)
        normalized_df = result.dataframe

    # Restore ID column
    if id_backup is not None and id_col in normalized_df.columns:
        normalized_df[id_col] = id_backup

    # Apply list normalization
    list_cols = [c for c in LIST_COLUMNS if c in normalized_df.columns]
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
    print(f"[NORMALIZATION] {dataset_name}: spec applied for {list(spec_dict.keys())}")
    if list_cols:
        print(f"[NORMALIZATION] {dataset_name}: list-like columns -> {list_cols}")

print("[NORMALIZATION] normalized datasets:")
for path in normalized_paths:
    print(f"  - {path}")
