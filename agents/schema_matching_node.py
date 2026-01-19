import os
from typing import Dict, List, Any, Optional

import pandas as pd

from PyDI.io import load_xml, load_parquet, load_csv
from PyDI.schemamatching import LLMBasedSchemaMatcher, LabelBasedSchemaMatcher


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return load_parquet(path)
    if ext == ".csv":
        return load_csv(path)
    if ext == ".xml":
        return load_xml(path, nested_handling="aggregate")
    raise ValueError(f"Unsupported format: {ext}. Supported: .csv, .parquet, .xml")


def _dataset_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _write_dataset(df: pd.DataFrame, source_path: str, out_dir: str) -> str:
    name = _dataset_name(source_path)
    ext = os.path.splitext(source_path)[1].lower()
    if ext == ".parquet":
        out_path = os.path.join(out_dir, f"{name}.parquet")
        df.to_parquet(out_path, index=False)
        return out_path
    out_path = os.path.join(out_dir, f"{name}.csv")
    df.to_csv(out_path, index=False)
    return out_path


def run_schema_matching(
    dataset_paths: List[str],
    model: Optional[Any] = None,
    output_dir: str = "output/schema-matching",
    num_rows: int = 10,
    debug: bool = False,
) -> Dict[str, Any]:
    if not dataset_paths:
        return {"datasets": [], "schema_correspondences": {}}

    os.makedirs(output_dir, exist_ok=True)

    datasets = [load_dataset(path) for path in dataset_paths]
    ref_path = dataset_paths[0]
    ref_name = _dataset_name(ref_path)
    ref_df = datasets[0]

    if model is not None:
        matcher = LLMBasedSchemaMatcher(chat_model=model, num_rows=num_rows, debug=debug)
    else:
        matcher = LabelBasedSchemaMatcher()

    correspondences: Dict[str, Any] = {}
    aligned_paths: List[str] = []
    aligned_columns: Dict[str, List[str]] = {}

    aligned_paths.append(_write_dataset(ref_df, ref_path, output_dir))
    aligned_columns[ref_name] = list(ref_df.columns)

    for path, df in zip(dataset_paths[1:], datasets[1:]):
        target_name = _dataset_name(path)
        key = f"{ref_name}_{target_name}"

        corr = matcher.match(ref_df, df)
        if hasattr(corr, "to_csv"):
            corr.to_csv(os.path.join(output_dir, f"{key}_schema_correspondences.csv"), index=False)
        if hasattr(corr, "to_dict"):
            correspondences[key] = corr.to_dict(orient="records")
        else:
            correspondences[key] = [{"raw": str(corr)}]

        rename_map = {}
        if hasattr(corr, "columns") and "target_column" in corr.columns and "source_column" in corr.columns:
            rename_map = corr.set_index("target_column")["source_column"].to_dict()
        df_aligned = df.rename(columns=rename_map) if rename_map else df.copy()

        aligned_paths.append(_write_dataset(df_aligned, path, output_dir))
        aligned_columns[target_name] = list(df_aligned.columns)

    print("[*] Schema alignment columns:")
    for name, cols in aligned_columns.items():
        print(f"    {name}: {cols}")

    return {"datasets": aligned_paths, "schema_correspondences": correspondences}
