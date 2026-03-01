import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd


def parse_correspondence_pair(filename: str, dataset_names: Set[str]) -> Optional[Tuple[str, str]]:
    if not filename.endswith(".csv"):
        return None
    stem = filename[:-4]
    if stem.startswith("correspondences_"):
        stem = stem[len("correspondences_"):]
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    for split_idx in range(1, len(parts)):
        left = "_".join(parts[:split_idx])
        right = "_".join(parts[split_idx:])
        if left == right:
            continue
        if left in dataset_names and right in dataset_names:
            return tuple(sorted((left, right)))
    return None


def collect_latest_correspondence_files(
    state: Dict[str, Any],
    correspondences_dir: str = "output/correspondences/",
) -> List[str]:
    if not os.path.exists(correspondences_dir):
        return []

    dataset_names = {Path(p).stem for p in state.get("datasets", [])}
    latest_by_pair: Dict[Tuple[str, str], Tuple[str, float]] = {}

    for filename in os.listdir(correspondences_dir):
        pair_key = parse_correspondence_pair(filename, dataset_names)
        if not pair_key:
            continue
        file_path = os.path.join(correspondences_dir, filename)
        try:
            modified = os.path.getmtime(file_path)
        except OSError:
            continue
        prev = latest_by_pair.get(pair_key)
        if prev is None or modified > prev[1]:
            latest_by_pair[pair_key] = (file_path, modified)

    selected_files = [value[0] for value in latest_by_pair.values()]

    pipeline_run_started_at = state.get("pipeline_run_started_at")
    if pipeline_run_started_at and selected_files:
        try:
            cutoff = float(pipeline_run_started_at) - 5.0
            fresh_files = [p for p in selected_files if os.path.getmtime(p) >= cutoff]
            if fresh_files:
                return sorted(fresh_files)
            return []
        except Exception:
            pass

    return sorted(selected_files)


def summarize_correspondence_entries(file_paths: List[str]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"files": {}}
    for path in file_paths:
        file_name = os.path.basename(path)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            summary["files"][file_name] = {"error": str(e)}
            continue

        columns = [str(c) for c in df.columns]
        id_like_cols = [c for c in columns if "id" in c.lower()][:6]
        score_col = None
        for candidate in ["score", "similarity", "confidence", "probability", "match_score"]:
            if candidate in df.columns:
                score_col = candidate
                break

        sample_cols = id_like_cols[:] if id_like_cols else columns[:4]
        if score_col and score_col not in sample_cols:
            sample_cols.append(score_col)

        sample_records = []
        try:
            sample_records = df[sample_cols].head(5).to_dict(orient="records")
        except Exception:
            sample_records = df.head(5).to_dict(orient="records")

        boundary_records: Any = []
        if score_col:
            try:
                scored = df[[*sample_cols]].copy()
                scored[score_col] = pd.to_numeric(scored[score_col], errors="coerce")
                scored = scored.dropna(subset=[score_col]).sort_values(score_col)
                if not scored.empty:
                    low_slice = scored.head(3)
                    high_slice = scored.tail(3)
                    boundary_records = {
                        "lowest_scores": low_slice.to_dict(orient="records"),
                        "highest_scores": high_slice.to_dict(orient="records"),
                    }
            except Exception:
                boundary_records = []

        summary["files"][file_name] = {
            "rows": int(len(df)),
            "columns": columns,
            "sample_rows": sample_records,
            "score_column": score_col,
            "score_boundaries": boundary_records,
        }
    return summary
