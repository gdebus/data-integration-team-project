import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

import config


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
    correspondences_dir: str = "",
) -> List[str]:
    if not correspondences_dir:
        correspondences_dir = config.CORRESPONDENCES_DIR
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


def expected_dataset_pairs(state: Dict[str, Any]) -> List[Tuple[str, str]]:
    pairs: set[Tuple[str, str]] = set()
    testsets = state.get("entity_matching_testsets", {}) if isinstance(state, dict) else {}
    if isinstance(testsets, dict) and testsets:
        for key in testsets.keys():
            if isinstance(key, (tuple, list)) and len(key) == 2:
                left = str(key[0]).strip().lower()
                right = str(key[1]).strip().lower()
                if left and right and left != right:
                    pairs.add(tuple(sorted((left, right))))
    if pairs:
        return sorted(list(pairs))

    dataset_names = []
    for path in state.get("datasets", []) if isinstance(state, dict) else []:
        try:
            dataset_names.append(os.path.splitext(os.path.basename(str(path)))[0].strip().lower())
        except Exception:
            continue
    dataset_names = sorted(list({n for n in dataset_names if n}))
    for i in range(len(dataset_names)):
        for j in range(i + 1, len(dataset_names)):
            pairs.add((dataset_names[i], dataset_names[j]))
    return sorted(list(pairs))


def resolve_correspondence_file(left: str, right: str, correspondences_dir: str = "") -> str:
    if not correspondences_dir:
        correspondences_dir = config.CORRESPONDENCES_DIR
    candidates = [
        os.path.join(correspondences_dir, f"correspondences_{left}_{right}.csv"),
        os.path.join(correspondences_dir, f"correspondences_{right}_{left}.csv"),
        os.path.join(correspondences_dir, f"correspondences_{left}__{right}.csv"),
        os.path.join(correspondences_dir, f"correspondences_{right}__{left}.csv"),
        os.path.join(correspondences_dir, f"{left}__{right}.csv"),
        os.path.join(correspondences_dir, f"{right}__{left}.csv"),
    ]
    existing = [path for path in candidates if os.path.exists(path)]
    if not existing:
        return ""
    if len(existing) == 1:
        return existing[0]
    existing.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return existing[0]


def csv_data_row_count(path: str) -> int:
    if not path or not os.path.exists(path):
        return 0
    rows = 0
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                if line.strip():
                    rows += 1
    except Exception:
        return 0
    return rows


def evaluate_correspondence_integrity(state: Dict[str, Any]) -> Dict[str, Any]:
    pair_checks: List[Dict[str, Any]] = []
    invalid_pairs: List[str] = []
    for left, right in expected_dataset_pairs(state):
        pair_key = f"{left}_{right}"
        path = resolve_correspondence_file(left, right)
        if not path:
            pair_checks.append({"pair": pair_key, "status": "missing", "path": "", "data_rows": 0})
            invalid_pairs.append(pair_key)
            continue
        rows = csv_data_row_count(path)
        status = "ok" if rows > 0 else "empty"
        pair_checks.append({"pair": pair_key, "status": status, "path": path, "data_rows": rows})
        if status != "ok":
            invalid_pairs.append(pair_key)

    return {
        "evaluated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "expected_pairs": [f"{l}_{r}" for l, r in expected_dataset_pairs(state)],
        "pair_checks": pair_checks,
        "invalid_pairs": sorted(list(dict.fromkeys(invalid_pairs))),
        "structurally_valid": len(invalid_pairs) == 0,
    }


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
