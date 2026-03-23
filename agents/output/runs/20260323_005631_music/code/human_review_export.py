import os
import re
import json
import ast
import csv
from datetime import datetime

import pandas as pd


CONTEXT = {
    "datasets": [
        "output/runs/20260323_005631_music/normalization/attempt_1/discogs.csv",
        "output/runs/20260323_005631_music/normalization/attempt_1/lastfm.csv",
        "output/runs/20260323_005631_music/normalization/attempt_1/musicbrainz.csv"
    ],
    "fusion_testset": "input/datasets/music/testsets/test_set.xml",
    "review_attributes": [
        "_id",
        "artist",
        "duration",
        "genre",
        "id",
        "label",
        "name",
        "release-country",
        "release-date",
        "tracks_track_duration",
        "tracks_track_name",
        "tracks_track_position",
        "eval_id"
    ],
    "evaluation_metrics": {
        "overall_accuracy": 0.7857142857142857,
        "macro_accuracy": 0.778726708074534,
        "num_evaluated_records": 23,
        "num_evaluated_attributes": 9,
        "total_evaluations": 154,
        "total_correct": 121,
        "tracks_track_name_accuracy": 0.5217391304347826,
        "tracks_track_name_count": 23,
        "tracks_track_position_accuracy": 0.8695652173913043,
        "tracks_track_position_count": 23,
        "name_accuracy": 0.9565217391304348,
        "name_count": 23,
        "release-country_accuracy": 0.0,
        "release-country_count": 0,
        "duration_accuracy": 0.7391304347826086,
        "duration_count": 23,
        "release-date_accuracy": 0.782608695652174,
        "release-date_count": 23,
        "label_accuracy": 0.625,
        "label_count": 16,
        "tracks_track_duration_accuracy": 0.0,
        "tracks_track_duration_count": 0,
        "artist_accuracy": 0.9565217391304348,
        "artist_count": 23
    },
    "auto_diagnostics": {
        "debug_reason_counts": {
            "mismatch": 46,
            "missing_fused_value": 10
        },
        "debug_reason_ratios": {
            "mismatch": 0.8214285714285714,
            "missing_fused_value": 0.17857142857142858
        },
        "evaluation_stage": "validation",
        "evaluation_testset_path": "input/datasets/music/testsets/test_set.xml",
        "id_alignment": {
            "gold_count": 23,
            "direct_coverage": 7,
            "direct_coverage_ratio": 0.30434782608695654,
            "mapped_coverage": 23,
            "mapped_coverage_ratio": 1.0,
            "missing_gold_ids": []
        },
        "overall_accuracy": 0.7157360406091371
    },
    "integration_diagnostics_report": {},
    "output_paths": {
        "fused_csv": "output/runs/20260323_005631_music/data_fusion/fusion_data.csv",
        "fused_debug_jsonl": "output/runs/20260323_005631_music/data_fusion/debug_fusion_data.jsonl",
        "evaluation_json": "output/runs/20260323_005631_music/pipeline_evaluation/pipeline_evaluation.json",
        "review_summary_json": "output/runs/20260323_005631_music/human_review/human_review_summary.json",
        "review_summary_md": "output/runs/20260323_005631_music/human_review/human_review_summary.md",
        "fused_review_csv": "output/runs/20260323_005631_music/human_review/fused_review_table.csv",
        "source_lineage_csv": "output/runs/20260323_005631_music/human_review/source_lineage_long.csv",
        "diff_csv": "output/runs/20260323_005631_music/human_review/fusion_vs_testset_diff.csv"
    }
}


def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def safe_str(v):
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v)


def normalize_text(v):
    s = safe_str(v).strip()
    return re.sub(r"\s+", " ", s).lower()


def read_csv_flexible(path, warnings):
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    except Exception:
        try:
            return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False, encoding="latin1")
        except Exception as e:
            warnings.append(f"Failed to read CSV {path}: {e}")
            return pd.DataFrame()


def detect_id_columns(df):
    if df is None or df.empty:
        return []
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    preferred = []
    exact_priority = [
        "_id", "id", "eval_id", "fusion_id", "entity_id", "record_id", "source_id"
    ]
    for p in exact_priority:
        if p in lower_map:
            preferred.append(lower_map[p])
    for c in cols:
        cl = c.lower()
        if c not in preferred and (cl.endswith("_id") or cl.startswith("id_") or " id" in cl or cl == "identifier"):
            preferred.append(c)
    for c in cols:
        cl = c.lower()
        if c not in preferred and "id" in cl:
            preferred.append(c)
    return preferred


def choose_primary_id_column(df):
    ids = detect_id_columns(df)
    return ids[0] if ids else (df.columns[0] if len(df.columns) else None)


def parse_fusion_sources(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [safe_str(x).strip() for x in value if safe_str(x).strip()]
    s = safe_str(value).strip()
    if not s:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            obj = parser(s)
            if isinstance(obj, list):
                return [safe_str(x).strip() for x in obj if safe_str(x).strip()]
            if isinstance(obj, dict):
                vals = []
                for k, v in obj.items():
                    if isinstance(v, list):
                        vals.extend([safe_str(x).strip() for x in v if safe_str(x).strip()])
                    else:
                        vals.append(safe_str(v).strip())
                return [x for x in vals if x]
            if isinstance(obj, str):
                return [obj.strip()] if obj.strip() else []
        except Exception:
            pass
    if "|" in s:
        parts = [p.strip() for p in s.split("|")]
    elif ";" in s:
        parts = [p.strip() for p in s.split(";")]
    elif "," in s:
        parts = [p.strip() for p in s.split(",")]
    else:
        parts = [s]
    return [p for p in parts if p]


def infer_dataset_name(path, idx):
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name if name else f"dataset_{idx+1}"


def build_source_indexes(dataset_paths, warnings):
    datasets = []
    global_id_to_rows = {}
    for i, path in enumerate(dataset_paths):
        info = {
            "path": path,
            "name": infer_dataset_name(path, i),
            "df": pd.DataFrame(),
            "id_cols": [],
            "primary_id_col": None,
            "row_index": {}
        }
        if not os.path.exists(path):
            warnings.append(f"Source dataset missing: {path}")
            datasets.append(info)
            continue
        df = read_csv_flexible(path, warnings)
        info["df"] = df
        info["id_cols"] = detect_id_columns(df)
        info["primary_id_col"] = choose_primary_id_column(df)
        row_index = {}
        for ridx, row in df.iterrows():
            for c in info["id_cols"]:
                val = safe_str(row.get(c, "")).strip()
                if val:
                    row_index.setdefault(val, row)
                    global_id_to_rows.setdefault(val, []).append((info["name"], path, row))
        info["row_index"] = row_index
        datasets.append(info)
    return datasets, global_id_to_rows


def find_source_rows(source_ids, datasets, global_id_to_rows):
    rows = []
    seen = set()
    for sid in source_ids:
        sid = safe_str(sid).strip()
        if not sid:
            continue
        if sid in global_id_to_rows:
            for dataset_name, path, row in global_id_to_rows[sid]:
                key = (sid, dataset_name)
                if key not in seen:
                    rows.append({
                        "source_id": sid,
                        "source_dataset": dataset_name,
                        "source_path": path,
                        "row": row
                    })
                    seen.add(key)
            continue
        for ds in datasets:
            if sid in ds["row_index"]:
                key = (sid, ds["name"])
                if key not in seen:
                    rows.append({
                        "source_id": sid,
                        "source_dataset": ds["name"],
                        "source_path": ds["path"],
                        "row": ds["row_index"][sid]
                    })
                    seen.add(key)
    return rows


def read_testset(path, warnings):
    if not path or not os.path.exists(path):
        warnings.append(f"Fusion testset missing or unavailable: {path}")
        return pd.DataFrame()
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return read_csv_flexible(path, warnings)
    if ext in (".json", ".jsonl"):
        try:
            if ext == ".jsonl":
                rows = []
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            rows.append(json.loads(line))
                return pd.DataFrame(rows).fillna("")
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                return pd.DataFrame(obj).fillna("")
            if isinstance(obj, dict):
                for k in ("records", "rows", "data", "items"):
                    if isinstance(obj.get(k), list):
                        return pd.DataFrame(obj[k]).fillna("")
                return pd.DataFrame([obj]).fillna("")
        except Exception as e:
            warnings.append(f"Failed to parse JSON testset {path}: {e}")
            return pd.DataFrame()
    if ext == ".xml":
        try:
            tables = pd.read_xml(path)
            if tables is None:
                return pd.DataFrame()
            if isinstance(tables, list):
                for t in tables:
                    if isinstance(t, pd.DataFrame) and not t.empty:
                        return t.fillna("")
                return pd.DataFrame()
            return tables.fillna("")
        except Exception:
            try:
                from xml.etree import ElementTree as ET
                tree = ET.parse(path)
                root = tree.getroot()
                rows = []
                for parent in root.iter():
                    children = list(parent)
                    if children and all(len(list(ch)) == 0 for ch in children):
                        row = {}
                        for ch in children:
                            tag = ch.tag.split("}")[-1]
                            row[tag] = (ch.text or "").strip()
                        if row:
                            rows.append(row)
                return pd.DataFrame(rows).fillna("")
            except Exception as e:
                warnings.append(f"Failed to parse XML testset {path}: {e}")
                return pd.DataFrame()
    warnings.append(f"Unsupported testset format for {path}; creating empty testset dataframe.")
    return pd.DataFrame()


def build_testset_index(test_df):
    if test_df is None or test_df.empty:
        return {}, None
    id_cols = detect_id_columns(test_df)
    primary = choose_primary_id_column(test_df)
    index = {}
    for _, row in test_df.iterrows():
        for c in id_cols:
            val = safe_str(row.get(c, "")).strip()
            if val and val not in index:
                index[val] = row
    return index, primary


def align_test_row(fused_row, test_index, fused_id_cols):
    candidate_ids = []
    for c in fused_id_cols:
        val = safe_str(fused_row.get(c, "")).strip()
        if val:
            candidate_ids.append(val)
    for c in fused_row.index:
        if "id" in c.lower():
            val = safe_str(fused_row.get(c, "")).strip()
            if val:
                candidate_ids.append(val)
    seen = set()
    for cid in candidate_ids:
        if cid in seen:
            continue
        seen.add(cid)
        if cid in test_index:
            return test_index[cid]
    return None


def main():
    warnings = []
    output_paths = CONTEXT.get("output_paths", {})
    review_attributes = CONTEXT.get("review_attributes", [])

    fused_csv_path = output_paths.get("fused_csv", "output/data_fusion/fusion_data.csv")
    if not os.path.exists(fused_csv_path):
        fallback = "output/data_fusion/fusion_data.csv"
        if os.path.exists(fallback):
            fused_csv_path = fallback
        else:
            warnings.append(f"Fused CSV missing: {output_paths.get('fused_csv')} and fallback {fallback}")

    fused_review_csv = output_paths.get("fused_review_csv", "output/human_review/fused_review_table.csv")
    source_lineage_csv = output_paths.get("source_lineage_csv", "output/human_review/source_lineage_long.csv")
    diff_csv = output_paths.get("diff_csv", "output/human_review/fusion_vs_testset_diff.csv")
    summary_json_path = output_paths.get("review_summary_json", "output/human_review/human_review_summary.json")
    summary_md_path = output_paths.get("review_summary_md", "output/human_review/human_review_summary.md")

    for p in [fused_review_csv, source_lineage_csv, diff_csv, summary_json_path, summary_md_path]:
        ensure_dir(p)

    fused_df = read_csv_flexible(fused_csv_path, warnings) if os.path.exists(fused_csv_path) else pd.DataFrame()
    fused_id_cols = detect_id_columns(fused_df)
    fused_primary_id = choose_primary_id_column(fused_df)

    datasets, global_id_to_rows = build_source_indexes(CONTEXT.get("datasets", []), warnings)

    test_df = read_testset(CONTEXT.get("fusion_testset"), warnings)
    test_index, test_primary_id = build_testset_index(test_df)

    wide_columns = []
    if fused_primary_id:
        wide_columns.append("fused_id")
    else:
        wide_columns.append("fused_id")
    for attr in review_attributes:
        wide_columns.extend([
            f"{attr}_test",
            f"{attr}_fused",
            f"{attr}_source_1",
            f"{attr}_source_2",
            f"{attr}_source_3",
        ])

    wide_rows = []
    lineage_rows = []
    diff_rows = []

    for ridx, fused_row in fused_df.iterrows():
        fused_entity_id = safe_str(fused_row.get(fused_primary_id, "")) if fused_primary_id else ""
        if not fused_entity_id:
            fused_entity_id = safe_str(fused_row.get("_id", "")) or safe_str(fused_row.get("id", "")) or f"row_{ridx}"

        source_ids = parse_fusion_sources(fused_row.get("_fusion_sources", ""))
        source_rows = find_source_rows(source_ids, datasets, global_id_to_rows)[:3]

        test_row = align_test_row(fused_row, test_index, fused_id_cols)

        wide_row = {"fused_id": fused_entity_id}
        for attr in review_attributes:
            fused_val = safe_str(fused_row.get(attr, ""))
            test_val = safe_str(test_row.get(attr, "")) if test_row is not None else ""
            wide_row[f"{attr}_test"] = test_val
            wide_row[f"{attr}_fused"] = fused_val
            for i in range(3):
                source_val = ""
                if i < len(source_rows):
                    source_val = safe_str(source_rows[i]["row"].get(attr, ""))
                wide_row[f"{attr}_source_{i+1}"] = source_val
        wide_rows.append(wide_row)

        for src in source_rows:
            row_out = {
                "fused_id": fused_entity_id,
                "source_id": safe_str(src["source_id"]),
                "source_dataset": safe_str(src["source_dataset"]),
            }
            for attr in review_attributes:
                row_out[f"source_{attr}"] = safe_str(src["row"].get(attr, ""))
                row_out[f"fused_{attr}"] = safe_str(fused_row.get(attr, ""))
            lineage_rows.append(row_out)

        if test_row is not None:
            overlapping = [c for c in review_attributes if c in fused_df.columns and c in test_df.columns]
            if not overlapping:
                overlapping = [c for c in fused_df.columns if c in set(test_df.columns)]
            for attr in overlapping:
                fval = safe_str(fused_row.get(attr, ""))
                tval = safe_str(test_row.get(attr, ""))
                if normalize_text(fval) != normalize_text(tval):
                    diff_rows.append({
                        "fused_id": fused_entity_id,
                        "attribute": attr,
                        "fused_value": fval,
                        "test_value": tval,
                        "difference_type": "mismatch" if fval and tval else ("missing_fused_value" if not fval and tval else "missing_test_value"),
                    })

    wide_df = pd.DataFrame(wide_rows, columns=wide_columns)
    if wide_df.empty:
        wide_df = pd.DataFrame(columns=wide_columns)
    wide_df.to_csv(fused_review_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    lineage_columns = ["fused_id", "source_id", "source_dataset"]
    for attr in review_attributes:
        lineage_columns.extend([f"source_{attr}", f"fused_{attr}"])
    lineage_df = pd.DataFrame(lineage_rows)
    if lineage_df.empty:
        lineage_df = pd.DataFrame(columns=lineage_columns)
    else:
        for c in lineage_columns:
            if c not in lineage_df.columns:
                lineage_df[c] = ""
        lineage_df = lineage_df[lineage_columns]
    lineage_df.to_csv(source_lineage_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    diff_columns = ["fused_id", "attribute", "fused_value", "test_value", "difference_type"]
    diff_df = pd.DataFrame(diff_rows, columns=diff_columns)
    if diff_df.empty:
        diff_df = pd.DataFrame(columns=diff_columns)
    diff_df.to_csv(diff_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    if test_df.empty:
        warnings.append("Fusion testset unavailable or unreadable; created empty fusion_vs_testset_diff.csv.")
    elif diff_df.empty:
        warnings.append("Fusion testset loaded, but no differences were recorded across overlapping columns.")

    counts = {
        "num_fused_rows": int(len(fused_df)),
        "num_review_rows": int(len(wide_df)),
        "num_lineage_rows": int(len(lineage_df)),
        "num_diff_rows": int(len(diff_df)),
        "num_source_datasets": int(len(CONTEXT.get("datasets", []))),
        "num_review_attributes": int(len(review_attributes)),
        "num_testset_rows": int(len(test_df)),
    }

    summary = {
        "summary": "Human review outputs created successfully with graceful handling of missing files, malformed values, and partial source/testset alignment.",
        "file_paths": {
            "fused_review_table": fused_review_csv,
            "source_lineage_long": source_lineage_csv,
            "fusion_vs_testset_diff": diff_csv,
            "human_review_summary_json": summary_json_path,
            "human_review_summary_md": summary_md_path,
        },
        "counts": counts,
        "warnings": warnings,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    md_lines = [
        "# Human Review Summary",
        "",
        f"- Created at: {summary['created_at']}",
        f"- Summary: {summary['summary']}",
        "",
        "## Counts",
    ]
    for k, v in counts.items():
        md_lines.append(f"- {k}: {v}")
    md_lines.extend([
        "",
        "## Output Files",
    ])
    for k, v in summary["file_paths"].items():
        md_lines.append(f"- {k}: `{v}`")
    md_lines.extend([
        "",
        "## Warnings",
    ])
    if warnings:
        md_lines.extend([f"- {w}" for w in warnings])
    else:
        md_lines.append("- None")
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))


if __name__ == "__main__":
    main()