import os
import re
import json
import ast
import csv
from datetime import datetime, timezone
from xml.etree import ElementTree as ET

import pandas as pd


CONTEXT = {
    "datasets": [
        "output/runs/20260323_072905_games/normalization/attempt_1/dbpedia.csv",
        "output/runs/20260323_072905_games/normalization/attempt_1/metacritic.csv",
        "output/runs/20260323_072905_games/normalization/attempt_1/sales.csv"
    ],
    "fusion_testset": "input/datasets/games/testsets/test_set_fusion.xml",
    "review_attributes": [
        "_id",
        "ESRB",
        "criticScore",
        "developer",
        "id",
        "name",
        "platform",
        "releaseYear",
        "series",
        "userScore",
        "globalSales",
        "publisher",
        "eval_id"
    ],
    "evaluation_metrics": {
        "overall_accuracy": 0.8857142857142857,
        "macro_accuracy": 0.8857142857142858,
        "num_evaluated_records": 10,
        "num_evaluated_attributes": 8,
        "total_evaluations": 70,
        "total_correct": 62,
        "criticScore_accuracy": 1.0,
        "criticScore_count": 10,
        "ESRB_accuracy": 0.9,
        "ESRB_count": 10,
        "publisher_accuracy": 0.8,
        "publisher_count": 10,
        "userScore_accuracy": 0.9,
        "userScore_count": 10,
        "releaseYear_accuracy": 0.0,
        "releaseYear_count": 0,
        "developer_accuracy": 0.9,
        "developer_count": 10,
        "name_accuracy": 0.8,
        "name_count": 10,
        "platform_accuracy": 0.9,
        "platform_count": 10
    },
    "auto_diagnostics": {
        "debug_reason_counts": {
            "mismatch": 6,
            "missing_fused_value": 2
        },
        "debug_reason_ratios": {
            "mismatch": 0.75,
            "missing_fused_value": 0.25
        },
        "evaluation_stage": "validation",
        "evaluation_testset_path": "input/datasets/games/testsets/validation_set_fusion.xml",
        "id_alignment": {
            "gold_count": 10,
            "direct_coverage": 3,
            "direct_coverage_ratio": 0.3,
            "mapped_coverage": 10,
            "mapped_coverage_ratio": 1.0,
            "missing_gold_ids": []
        },
        "overall_accuracy": 0.8857142857142857
    },
    "integration_diagnostics_report": {},
    "output_paths": {
        "fused_csv": "output/runs/20260323_072905_games/data_fusion/fusion_data.csv",
        "fused_debug_jsonl": "output/runs/20260323_072905_games/data_fusion/debug_fusion_data.jsonl",
        "evaluation_json": "output/runs/20260323_072905_games/pipeline_evaluation/pipeline_evaluation.json",
        "review_summary_json": "output/runs/20260323_072905_games/human_review/human_review_summary.json",
        "review_summary_md": "output/runs/20260323_072905_games/human_review/human_review_summary.md",
        "fused_review_csv": "output/runs/20260323_072905_games/human_review/fused_review_table.csv",
        "source_lineage_csv": "output/runs/20260323_072905_games/human_review/source_lineage_long.csv",
        "diff_csv": "output/runs/20260323_072905_games/human_review/fusion_vs_testset_diff.csv"
    }
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


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
    if s.lower() in {"nan", "none", "null", ""}:
        return ""
    return s


def read_csv_flex(path, warnings):
    if not path or not os.path.exists(path):
        warnings.append(f"Missing CSV file: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    except Exception:
        try:
            return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        except Exception as e:
            warnings.append(f"Failed to read CSV {path}: {e}")
            return pd.DataFrame()


def detect_id_column(df):
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    preferred = ["_id", "id", "eval_id"]
    lower_map = {c.lower(): c for c in cols}
    for p in preferred:
        if p.lower() in lower_map:
            return lower_map[p.lower()]
    candidates = []
    for c in cols:
        cl = c.lower()
        if cl == "pk" or cl.endswith("_id") or cl == "identifier" or " id" in cl or cl.startswith("id_"):
            candidates.append(c)
    if candidates:
        for c in candidates:
            series = df[c].astype(str)
            non_empty = series[series.str.strip() != ""]
            if len(non_empty) > 0 and non_empty.nunique(dropna=True) >= max(1, int(0.8 * len(non_empty))):
                return c
        return candidates[0]
    for c in cols:
        series = df[c].astype(str)
        non_empty = series[series.str.strip() != ""]
        if len(non_empty) > 0 and non_empty.nunique(dropna=True) == len(non_empty):
            return c
    return cols[0] if cols else None


def parse_fusion_sources(val):
    if val is None:
        return []
    if isinstance(val, list):
        return [normalize_text(x) for x in val if normalize_text(x)]
    s = normalize_text(val)
    if not s:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(s)
            if isinstance(parsed, list):
                return [normalize_text(x) for x in parsed if normalize_text(x)]
            if isinstance(parsed, dict):
                out = []
                for k, v in parsed.items():
                    if isinstance(v, list):
                        out.extend([normalize_text(x) for x in v if normalize_text(x)])
                    else:
                        vv = normalize_text(v)
                        if vv:
                            out.append(vv)
                return out
            if isinstance(parsed, str):
                return [normalize_text(parsed)] if normalize_text(parsed) else []
        except Exception:
            pass
    parts = re.split(r"[|;,]", s)
    cleaned = [normalize_text(p) for p in parts if normalize_text(p)]
    if cleaned:
        return cleaned
    return [s]


def dataset_name_from_path(path):
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name or base or "unknown_dataset"


def build_source_indices(dataset_paths, warnings):
    datasets = []
    global_index = {}
    for path in dataset_paths:
        df = read_csv_flex(path, warnings)
        ds_name = dataset_name_from_path(path)
        id_col = detect_id_column(df) if not df.empty else None
        row_index = {}
        if not df.empty and id_col and id_col in df.columns:
            for _, row in df.iterrows():
                rid = normalize_text(row.get(id_col, ""))
                if rid and rid not in row_index:
                    row_index[rid] = row.to_dict()
                    global_index[rid] = {"dataset": ds_name, "row": row.to_dict(), "id_col": id_col, "path": path}
        else:
            warnings.append(f"Could not detect ID column for dataset: {path}")
        datasets.append({
            "path": path,
            "name": ds_name,
            "df": df,
            "id_col": id_col,
            "row_index": row_index
        })
    return datasets, global_index


def try_parse_xml_testset(path, warnings):
    columns = ["_id", "id", "eval_id"]
    if not path or not os.path.exists(path):
        warnings.append(f"Fusion testset file not found: {path}")
        return pd.DataFrame(columns=columns)
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except Exception as e:
        warnings.append(f"Failed to parse fusion testset XML {path}: {e}")
        return pd.DataFrame(columns=columns)

    rows = []
    candidate_elems = [elem for elem in root.iter() if len(list(elem)) > 0]
    best = None
    best_score = -1
    for elem in candidate_elems:
        child_tags = [child.tag for child in list(elem) if len(list(child)) == 0]
        score = len(set(child_tags))
        if score > best_score:
            best = elem
            best_score = score

    if best is None:
        return pd.DataFrame(columns=columns)

    parent = root
    best_tag = best.tag
    repeated = [elem for elem in root.iter() if elem.tag == best_tag and len(list(elem)) > 0]
    for elem in repeated:
        row = {}
        attrs = dict(elem.attrib)
        for k, v in attrs.items():
            row[k] = safe_str(v)
        for child in list(elem):
            if len(list(child)) == 0:
                row[child.tag] = safe_str(child.text)
            else:
                text_children = [gc for gc in list(child) if len(list(gc)) == 0]
                if text_children:
                    for gc in text_children:
                        key = f"{child.tag}.{gc.tag}"
                        row[key] = safe_str(gc.text)
        if row:
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows, dtype=str).fillna("")
    return df


def choose_fused_id_col(df):
    return detect_id_column(df)


def build_testset_index(test_df):
    idx = {}
    if test_df is None or test_df.empty:
        return idx
    for c in ["_id", "id", "eval_id"]:
        if c in test_df.columns:
            for _, row in test_df.iterrows():
                key = normalize_text(row.get(c, ""))
                if key and key not in idx:
                    idx[key] = row.to_dict()
    return idx


def map_test_rows_to_fused(fused_df, fused_id_col, test_index, review_attributes):
    mapping = {}
    if fused_df is None or fused_df.empty:
        return mapping
    candidate_cols = []
    for c in [fused_id_col, "_id", "id", "eval_id"]:
        if c and c in fused_df.columns and c not in candidate_cols:
            candidate_cols.append(c)
    for _, row in fused_df.iterrows():
        f_id = normalize_text(row.get(fused_id_col, "")) if fused_id_col else ""
        matched = None
        for c in candidate_cols:
            key = normalize_text(row.get(c, ""))
            if key and key in test_index:
                matched = test_index[key]
                break
        if matched is None:
            fused_vals = {a: normalize_text(row.get(a, "")) for a in review_attributes if a in fused_df.columns}
            best_score = 0
            best_row = None
            for trow in test_index.values():
                score = 0
                for a, fv in fused_vals.items():
                    tv = normalize_text(trow.get(a, ""))
                    if fv and tv and fv == tv:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_row = trow
            if best_score >= 2:
                matched = best_row
        if f_id:
            mapping[f_id] = matched or {}
    return mapping


def build_review_table(fused_df, fused_id_col, review_attributes, source_global_index, test_mapping):
    required_columns = []
    for attr in review_attributes:
        required_columns.extend([
            f"{attr}_test",
            f"{attr}_fused",
            f"{attr}_source_1",
            f"{attr}_source_2",
            f"{attr}_source_3",
        ])

    rows = []
    if fused_df is None or fused_df.empty:
        return pd.DataFrame(columns=required_columns)

    for _, row in fused_df.iterrows():
        out = {col: "" for col in required_columns}
        source_ids = parse_fusion_sources(row.get("_fusion_sources", ""))
        source_rows = []
        for sid in source_ids[:3]:
            source_info = source_global_index.get(sid)
            if source_info:
                source_rows.append(source_info["row"])
            else:
                source_rows.append({})
        while len(source_rows) < 3:
            source_rows.append({})

        f_id = normalize_text(row.get(fused_id_col, "")) if fused_id_col else ""
        test_row = test_mapping.get(f_id, {}) if f_id else {}

        for attr in review_attributes:
            out[f"{attr}_test"] = normalize_text(test_row.get(attr, "")) if test_row else ""
            out[f"{attr}_fused"] = normalize_text(row.get(attr, "")) if attr in fused_df.columns else ""
            out[f"{attr}_source_1"] = normalize_text(source_rows[0].get(attr, "")) if source_rows[0] else ""
            out[f"{attr}_source_2"] = normalize_text(source_rows[1].get(attr, "")) if source_rows[1] else ""
            out[f"{attr}_source_3"] = normalize_text(source_rows[2].get(attr, "")) if source_rows[2] else ""
        rows.append(out)

    return pd.DataFrame(rows, columns=required_columns)


def build_lineage_long(fused_df, fused_id_col, review_attributes, source_global_index):
    rows = []
    if fused_df is None or fused_df.empty:
        return pd.DataFrame(columns=["fused_id", "source_id", "source_dataset"] +
                                   [f"source__{a}" for a in review_attributes] +
                                   [f"fused__{a}" for a in review_attributes])
    for _, frow in fused_df.iterrows():
        fused_id = normalize_text(frow.get(fused_id_col, "")) if fused_id_col else ""
        source_ids = parse_fusion_sources(frow.get("_fusion_sources", ""))
        if not source_ids:
            row = {"fused_id": fused_id, "source_id": "", "source_dataset": ""}
            for a in review_attributes:
                row[f"source__{a}"] = ""
                row[f"fused__{a}"] = normalize_text(frow.get(a, "")) if a in fused_df.columns else ""
            rows.append(row)
            continue
        for sid in source_ids:
            sinfo = source_global_index.get(sid, {})
            srow = sinfo.get("row", {})
            ds = sinfo.get("dataset", "")
            row = {"fused_id": fused_id, "source_id": sid, "source_dataset": ds}
            for a in review_attributes:
                row[f"source__{a}"] = normalize_text(srow.get(a, "")) if srow else ""
                row[f"fused__{a}"] = normalize_text(frow.get(a, "")) if a in fused_df.columns else ""
            rows.append(row)
    columns = ["fused_id", "source_id", "source_dataset"] + \
              [f"source__{a}" for a in review_attributes] + \
              [f"fused__{a}" for a in review_attributes]
    return pd.DataFrame(rows, columns=columns)


def build_diff_csv(fused_df, fused_id_col, review_attributes, test_mapping, test_df, warnings):
    columns = ["fused_id", "attribute", "fused_value", "test_value", "is_diff", "note"]
    if test_df is None or test_df.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    if fused_df is None or fused_df.empty:
        return pd.DataFrame(columns=columns)
    overlapping = [a for a in review_attributes if a in fused_df.columns and a in test_df.columns]
    for _, frow in fused_df.iterrows():
        fused_id = normalize_text(frow.get(fused_id_col, "")) if fused_id_col else ""
        trow = test_mapping.get(fused_id, {})
        if not trow:
            rows.append({
                "fused_id": fused_id,
                "attribute": "",
                "fused_value": "",
                "test_value": "",
                "is_diff": "",
                "note": "No matched testset row"
            })
            continue
        for attr in overlapping:
            fv = normalize_text(frow.get(attr, ""))
            tv = normalize_text(trow.get(attr, ""))
            if fv != tv:
                rows.append({
                    "fused_id": fused_id,
                    "attribute": attr,
                    "fused_value": fv,
                    "test_value": tv,
                    "is_diff": "1",
                    "note": "mismatch"
                })
    return pd.DataFrame(rows, columns=columns)


def write_csv(df, path):
    ensure_dir(os.path.dirname(path))
    if df is None:
        df = pd.DataFrame()
    df.to_csv(path, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)


def main():
    warnings = []
    output_paths = CONTEXT.get("output_paths", {})
    review_attributes = CONTEXT.get("review_attributes", [])

    fused_csv = output_paths.get("fused_csv", "output/data_fusion/fusion_data.csv")
    if not os.path.exists(fused_csv):
        fallback = "output/data_fusion/fusion_data.csv"
        if os.path.exists(fallback):
            fused_csv = fallback
        else:
            warnings.append(f"Fused CSV not found at configured path or fallback: {output_paths.get('fused_csv')} | {fallback}")

    fused_review_csv = output_paths.get("fused_review_csv", "output/human_review/fused_review_table.csv")
    source_lineage_csv = output_paths.get("source_lineage_csv", "output/human_review/source_lineage_long.csv")
    diff_csv = output_paths.get("diff_csv", "output/human_review/fusion_vs_testset_diff.csv")
    summary_json = output_paths.get("review_summary_json", "output/human_review/human_review_summary.json")
    summary_md = output_paths.get("review_summary_md", "output/human_review/human_review_summary.md")

    for p in [fused_review_csv, source_lineage_csv, diff_csv, summary_json, summary_md]:
        ensure_dir(os.path.dirname(p))

    fused_df = read_csv_flex(fused_csv, warnings)
    fused_id_col = choose_fused_id_col(fused_df) if not fused_df.empty else None
    if fused_df.empty:
        warnings.append("Fused dataframe is empty or unreadable.")
    if not fused_id_col and not fused_df.empty:
        warnings.append("Could not robustly detect fused ID column.")

    dataset_paths = CONTEXT.get("datasets", [])
    _, source_global_index = build_source_indices(dataset_paths, warnings)

    test_df = try_parse_xml_testset(CONTEXT.get("fusion_testset"), warnings)
    test_index = build_testset_index(test_df)
    test_mapping = map_test_rows_to_fused(fused_df, fused_id_col, test_index, review_attributes) if fused_id_col else {}

    review_df = build_review_table(fused_df, fused_id_col, review_attributes, source_global_index, test_mapping)
    lineage_df = build_lineage_long(fused_df, fused_id_col, review_attributes, source_global_index)
    diff_df = build_diff_csv(fused_df, fused_id_col, review_attributes, test_mapping, test_df, warnings)

    write_csv(review_df, fused_review_csv)
    write_csv(lineage_df, source_lineage_csv)
    write_csv(diff_df, diff_csv)

    counts = {
        "fused_rows": int(len(fused_df)) if fused_df is not None else 0,
        "review_rows": int(len(review_df)) if review_df is not None else 0,
        "lineage_rows": int(len(lineage_df)) if lineage_df is not None else 0,
        "diff_rows": int(len(diff_df)) if diff_df is not None else 0,
        "source_datasets": int(len(dataset_paths)),
        "source_indexed_records": int(len(source_global_index)),
        "testset_rows": int(len(test_df)) if test_df is not None else 0,
        "testset_mapped_rows": int(sum(1 for v in test_mapping.values() if v)),
    }

    summary_text = "Human-review outputs created."
    if test_df.empty:
        summary_text += " Fusion testset unavailable or unparsable; diff file created as empty or minimal."
    elif counts["testset_mapped_rows"] == 0:
        summary_text += " Testset loaded but no fused rows were confidently mapped."
    else:
        summary_text += f" Testset mapped for {counts['testset_mapped_rows']} fused rows."

    summary = {
        "summary": summary_text,
        "file_paths": {
            "fused_review_table": fused_review_csv,
            "source_lineage_long": source_lineage_csv,
            "fusion_vs_testset_diff": diff_csv,
            "human_review_summary_json": summary_json,
            "human_review_summary_md": summary_md,
        },
        "counts": counts,
        "warnings": warnings,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    md_lines = [
        "# Human Review Summary",
        "",
        f"**Summary:** {summary['summary']}",
        "",
        "## Files",
    ]
    for k, v in summary["file_paths"].items():
        md_lines.append(f"- **{k}**: `{v}`")
    md_lines.extend([
        "",
        "## Counts",
    ])
    for k, v in counts.items():
        md_lines.append(f"- **{k}**: {v}")
    md_lines.extend([
        "",
        "## Warnings",
    ])
    if warnings:
        for w in warnings:
            md_lines.append(f"- {w}")
    else:
        md_lines.append("- None")
    md_lines.extend([
        "",
        f"**Created at:** {summary['created_at']}",
        ""
    ])

    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))


if __name__ == "__main__":
    main()