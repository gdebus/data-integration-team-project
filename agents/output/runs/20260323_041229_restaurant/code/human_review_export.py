import os
import re
import json
import ast
from datetime import datetime, timezone

import pandas as pd


CONTEXT = {
    "datasets": [
        "output/runs/20260323_041229_restaurant/normalization/attempt_1/kaggle_small.csv",
        "output/runs/20260323_041229_restaurant/normalization/attempt_1/uber_eats_small.csv",
        "output/runs/20260323_041229_restaurant/normalization/attempt_1/yelp_small.csv"
    ],
    "fusion_testset": "input/datasets/restaurant/testsets/Restaurant_Fusion_Test_Set.csv",
    "review_attributes": [
        "_id",
        "address_line1",
        "address_line2",
        "categories",
        "city",
        "country",
        "house_number",
        "id",
        "latitude",
        "longitude",
        "map_url",
        "name",
        "name_norm",
        "phone_e164",
        "phone_raw",
        "postal_code",
        "rating",
        "rating_count",
        "source",
        "state",
        "street",
        "website",
        "eval_id"
    ],
    "evaluation_metrics": {
        "overall_accuracy": 0.86,
        "macro_accuracy": 0.8666666666666668,
        "num_evaluated_records": 20,
        "num_evaluated_attributes": 21,
        "total_evaluations": 400,
        "total_correct": 344,
        "address_line2_accuracy": 1.0,
        "address_line2_count": 5,
        "rating_count_accuracy": 1.0,
        "rating_count_count": 16,
        "street_accuracy": 1.0,
        "street_count": 20,
        "name_norm_accuracy": 0.9,
        "name_norm_count": 20,
        "country_accuracy": 1.0,
        "country_count": 20,
        "rating_accuracy": 0.5,
        "rating_count": 20,
        "postal_code_accuracy": 1.0,
        "postal_code_count": 20,
        "name_accuracy": 0.9,
        "name_count": 20,
        "latitude_accuracy": 1.0,
        "latitude_count": 20,
        "source_accuracy": 0.3,
        "source_count": 20,
        "_id_accuracy": 0.3,
        "_id_count": 20,
        "longitude_accuracy": 1.0,
        "longitude_count": 20,
        "city_accuracy": 1.0,
        "city_count": 20,
        "house_number_accuracy": 1.0,
        "house_number_count": 20,
        "website_accuracy": 1.0,
        "website_count": 19,
        "phone_raw_accuracy": 1.0,
        "phone_raw_count": 20,
        "address_line1_accuracy": 0.8,
        "address_line1_count": 20,
        "state_accuracy": 1.0,
        "state_count": 20,
        "categories_accuracy": 1.0,
        "categories_count": 20,
        "phone_e164_accuracy": 1.0,
        "phone_e164_count": 20,
        "map_url_accuracy": 0.5,
        "map_url_count": 20
    },
    "auto_diagnostics": {
        "debug_reason_counts": {
            "mismatch": 56
        },
        "debug_reason_ratios": {
            "mismatch": 1.0
        },
        "evaluation_stage": "validation",
        "evaluation_testset_path": "input/datasets/restaurant/testsets/Restaurant_Fusion_Validation_Set.csv",
        "id_alignment": {
            "gold_count": 20,
            "direct_coverage": 6,
            "direct_coverage_ratio": 0.3,
            "mapped_coverage": 20,
            "mapped_coverage_ratio": 1.0,
            "missing_gold_ids": []
        },
        "overall_accuracy": 0.86
    },
    "integration_diagnostics_report": {},
    "output_paths": {
        "fused_csv": "output/runs/20260323_041229_restaurant/data_fusion/fusion_data.csv",
        "fused_debug_jsonl": "output/runs/20260323_041229_restaurant/data_fusion/debug_fusion_data.jsonl",
        "evaluation_json": "output/runs/20260323_041229_restaurant/pipeline_evaluation/pipeline_evaluation.json",
        "review_summary_json": "output/runs/20260323_041229_restaurant/human_review/human_review_summary.json",
        "review_summary_md": "output/runs/20260323_041229_restaurant/human_review/human_review_summary.md",
        "fused_review_csv": "output/runs/20260323_041229_restaurant/human_review/fused_review_table.csv",
        "source_lineage_csv": "output/runs/20260323_041229_restaurant/human_review/source_lineage_long.csv",
        "diff_csv": "output/runs/20260323_041229_restaurant/human_review/fusion_vs_testset_diff.csv"
    }
}


def ensure_parent_dir(path):
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


def normalize_key(s):
    return re.sub(r'[^a-z0-9]+', '', safe_str(s).strip().lower())


def read_csv_safe(path, warnings):
    if not path or not os.path.exists(path):
        warnings.append(f"Missing CSV: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as e:
        warnings.append(f"Failed reading CSV {path}: {e}")
        try:
            return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="latin1")
        except Exception as e2:
            warnings.append(f"Retry failed for CSV {path}: {e2}")
            return pd.DataFrame()


def detect_id_column(df):
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    priority_exact = ["_id", "id", "eval_id", "entity_id", "record_id", "fusion_id", "uuid", "pk"]
    for c in priority_exact:
        if c in cols:
            return c
    norm_map = {c: normalize_key(c) for c in cols}
    ranked = []
    for c in cols:
        n = norm_map[c]
        score = 0
        if n in {"id", "_id", "evalid", "entityid", "recordid", "fusionid", "uuid", "pk"}:
            score += 100
        if n.endswith("id"):
            score += 40
        if "id" == n:
            score += 30
        if "source" in n and "id" in n:
            score += 10
        non_empty = int((df[c].astype(str).str.strip() != "").sum())
        unique = int(df[c].astype(str).nunique(dropna=False))
        ratio = unique / non_empty if non_empty else 0
        score += ratio * 10
        ranked.append((score, c))
    ranked.sort(reverse=True)
    return ranked[0][1] if ranked else None


def parse_fusion_sources(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [safe_str(x).strip() for x in value if safe_str(x).strip()]
    s = safe_str(value).strip()
    if not s or s.lower() in {"nan", "none", "null", "[]", "{}"}:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            obj = parser(s)
            if isinstance(obj, list):
                out = []
                for item in obj:
                    if isinstance(item, dict):
                        sid = safe_str(item.get("id") or item.get("source_id") or item.get("_id"))
                        if sid.strip():
                            out.append(sid.strip())
                    else:
                        val = safe_str(item).strip()
                        if val:
                            out.append(val)
                return out
            if isinstance(obj, dict):
                out = []
                for key in ("ids", "source_ids", "sources"):
                    if key in obj and isinstance(obj[key], list):
                        for item in obj[key]:
                            val = safe_str(item).strip()
                            if val:
                                out.append(val)
                        if out:
                            return out
                sid = safe_str(obj.get("id") or obj.get("source_id") or obj.get("_id")).strip()
                return [sid] if sid else []
        except Exception:
            pass
    if "|" in s:
        return [x.strip() for x in s.split("|") if x.strip()]
    if ";" in s:
        return [x.strip() for x in s.split(";") if x.strip()]
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    return [s] if s else []


def build_source_indexes(dataset_paths, warnings):
    datasets = []
    global_index = {}
    for path in dataset_paths:
        df = read_csv_safe(path, warnings)
        dataset_name = os.path.splitext(os.path.basename(path))[0]
        id_col = detect_id_column(df)
        if df.empty:
            datasets.append({"path": path, "name": dataset_name, "df": df, "id_col": id_col, "index": {}})
            continue
        index = {}
        if id_col and id_col in df.columns:
            for _, row in df.iterrows():
                rid = safe_str(row.get(id_col)).strip()
                if rid:
                    index[rid] = row.to_dict()
                    global_index[rid] = {"dataset": dataset_name, "row": row.to_dict(), "path": path}
        datasets.append({"path": path, "name": dataset_name, "df": df, "id_col": id_col, "index": index})
        if not id_col:
            warnings.append(f"Could not robustly detect ID column for source dataset: {path}")
    return datasets, global_index


def detect_fused_id_col(df):
    if df is None or df.empty:
        return None
    preferred = ["_id", "id", "eval_id", "fusion_id", "entity_id", "record_id"]
    for c in preferred:
        if c in df.columns:
            return c
    return detect_id_column(df)


def detect_source_field(df):
    candidates = ["_fusion_sources", "fusion_sources", "source_ids", "sources", "_sources"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_testset_index(test_df, review_attributes):
    if test_df is None or test_df.empty:
        return None, {}
    candidate_cols = []
    for c in ["eval_id", "_id", "id"]:
        if c in test_df.columns:
            candidate_cols.append(c)
    for c in review_attributes:
        if c in test_df.columns and c not in candidate_cols and normalize_key(c).endswith("id"):
            candidate_cols.append(c)
    if not candidate_cols:
        id_col = detect_id_column(test_df)
        candidate_cols = [id_col] if id_col else []
    index = {}
    for c in candidate_cols:
        sub = {}
        for _, row in test_df.iterrows():
            key = safe_str(row.get(c)).strip()
            if key:
                sub[key] = row.to_dict()
        if sub:
            index[c] = sub
    primary = candidate_cols[0] if candidate_cols else None
    return primary, index


def find_test_row(fused_row, test_indexes, review_attributes):
    if not test_indexes:
        return {}
    candidates = []
    for c in ["eval_id", "_id", "id"]:
        val = safe_str(fused_row.get(c)).strip()
        if val:
            candidates.append((c, val))
    for c in review_attributes:
        if normalize_key(c).endswith("id"):
            val = safe_str(fused_row.get(c)).strip()
            if val:
                candidates.append((c, val))
    seen = set()
    for col_name, val in candidates:
        if (col_name, val) in seen:
            continue
        seen.add((col_name, val))
        for test_col, idx in test_indexes.items():
            if val in idx:
                return idx[val]
    return {}


def rows_equal(a, b):
    return safe_str(a).strip() == safe_str(b).strip()


def main():
    warnings = []
    review_attributes = CONTEXT.get("review_attributes", [])
    output_paths = CONTEXT.get("output_paths", {})
    fused_csv_path = output_paths.get("fused_csv") or "output/data_fusion/fusion_data.csv"
    fallback_fused = "output/data_fusion/fusion_data.csv"
    if not os.path.exists(fused_csv_path) and os.path.exists(fallback_fused):
        fused_csv_path = fallback_fused
    fused_review_csv = output_paths.get("fused_review_csv", "output/human_review/fused_review_table.csv")
    source_lineage_csv = output_paths.get("source_lineage_csv", "output/human_review/source_lineage_long.csv")
    diff_csv = output_paths.get("diff_csv", "output/human_review/fusion_vs_testset_diff.csv")
    summary_json = output_paths.get("review_summary_json", "output/human_review/human_review_summary.json")
    summary_md = output_paths.get("review_summary_md", "output/human_review/human_review_summary.md")

    for p in [fused_review_csv, source_lineage_csv, diff_csv, summary_json, summary_md]:
        ensure_parent_dir(p)

    fused_df = read_csv_safe(fused_csv_path, warnings)
    fused_id_col = detect_fused_id_col(fused_df)
    source_field = detect_source_field(fused_df)

    datasets, source_global_index = build_source_indexes(CONTEXT.get("datasets", []), warnings)

    testset_path = CONTEXT.get("fusion_testset")
    test_df = read_csv_safe(testset_path, warnings) if testset_path else pd.DataFrame()
    test_primary_col, test_indexes = build_testset_index(test_df, review_attributes)

    wide_rows = []
    lineage_rows = []
    diff_rows = []

    mandatory_columns = []
    if fused_id_col:
        mandatory_columns.append("fused_id")
    else:
        mandatory_columns.append("fused_id")
    for attr in review_attributes:
        mandatory_columns.extend([
            f"{attr}_test",
            f"{attr}_fused",
            f"{attr}_source_1",
            f"{attr}_source_2",
            f"{attr}_source_3",
        ])

    if fused_df.empty:
        warnings.append(f"Fused output is missing or empty: {fused_csv_path}")
        wide_df = pd.DataFrame(columns=mandatory_columns)
        lineage_df = pd.DataFrame(columns=["fused_id", "source_id", "source_dataset"] + [f"source__{a}" for a in review_attributes] + [f"fused__{a}" for a in review_attributes])
        diff_df = pd.DataFrame(columns=["fused_id", "test_id", "attribute", "fused_value", "test_value", "is_diff"])
    else:
        if not fused_id_col:
            warnings.append("Could not robustly detect fused ID column; using row index-based fused_id.")
        if not source_field:
            warnings.append("Could not detect source lineage field in fused output; source mappings may be empty.")

        for row_idx, (_, fused_row) in enumerate(fused_df.iterrows()):
            fused_dict = fused_row.to_dict()
            fused_id = safe_str(fused_dict.get(fused_id_col)).strip() if fused_id_col and fused_id_col in fused_df.columns else str(row_idx)

            source_ids = parse_fusion_sources(fused_dict.get(source_field)) if source_field else []
            source_matches = []
            for sid in source_ids:
                meta = source_global_index.get(sid)
                if meta:
                    source_matches.append({"source_id": sid, "dataset": meta["dataset"], "row": meta["row"]})
                else:
                    source_matches.append({"source_id": sid, "dataset": "", "row": {}})
                    warnings.append(f"Source ID from fused row not found in source datasets: {sid}")

            test_row = find_test_row(fused_dict, test_indexes, review_attributes) if test_indexes else {}

            wide_row = {"fused_id": fused_id}
            for attr in review_attributes:
                wide_row[f"{attr}_test"] = safe_str(test_row.get(attr, ""))
                wide_row[f"{attr}_fused"] = safe_str(fused_dict.get(attr, ""))
                for i in range(3):
                    val = ""
                    if i < len(source_matches):
                        val = safe_str(source_matches[i]["row"].get(attr, ""))
                    wide_row[f"{attr}_source_{i+1}"] = val
            wide_rows.append(wide_row)

            for sm in source_matches:
                lineage_row = {
                    "fused_id": fused_id,
                    "source_id": sm["source_id"],
                    "source_dataset": sm["dataset"],
                }
                for attr in review_attributes:
                    lineage_row[f"source__{attr}"] = safe_str(sm["row"].get(attr, ""))
                    lineage_row[f"fused__{attr}"] = safe_str(fused_dict.get(attr, ""))
                lineage_rows.append(lineage_row)

            if not test_df.empty and test_row:
                test_id_val = ""
                for c in [test_primary_col, "eval_id", "_id", "id"]:
                    if c:
                        test_id_val = safe_str(test_row.get(c, "")).strip()
                        if test_id_val:
                            break
                overlap_cols = [c for c in review_attributes if c in fused_df.columns and c in test_df.columns]
                for attr in overlap_cols:
                    fused_val = safe_str(fused_dict.get(attr, ""))
                    test_val = safe_str(test_row.get(attr, ""))
                    if not rows_equal(fused_val, test_val):
                        diff_rows.append({
                            "fused_id": fused_id,
                            "test_id": test_id_val,
                            "attribute": attr,
                            "fused_value": fused_val,
                            "test_value": test_val,
                            "is_diff": True,
                        })

        wide_df = pd.DataFrame(wide_rows)
        for c in mandatory_columns:
            if c not in wide_df.columns:
                wide_df[c] = ""
        wide_df = wide_df[mandatory_columns]

        lineage_columns = ["fused_id", "source_id", "source_dataset"] + [f"source__{a}" for a in review_attributes] + [f"fused__{a}" for a in review_attributes]
        lineage_df = pd.DataFrame(lineage_rows)
        for c in lineage_columns:
            if c not in lineage_df.columns:
                lineage_df[c] = ""
        lineage_df = lineage_df[lineage_columns]

        diff_columns = ["fused_id", "test_id", "attribute", "fused_value", "test_value", "is_diff"]
        diff_df = pd.DataFrame(diff_rows)
        for c in diff_columns:
            if c not in diff_df.columns:
                diff_df[c] = ""
        diff_df = diff_df[diff_columns]

    if test_df.empty:
        warnings.append("Fusion testset unavailable or empty; created empty diff CSV.")

    try:
        wide_df.to_csv(fused_review_csv, index=False)
    except Exception as e:
        warnings.append(f"Failed writing fused review CSV: {e}")

    try:
        lineage_df.to_csv(source_lineage_csv, index=False)
    except Exception as e:
        warnings.append(f"Failed writing source lineage CSV: {e}")

    try:
        diff_df.to_csv(diff_csv, index=False)
    except Exception as e:
        warnings.append(f"Failed writing diff CSV: {e}")

    counts = {
        "fused_rows": int(len(fused_df)) if fused_df is not None else 0,
        "review_rows": int(len(wide_df)),
        "lineage_rows": int(len(lineage_df)),
        "diff_rows": int(len(diff_df)),
        "source_datasets": int(len(CONTEXT.get("datasets", []))),
        "review_attributes": int(len(review_attributes)),
        "testset_rows": int(len(test_df)) if test_df is not None else 0,
    }

    summary = {
        "summary": "Human-review outputs created for fused entities, source lineage, and fusion-vs-testset diffs.",
        "file_paths": {
            "fused_review_table": fused_review_csv,
            "source_lineage_long": source_lineage_csv,
            "fusion_vs_testset_diff": diff_csv,
            "human_review_summary_json": summary_json,
            "human_review_summary_md": summary_md,
            "fused_csv_input": fused_csv_path,
            "fusion_testset_input": testset_path or "",
        },
        "counts": counts,
        "warnings": warnings,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

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
    md_lines.extend(["", "## Files"])
    for k, v in summary["file_paths"].items():
        md_lines.append(f"- {k}: {v}")
    md_lines.extend(["", "## Warnings"])
    if warnings:
        for w in warnings:
            md_lines.append(f"- {w}")
    else:
        md_lines.append("- None")

    try:
        with open(summary_md, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
    except Exception:
        pass


if __name__ == "__main__":
    main()