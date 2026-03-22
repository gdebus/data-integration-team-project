import os
import re
import json
import ast
from datetime import datetime, timezone

import pandas as pd


CONTEXT_PAYLOAD = {
    "datasets": [
        "output/runs/20260322_162759_restaurant/normalization/attempt_1/kaggle_small.csv",
        "output/runs/20260322_162759_restaurant/normalization/attempt_1/uber_eats_small.csv",
        "output/runs/20260322_162759_restaurant/normalization/attempt_1/yelp_small.csv"
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
        "overall_accuracy": 0.8842105263157894,
        "macro_accuracy": 0.89,
        "num_evaluated_records": 20,
        "num_evaluated_attributes": 21,
        "total_evaluations": 380,
        "total_correct": 336,
        "longitude_accuracy": 1.0,
        "longitude_count": 20,
        "rating_count_accuracy": 1.0,
        "rating_count_count": 16,
        "address_line1_accuracy": 0.95,
        "address_line1_count": 20,
        "city_accuracy": 1.0,
        "city_count": 20,
        "phone_raw_accuracy": 1.0,
        "phone_raw_count": 20,
        "latitude_accuracy": 1.0,
        "latitude_count": 20,
        "address_line2_accuracy": 1.0,
        "address_line2_count": 5,
        "phone_e164_accuracy": 1.0,
        "phone_e164_count": 20,
        "map_url_accuracy": 0.7,
        "map_url_count": 20,
        "country_accuracy": 0.0,
        "country_count": 0,
        "_id_accuracy": 0.3,
        "_id_count": 20,
        "postal_code_accuracy": 1.0,
        "postal_code_count": 20,
        "website_accuracy": 1.0,
        "website_count": 19,
        "state_accuracy": 1.0,
        "state_count": 20,
        "name_norm_accuracy": 0.9,
        "name_norm_count": 20,
        "street_accuracy": 0.75,
        "street_count": 20,
        "categories_accuracy": 1.0,
        "categories_count": 20,
        "source_accuracy": 0.7,
        "source_count": 20,
        "name_accuracy": 0.8,
        "name_count": 20,
        "house_number_accuracy": 1.0,
        "house_number_count": 20,
        "rating_accuracy": 0.7,
        "rating_count": 20
    },
    "auto_diagnostics": {
        "debug_reason_counts": {
            "mismatch": 44
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
        "overall_accuracy": 0.8842105263157894
    },
    "integration_diagnostics_report": {},
    "output_paths": {
        "fused_csv": "output/runs/20260322_162759_restaurant/data_fusion/fusion_data.csv",
        "fused_debug_jsonl": "output/runs/20260322_162759_restaurant/data_fusion/debug_fusion_data.jsonl",
        "evaluation_json": "output/runs/20260322_162759_restaurant/pipeline_evaluation/pipeline_evaluation.json",
        "review_summary_json": "output/runs/20260322_162759_restaurant/human_review/human_review_summary.json",
        "review_summary_md": "output/runs/20260322_162759_restaurant/human_review/human_review_summary.md",
        "fused_review_csv": "output/runs/20260322_162759_restaurant/human_review/fused_review_table.csv",
        "source_lineage_csv": "output/runs/20260322_162759_restaurant/human_review/source_lineage_long.csv",
        "diff_csv": "output/runs/20260322_162759_restaurant/human_review/fusion_vs_testset_diff.csv"
    }
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def stringify(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
    return str(value)


def normalize_text(value):
    s = stringify(value).strip()
    if s == "":
        return ""
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.casefold()


def safe_read_csv(path, warnings):
    if not path or not os.path.exists(path):
        warnings.append(f"Missing CSV file: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as e:
        warnings.append(f"Failed to read CSV {path}: {e}")
        try:
            return pd.read_csv(path, dtype=str, keep_default_na=False, engine="python", on_bad_lines="skip")
        except Exception as e2:
            warnings.append(f"Fallback read failed for {path}: {e2}")
            return pd.DataFrame()


def detect_id_column(df, preferred=None):
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}

    if preferred:
        for p in preferred:
            if p in cols:
                return p
            if p.lower() in lower_map:
                return lower_map[p.lower()]

    candidates = [
        "_id", "id", "eval_id", "entity_id", "record_id", "fusion_id",
        "source_id", "row_id", "uuid", "pk"
    ]
    for c in candidates:
        if c in cols:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    scored = []
    for c in cols:
        lc = c.lower()
        score = 0
        if lc == "id":
            score += 100
        if lc.endswith("_id") or lc.startswith("id_") or "id" == lc:
            score += 80
        elif "id" in lc:
            score += 40
        non_empty = df[c].astype(str).str.strip().replace("", pd.NA).dropna()
        uniq = non_empty.nunique(dropna=True)
        total = len(non_empty)
        if total > 0:
            ratio = uniq / total
            score += int(ratio * 20)
        scored.append((score, c))
    scored.sort(reverse=True)
    return scored[0][1] if scored else None


def parse_fusion_sources(value):
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass

    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]

    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return []

    parsed = None
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(s)
            break
        except Exception:
            continue

    if parsed is None:
        parts = [p.strip() for p in re.split(r"[|;,]", s) if p.strip()]
        return [{"raw": p} for p in parts]

    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, str):
        return [{"raw": parsed}]
    return [{"raw": stringify(parsed)}]


def source_dataset_name(path):
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name or base


def build_dataset_lookup(df, dataset_name, warnings):
    id_col = detect_id_column(df)
    lookup = {}
    normalized_lookup = {}
    if df is None or df.empty:
        warnings.append(f"Dataset {dataset_name} is empty or unavailable.")
        return {"dataset_name": dataset_name, "df": df, "id_col": id_col, "lookup": lookup, "normalized_lookup": normalized_lookup}

    if id_col is None:
        warnings.append(f"Could not detect ID column for dataset {dataset_name}.")
        return {"dataset_name": dataset_name, "df": df, "id_col": id_col, "lookup": lookup, "normalized_lookup": normalized_lookup}

    for _, row in df.iterrows():
        raw_id = stringify(row.get(id_col, ""))
        if raw_id != "":
            lookup[raw_id] = row
            normalized_lookup[normalize_text(raw_id)] = row

    return {
        "dataset_name": dataset_name,
        "df": df,
        "id_col": id_col,
        "lookup": lookup,
        "normalized_lookup": normalized_lookup
    }


def extract_candidate_source_ids(item):
    ids = []
    if isinstance(item, dict):
        keys = [
            "source_id", "id", "_id", "record_id", "row_id",
            "entity_id", "original_id", "pk", "raw"
        ]
        for k in keys:
            if k in item and stringify(item.get(k)) != "":
                ids.append(stringify(item.get(k)))
        for k, v in item.items():
            if "id" in str(k).lower() and stringify(v) != "":
                ids.append(stringify(v))
    else:
        val = stringify(item)
        if val != "":
            ids.append(val)
    deduped = []
    seen = set()
    for x in ids:
        nx = normalize_text(x)
        if nx not in seen and nx != "":
            seen.add(nx)
            deduped.append(x)
    return deduped


def extract_candidate_dataset_names(item):
    names = []
    if isinstance(item, dict):
        keys = ["source_dataset", "dataset", "source", "table", "file", "origin"]
        for k in keys:
            if k in item and stringify(item.get(k)) != "":
                names.append(stringify(item.get(k)))
    deduped = []
    seen = set()
    for x in names:
        nx = normalize_text(os.path.splitext(os.path.basename(x))[0])
        if nx not in seen and nx != "":
            seen.add(nx)
            deduped.append(x)
    return deduped


def match_source_rows(parsed_sources, dataset_infos, warnings):
    matched = []
    for item in parsed_sources:
        candidate_ids = extract_candidate_source_ids(item)
        candidate_datasets = extract_candidate_dataset_names(item)

        dataset_priority = []
        if candidate_datasets:
            for ds_name in candidate_datasets:
                ds_norm = normalize_text(os.path.splitext(os.path.basename(ds_name))[0])
                for info in dataset_infos:
                    info_norm = normalize_text(info["dataset_name"])
                    if ds_norm == info_norm or ds_norm in info_norm or info_norm in ds_norm:
                        dataset_priority.append(info)
            for info in dataset_infos:
                if info not in dataset_priority:
                    dataset_priority.append(info)
        else:
            dataset_priority = list(dataset_infos)

        found = None
        for cid in candidate_ids:
            cid_norm = normalize_text(cid)
            for info in dataset_priority:
                if cid in info["lookup"]:
                    found = (cid, info)
                    break
                if cid_norm in info["normalized_lookup"]:
                    found = (cid, info)
                    break
            if found:
                break

        if found:
            cid, info = found
            row = info["lookup"].get(cid)
            if row is None:
                row = info["normalized_lookup"].get(normalize_text(cid))
            matched.append({
                "source_id": stringify(row.get(info["id_col"], cid)) if row is not None else cid,
                "source_dataset": info["dataset_name"],
                "source_row": row if row is not None else pd.Series(dtype=object)
            })
        else:
            matched.append({
                "source_id": candidate_ids[0] if candidate_ids else "",
                "source_dataset": candidate_datasets[0] if candidate_datasets else "",
                "source_row": pd.Series(dtype=object)
            })
            warnings.append(f"Could not map source reference: {stringify(item)}")
    return matched


def find_fused_id_col(df):
    return detect_id_column(df, preferred=["_id", "id", "fusion_id", "entity_id", "eval_id"])


def build_testset_index(test_df, review_attributes, warnings):
    if test_df is None or test_df.empty:
        return None, None, {}

    test_id_col = detect_id_column(test_df, preferred=["eval_id", "_id", "id", "fusion_id", "entity_id"])
    if test_id_col is None:
        warnings.append("Could not detect ID column in fusion_testset.")
    test_indexes = {}
    for col in [c for c in [test_id_col, "_id", "id", "eval_id"] if c and c in test_df.columns]:
        idx = {}
        for _, row in test_df.iterrows():
            val = stringify(row.get(col, ""))
            if val != "":
                idx[val] = row
                idx[normalize_text(val)] = row
        test_indexes[col] = idx
    return test_id_col, test_df, test_indexes


def match_test_row(fused_row, review_attributes, test_indexes):
    if not test_indexes:
        return None

    candidates = []
    for key in ["eval_id", "_id", "id"]:
        if key in fused_row.index:
            val = stringify(fused_row.get(key, ""))
            if val != "":
                candidates.extend([val, normalize_text(val)])

    for idx in test_indexes.values():
        for c in candidates:
            if c in idx:
                return idx[c]

    overlap_attrs = [a for a in review_attributes if a in fused_row.index]
    best_row = None
    best_score = 0
    sample_indexes = list(test_indexes.values())
    if not sample_indexes:
        return None
    sample_rows_seen = set()
    for idx in sample_indexes:
        for _, row in idx.items():
            row_key = id(row)
            if row_key in sample_rows_seen:
                continue
            sample_rows_seen.add(row_key)
            score = 0
            for a in overlap_attrs:
                fv = normalize_text(fused_row.get(a, ""))
                tv = normalize_text(row.get(a, ""))
                if fv != "" and tv != "" and fv == tv:
                    score += 1
            if score > best_score:
                best_score = score
                best_row = row
    return best_row if best_score > 0 else None


def main():
    warnings = []
    outputs = CONTEXT_PAYLOAD.get("output_paths", {})
    review_attributes = CONTEXT_PAYLOAD.get("review_attributes", [])

    fused_csv = outputs.get("fused_csv") or "output/data_fusion/fusion_data.csv"
    if not os.path.exists(fused_csv):
        fallback_fused = "output/data_fusion/fusion_data.csv"
        if os.path.exists(fallback_fused):
            fused_csv = fallback_fused
        else:
            warnings.append(f"Fused CSV not found at configured path or fallback: {fused_csv}, {fallback_fused}")

    fused_review_csv = outputs.get("fused_review_csv", "output/human_review/fused_review_table.csv")
    source_lineage_csv = outputs.get("source_lineage_csv", "output/human_review/source_lineage_long.csv")
    diff_csv = outputs.get("diff_csv", "output/human_review/fusion_vs_testset_diff.csv")
    summary_json = outputs.get("review_summary_json", "output/human_review/human_review_summary.json")
    summary_md = outputs.get("review_summary_md", "output/human_review/human_review_summary.md")

    ensure_dir(os.path.dirname(fused_review_csv) or ".")
    ensure_dir(os.path.dirname(source_lineage_csv) or ".")
    ensure_dir(os.path.dirname(diff_csv) or ".")
    ensure_dir(os.path.dirname(summary_json) or ".")
    ensure_dir(os.path.dirname(summary_md) or ".")

    fused_df = safe_read_csv(fused_csv, warnings)
    fused_id_col = find_fused_id_col(fused_df)
    if fused_id_col is None and not fused_df.empty:
        warnings.append("Could not detect fused ID column.")

    dataset_infos = []
    for path in CONTEXT_PAYLOAD.get("datasets", []):
        df = safe_read_csv(path, warnings)
        dataset_infos.append(build_dataset_lookup(df, source_dataset_name(path), warnings))

    testset_path = CONTEXT_PAYLOAD.get("fusion_testset")
    test_df = safe_read_csv(testset_path, warnings) if testset_path else pd.DataFrame()
    test_id_col, test_df, test_indexes = build_testset_index(test_df, review_attributes, warnings)

    wide_rows = []
    lineage_rows = []
    diff_rows = []

    mandatory_columns = []
    fused_identifier_column_name = "fused_id"
    mandatory_prefix = [fused_identifier_column_name]
    if fused_id_col and fused_id_col != fused_identifier_column_name:
        mandatory_prefix.append(fused_id_col)
    for attr in review_attributes:
        mandatory_columns.extend([
            f"{attr}_test",
            f"{attr}_fused",
            f"{attr}_source_1",
            f"{attr}_source_2",
            f"{attr}_source_3",
        ])

    if fused_df.empty:
        warnings.append("Fused dataframe is empty; output files will be created with headers only.")
    else:
        for _, fused_row in fused_df.iterrows():
            fused_id = ""
            if fused_id_col and fused_id_col in fused_row.index:
                fused_id = stringify(fused_row.get(fused_id_col, ""))
            if fused_id == "" and "_id" in fused_row.index:
                fused_id = stringify(fused_row.get("_id", ""))
            if fused_id == "" and "id" in fused_row.index:
                fused_id = stringify(fused_row.get("id", ""))

            parsed_sources = parse_fusion_sources(fused_row.get("_fusion_sources", ""))
            matched_sources = match_source_rows(parsed_sources, dataset_infos, warnings)

            test_row = match_test_row(fused_row, review_attributes, test_indexes)

            wide = {col: "" for col in mandatory_prefix + mandatory_columns}
            wide["fused_id"] = fused_id
            if fused_id_col and fused_id_col in wide:
                wide[fused_id_col] = fused_id

            for attr in review_attributes:
                wide[f"{attr}_test"] = stringify(test_row.get(attr, "")) if test_row is not None and attr in test_row.index else ""
                wide[f"{attr}_fused"] = stringify(fused_row.get(attr, "")) if attr in fused_row.index else ""
                for i in range(3):
                    val = ""
                    if i < len(matched_sources):
                        src_row = matched_sources[i]["source_row"]
                        if src_row is not None and hasattr(src_row, "index") and attr in src_row.index:
                            val = stringify(src_row.get(attr, ""))
                    wide[f"{attr}_source_{i+1}"] = val

            wide_rows.append(wide)

            for src in matched_sources:
                lineage_row = {
                    "fused_id": fused_id,
                    "source_id": stringify(src.get("source_id", "")),
                    "source_dataset": stringify(src.get("source_dataset", ""))
                }
                src_row = src.get("source_row", pd.Series(dtype=object))
                for attr in review_attributes:
                    lineage_row[f"source__{attr}"] = stringify(src_row.get(attr, "")) if hasattr(src_row, "index") and attr in src_row.index else ""
                    lineage_row[f"fused__{attr}"] = stringify(fused_row.get(attr, "")) if attr in fused_row.index else ""
                lineage_rows.append(lineage_row)

            if test_row is not None:
                overlap = [c for c in review_attributes if c in fused_row.index and c in test_row.index]
                for attr in overlap:
                    fused_val = stringify(fused_row.get(attr, ""))
                    test_val = stringify(test_row.get(attr, ""))
                    if normalize_text(fused_val) != normalize_text(test_val):
                        diff_rows.append({
                            "fused_id": fused_id,
                            "attribute": attr,
                            "fused_value": fused_val,
                            "testset_value": test_val,
                            "is_diff": True
                        })

    fused_review_df = pd.DataFrame(wide_rows)
    for col in mandatory_prefix + mandatory_columns:
        if col not in fused_review_df.columns:
            fused_review_df[col] = ""
    fused_review_df = fused_review_df[mandatory_prefix + mandatory_columns]
    fused_review_df.to_csv(fused_review_csv, index=False)

    lineage_columns = ["fused_id", "source_id", "source_dataset"]
    for attr in review_attributes:
        lineage_columns.append(f"source__{attr}")
    for attr in review_attributes:
        lineage_columns.append(f"fused__{attr}")
    lineage_df = pd.DataFrame(lineage_rows)
    for col in lineage_columns:
        if col not in lineage_df.columns:
            lineage_df[col] = ""
    lineage_df = lineage_df[lineage_columns]
    lineage_df.to_csv(source_lineage_csv, index=False)

    diff_columns = ["fused_id", "attribute", "fused_value", "testset_value", "is_diff"]
    diff_df = pd.DataFrame(diff_rows)
    for col in diff_columns:
        if col not in diff_df.columns:
            diff_df[col] = ""
    diff_df = diff_df[diff_columns]
    diff_df.to_csv(diff_csv, index=False)

    summary = {
        "summary": {
            "status": "completed",
            "message": "Human-review outputs generated." if not fused_df.empty else "Human-review outputs generated with empty fused input."
        },
        "file_paths": {
            "fused_review_table": fused_review_csv,
            "source_lineage_long": source_lineage_csv,
            "fusion_vs_testset_diff": diff_csv,
            "human_review_summary_json": summary_json,
            "human_review_summary_md": summary_md
        },
        "counts": {
            "fused_rows": int(len(fused_df)),
            "fused_review_rows": int(len(fused_review_df)),
            "source_lineage_rows": int(len(lineage_df)),
            "diff_rows": int(len(diff_df)),
            "source_datasets": int(len(dataset_infos)),
            "review_attributes": int(len(review_attributes)),
            "testset_rows": int(len(test_df)) if test_df is not None else 0
        },
        "warnings": warnings,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    if test_df is None or test_df.empty:
        summary["summary"]["note"] = "Fusion testset unavailable; created empty diff CSV."
    elif diff_df.empty:
        summary["summary"]["note"] = "Fusion testset available; no diffs recorded or no overlapping mismatches found."

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    md_lines = [
        "# Human Review Summary",
        "",
        f"- Status: {summary['summary'].get('status', '')}",
        f"- Message: {summary['summary'].get('message', '')}",
        f"- Created at: {summary['created_at']}",
        "",
        "## Counts",
    ]
    for k, v in summary["counts"].items():
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
        for w in warnings:
            md_lines.append(f"- {w}")
    else:
        md_lines.append("- None")
    if "note" in summary["summary"]:
        md_lines.extend([
            "",
            "## Note",
            f"- {summary['summary']['note']}"
        ])

    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))


if __name__ == "__main__":
    main()