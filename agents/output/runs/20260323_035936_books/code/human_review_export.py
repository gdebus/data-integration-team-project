import os
import json
import ast
from datetime import datetime, timezone

import pandas as pd


CONTEXT_PAYLOAD = {
    "datasets": [
        "output/runs/20260323_035936_books/normalization/attempt_2/amazon_small.csv",
        "output/runs/20260323_035936_books/normalization/attempt_2/goodreads_small.csv",
        "output/runs/20260323_035936_books/normalization/attempt_2/metabooks_small.csv"
    ],
    "fusion_testset": "input/datasets/books/testsets/test_set.csv",
    "review_attributes": [
        "_id",
        "author",
        "bookformat",
        "edition",
        "genres",
        "id",
        "isbn_clean",
        "language",
        "numratings",
        "page_count",
        "price",
        "publish_year",
        "publisher",
        "rating",
        "title",
        "eval_id"
    ],
    "evaluation_metrics": {
        "overall_accuracy": 0.9125,
        "macro_accuracy": 0.9125,
        "num_evaluated_records": 10,
        "num_evaluated_attributes": 8,
        "total_evaluations": 80,
        "total_correct": 73,
        "language_accuracy": 1.0,
        "language_count": 10,
        "publisher_accuracy": 0.9,
        "publisher_count": 10,
        "author_accuracy": 0.9,
        "author_count": 10,
        "publish_year_accuracy": 1.0,
        "publish_year_count": 10,
        "genres_accuracy": 0.9,
        "genres_count": 10,
        "isbn_clean_accuracy": 1.0,
        "isbn_clean_count": 10,
        "title_accuracy": 0.8,
        "title_count": 10,
        "page_count_accuracy": 0.8,
        "page_count_count": 10
    },
    "auto_diagnostics": {
        "debug_reason_counts": {
            "mismatch": 7
        },
        "debug_reason_ratios": {
            "mismatch": 1.0
        },
        "evaluation_stage": "validation",
        "evaluation_testset_path": "input/datasets/books/testsets/validation_set.csv",
        "id_alignment": {
            "gold_count": 10,
            "direct_coverage": 10,
            "direct_coverage_ratio": 1.0,
            "mapped_coverage": 10,
            "mapped_coverage_ratio": 1.0,
            "missing_gold_ids": []
        },
        "overall_accuracy": 0.9125
    },
    "integration_diagnostics_report": {},
    "output_paths": {
        "fused_csv": "output/runs/20260323_035936_books/data_fusion/fusion_data.csv",
        "fused_debug_jsonl": "output/runs/20260323_035936_books/data_fusion/debug_fusion_data.jsonl",
        "evaluation_json": "output/runs/20260323_035936_books/pipeline_evaluation/pipeline_evaluation.json",
        "review_summary_json": "output/runs/20260323_035936_books/human_review/human_review_summary.json",
        "review_summary_md": "output/runs/20260323_035936_books/human_review/human_review_summary.md",
        "fused_review_csv": "output/runs/20260323_035936_books/human_review/fused_review_table.csv",
        "source_lineage_csv": "output/runs/20260323_035936_books/human_review/source_lineage_long.csv",
        "diff_csv": "output/runs/20260323_035936_books/human_review/fusion_vs_testset_diff.csv"
    }
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_read_csv(path, warnings):
    if not path or not os.path.exists(path):
        warnings.append(f"Missing CSV file: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=object, keep_default_na=False)
    except Exception as e:
        warnings.append(f"Failed reading CSV {path}: {e}")
        return pd.DataFrame()


def normalize_str(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def detect_id_columns(df):
    if df is None or df.empty:
        return []
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    preferred = []
    exact_priority = [
        "_id", "id", "eval_id", "record_id", "entity_id", "source_id", "fused_id",
        "cluster_id", "canonical_id", "gold_id"
    ]
    for c in exact_priority:
        if c in cols:
            preferred.append(c)
        elif c.lower() in lower_map:
            preferred.append(lower_map[c.lower()])
    token_candidates = []
    for c in cols:
        cl = c.lower()
        if cl.endswith("_id") or cl == "id" or cl == "_id" or " id" in cl or "identifier" in cl:
            if c not in preferred:
                token_candidates.append(c)
    uniqueness_scored = []
    n = len(df)
    for c in cols:
        try:
            non_empty = df[c].astype(str).str.strip()
            non_empty = non_empty[non_empty != ""]
            uniq_ratio = (non_empty.nunique(dropna=True) / max(1, len(non_empty))) if len(non_empty) else 0
            if uniq_ratio >= 0.8:
                uniqueness_scored.append((c, uniq_ratio))
        except Exception:
            continue
    uniqueness_scored.sort(key=lambda x: (-x[1], x[0]))
    ordered = []
    for c in preferred + token_candidates + [c for c, _ in uniqueness_scored]:
        if c not in ordered:
            ordered.append(c)
    return ordered


def choose_primary_id_column(df):
    candidates = detect_id_columns(df)
    return candidates[0] if candidates else None


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
    if not s:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(s)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
            if isinstance(parsed, str):
                return [parsed]
        except Exception:
            pass
    if "|" in s:
        return [p.strip() for p in s.split("|") if p.strip()]
    if ";" in s:
        return [p.strip() for p in s.split(";") if p.strip()]
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]
    return [s]


def extract_source_tokens(item):
    result = {
        "raw": item,
        "source_id": "",
        "source_dataset": "",
        "row_index": None,
    }
    if isinstance(item, dict):
        for k in ["source_id", "id", "_id", "record_id"]:
            if k in item and normalize_str(item.get(k)):
                result["source_id"] = normalize_str(item.get(k))
                break
        for k in ["source_dataset", "dataset", "source", "table", "file", "dataset_name"]:
            if k in item and normalize_str(item.get(k)):
                result["source_dataset"] = normalize_str(item.get(k))
                break
        for k in ["row_index", "index", "row", "source_row_index"]:
            if k in item and normalize_str(item.get(k)):
                try:
                    result["row_index"] = int(float(str(item.get(k))))
                except Exception:
                    result["row_index"] = None
                break
        return result

    text = normalize_str(item)
    if not text:
        return result

    if "::" in text:
        left, right = text.split("::", 1)
        result["source_dataset"] = normalize_str(left)
        result["source_id"] = normalize_str(right)
        return result
    if ":" in text:
        left, right = text.split(":", 1)
        if right.strip():
            result["source_dataset"] = normalize_str(left)
            result["source_id"] = normalize_str(right)
            return result
    result["source_id"] = text
    return result


def basename_no_ext(path):
    return os.path.splitext(os.path.basename(path))[0]


def build_dataset_indexes(dataset_paths, warnings):
    dataset_info = []
    for path in dataset_paths:
        df = safe_read_csv(path, warnings)
        dataset_name = basename_no_ext(path)
        id_candidates = detect_id_columns(df)
        primary_id = id_candidates[0] if id_candidates else None
        by_id = {}
        if not df.empty and primary_id in df.columns:
            try:
                for idx, row in df.iterrows():
                    val = normalize_str(row.get(primary_id, ""))
                    if val and val not in by_id:
                        by_id[val] = idx
            except Exception as e:
                warnings.append(f"Failed building ID index for {path}: {e}")
        dataset_info.append({
            "path": path,
            "dataset_name": dataset_name,
            "df": df,
            "id_candidates": id_candidates,
            "primary_id": primary_id,
            "by_id": by_id,
        })
    return dataset_info


def match_dataset(token_dataset, dataset_info):
    token_dataset_norm = normalize_str(token_dataset).lower()
    if not token_dataset_norm:
        return None
    for info in dataset_info:
        options = {
            normalize_str(info["dataset_name"]).lower(),
            normalize_str(info["path"]).lower(),
            normalize_str(os.path.basename(info["path"])).lower(),
        }
        if token_dataset_norm in options:
            return info
    for info in dataset_info:
        if token_dataset_norm and token_dataset_norm in normalize_str(info["path"]).lower():
            return info
    return None


def resolve_source_row(token, dataset_info):
    source_id = normalize_str(token.get("source_id", ""))
    row_index = token.get("row_index", None)
    token_dataset = token.get("source_dataset", "")

    candidate_datasets = []
    matched = match_dataset(token_dataset, dataset_info) if token_dataset else None
    if matched is not None:
        candidate_datasets = [matched]
    else:
        candidate_datasets = list(dataset_info)

    for info in candidate_datasets:
        df = info["df"]
        if df is None or df.empty:
            continue
        if source_id and source_id in info["by_id"]:
            idx = info["by_id"][source_id]
            return info, idx
        for col in info["id_candidates"]:
            if col in df.columns and source_id:
                try:
                    matches = df.index[df[col].astype(str).str.strip() == source_id].tolist()
                    if matches:
                        return info, matches[0]
                except Exception:
                    pass
        if row_index is not None and 0 <= row_index < len(df):
            return info, row_index
    return None, None


def choose_fused_id_column(df):
    candidates = detect_id_columns(df)
    preferred_order = ["_id", "id", "eval_id", "fused_id", "entity_id", "cluster_id"]
    lower_map = {c.lower(): c for c in candidates}
    for p in preferred_order:
        if p in candidates:
            return p
        if p.lower() in lower_map:
            return lower_map[p.lower()]
    return candidates[0] if candidates else None


def find_testset_match_columns(fused_df, test_df):
    fused_ids = detect_id_columns(fused_df)
    test_ids = detect_id_columns(test_df)
    best = (None, None, 0)
    for fc in fused_ids:
        try:
            fvals = set(fused_df[fc].astype(str).str.strip())
        except Exception:
            continue
        for tc in test_ids:
            try:
                tvals = set(test_df[tc].astype(str).str.strip())
            except Exception:
                continue
            overlap = len({v for v in fvals.intersection(tvals) if v})
            if overlap > best[2]:
                best = (fc, tc, overlap)
    return best


def make_wide_review_table(fused_df, test_df, review_attributes, dataset_info, warnings):
    fused_id_col = choose_fused_id_column(fused_df)
    test_fused_col = None
    test_id_col = None
    if not test_df.empty and fused_id_col:
        fc, tc, overlap = find_testset_match_columns(fused_df, test_df)
        if overlap > 0:
            test_fused_col = fc
            test_id_col = tc
        else:
            warnings.append("Could not find overlapping ID columns between fused data and testset.")
    test_lookup = {}
    if test_fused_col and test_id_col:
        try:
            for _, row in test_df.iterrows():
                test_lookup[normalize_str(row.get(test_id_col, ""))] = row
        except Exception as e:
            warnings.append(f"Failed building testset lookup: {e}")

    rows = []
    for idx, fused_row in fused_df.iterrows():
        out = {}
        fused_id_value = normalize_str(fused_row.get(fused_id_col, "")) if fused_id_col else str(idx)
        out["fused_id"] = fused_id_value

        src_items = parse_fusion_sources(fused_row.get("_fusion_sources", ""))
        resolved_sources = []
        for item in src_items:
            token = extract_source_tokens(item)
            info, src_idx = resolve_source_row(token, dataset_info)
            if info is not None and src_idx is not None:
                try:
                    src_row = info["df"].iloc[src_idx]
                except Exception:
                    src_row = None
                resolved_sources.append((info, src_idx, src_row, token))
            else:
                resolved_sources.append((None, None, None, token))

        test_row = test_lookup.get(fused_id_value) if fused_id_value else None

        for attr in review_attributes:
            out[f"{attr}_test"] = normalize_str(test_row.get(attr, "")) if test_row is not None and attr in test_row.index else ""
            out[f"{attr}_fused"] = normalize_str(fused_row.get(attr, "")) if attr in fused_df.columns else ""
            for i in range(1, 4):
                colname = f"{attr}_source_{i}"
                if i <= len(resolved_sources):
                    _, _, src_row, _ = resolved_sources[i - 1]
                    if src_row is not None and attr in src_row.index:
                        out[colname] = normalize_str(src_row.get(attr, ""))
                    else:
                        out[colname] = ""
                else:
                    out[colname] = ""
        rows.append(out)

    columns = ["fused_id"]
    for attr in review_attributes:
        columns.extend([
            f"{attr}_test",
            f"{attr}_fused",
            f"{attr}_source_1",
            f"{attr}_source_2",
            f"{attr}_source_3",
        ])
    wide_df = pd.DataFrame(rows)
    for c in columns:
        if c not in wide_df.columns:
            wide_df[c] = ""
    wide_df = wide_df[columns]
    return wide_df


def make_lineage_long(fused_df, review_attributes, dataset_info):
    fused_id_col = choose_fused_id_column(fused_df)
    records = []
    for idx, fused_row in fused_df.iterrows():
        fused_id_value = normalize_str(fused_row.get(fused_id_col, "")) if fused_id_col else str(idx)
        src_items = parse_fusion_sources(fused_row.get("_fusion_sources", ""))
        if not src_items:
            base = {
                "fused_id": fused_id_value,
                "source_id": "",
                "source_dataset": "",
            }
            for attr in review_attributes:
                base[f"source__{attr}"] = ""
                base[f"fused__{attr}"] = normalize_str(fused_row.get(attr, "")) if attr in fused_df.columns else ""
            records.append(base)
            continue

        for item in src_items:
            token = extract_source_tokens(item)
            info, src_idx = resolve_source_row(token, dataset_info)
            src_row = None
            source_dataset = normalize_str(token.get("source_dataset", ""))
            source_id = normalize_str(token.get("source_id", ""))
            if info is not None and src_idx is not None:
                source_dataset = info["dataset_name"]
                try:
                    src_row = info["df"].iloc[src_idx]
                except Exception:
                    src_row = None
                if not source_id and info["primary_id"] and src_row is not None and info["primary_id"] in src_row.index:
                    source_id = normalize_str(src_row.get(info["primary_id"], ""))
            base = {
                "fused_id": fused_id_value,
                "source_id": source_id,
                "source_dataset": source_dataset,
            }
            for attr in review_attributes:
                base[f"source__{attr}"] = normalize_str(src_row.get(attr, "")) if src_row is not None and attr in src_row.index else ""
                base[f"fused__{attr}"] = normalize_str(fused_row.get(attr, "")) if attr in fused_df.columns else ""
            records.append(base)
    return pd.DataFrame(records)


def make_diff_table(fused_df, test_df, review_attributes, warnings):
    columns = ["fused_id", "testset_id", "matched_on_fused_column", "matched_on_test_column", "attribute", "fused_value", "test_value", "is_equal"]
    if test_df.empty or fused_df.empty:
        return pd.DataFrame(columns=columns), "Fusion testset unavailable or empty."

    fused_col, test_col, overlap = find_testset_match_columns(fused_df, test_df)
    if not fused_col or not test_col or overlap == 0:
        warnings.append("No overlapping ID columns found for fusion_vs_testset_diff.")
        return pd.DataFrame(columns=columns), "No overlapping ID columns found between fused data and testset."

    test_lookup = {}
    for _, row in test_df.iterrows():
        test_lookup[normalize_str(row.get(test_col, ""))] = row

    records = []
    for _, fused_row in fused_df.iterrows():
        fused_id = normalize_str(fused_row.get(fused_col, ""))
        if fused_id not in test_lookup:
            continue
        test_row = test_lookup[fused_id]
        overlapping_attrs = [a for a in review_attributes if a in fused_df.columns and a in test_df.columns]
        for attr in overlapping_attrs:
            fused_val = normalize_str(fused_row.get(attr, ""))
            test_val = normalize_str(test_row.get(attr, ""))
            equal = fused_val == test_val
            if not equal:
                records.append({
                    "fused_id": fused_id,
                    "testset_id": normalize_str(test_row.get(test_col, "")),
                    "matched_on_fused_column": fused_col,
                    "matched_on_test_column": test_col,
                    "attribute": attr,
                    "fused_value": fused_val,
                    "test_value": test_val,
                    "is_equal": equal,
                })
    return pd.DataFrame(records, columns=columns), f"Matched fused/testset using {fused_col} -> {test_col} with overlap {overlap}."


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_md(path, summary_json):
    lines = []
    lines.append("# Human Review Summary")
    lines.append("")
    lines.append(f"- Created at: {summary_json.get('created_at', '')}")
    lines.append(f"- Summary: {summary_json.get('summary', '')}")
    lines.append("")
    lines.append("## Counts")
    for k, v in summary_json.get("counts", {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Files")
    for k, v in summary_json.get("file_paths", {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Warnings")
    warnings = summary_json.get("warnings", [])
    if warnings:
        for w in warnings:
            lines.append(f"- {w}")
    else:
        lines.append("- None")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    warnings = []
    output_paths = CONTEXT_PAYLOAD.get("output_paths", {})
    review_attributes = CONTEXT_PAYLOAD.get("review_attributes", [])

    fused_csv_path = "output/data_fusion/fusion_data.csv"
    if not os.path.exists(fused_csv_path):
        alt = output_paths.get("fused_csv", "")
        if alt and os.path.exists(alt):
            fused_csv_path = alt

    fused_review_csv = output_paths.get("fused_review_csv", "output/human_review/fused_review_table.csv")
    source_lineage_csv = output_paths.get("source_lineage_csv", "output/human_review/source_lineage_long.csv")
    diff_csv = output_paths.get("diff_csv", "output/human_review/fusion_vs_testset_diff.csv")
    review_summary_json = output_paths.get("review_summary_json", "output/human_review/human_review_summary.json")
    review_summary_md = output_paths.get("review_summary_md", "output/human_review/human_review_summary.md")

    for p in [fused_review_csv, source_lineage_csv, diff_csv, review_summary_json, review_summary_md]:
        ensure_dir(os.path.dirname(p))

    fused_df = safe_read_csv(fused_csv_path, warnings)
    test_df = safe_read_csv(CONTEXT_PAYLOAD.get("fusion_testset", ""), warnings)
    dataset_info = build_dataset_indexes(CONTEXT_PAYLOAD.get("datasets", []), warnings)

    if fused_df.empty:
        warnings.append("Fused dataframe is empty; outputs will be created with headers only.")

    wide_df = make_wide_review_table(fused_df, test_df, review_attributes, dataset_info, warnings) if not fused_df.empty else pd.DataFrame(
        columns=["fused_id"] + [col for attr in review_attributes for col in [
            f"{attr}_test", f"{attr}_fused", f"{attr}_source_1", f"{attr}_source_2", f"{attr}_source_3"
        ]]
    )
    lineage_df = make_lineage_long(fused_df, review_attributes, dataset_info) if not fused_df.empty else pd.DataFrame(
        columns=["fused_id", "source_id", "source_dataset"] + [f"source__{a}" for a in review_attributes] + [f"fused__{a}" for a in review_attributes]
    )
    diff_df, diff_note = make_diff_table(fused_df, test_df, review_attributes, warnings)

    try:
        wide_df.to_csv(fused_review_csv, index=False)
    except Exception as e:
        warnings.append(f"Failed writing fused review table: {e}")
    try:
        lineage_df.to_csv(source_lineage_csv, index=False)
    except Exception as e:
        warnings.append(f"Failed writing source lineage table: {e}")
    try:
        if diff_df.empty:
            diff_df = pd.DataFrame(columns=["fused_id", "testset_id", "matched_on_fused_column", "matched_on_test_column", "attribute", "fused_value", "test_value", "is_equal"])
        diff_df.to_csv(diff_csv, index=False)
    except Exception as e:
        warnings.append(f"Failed writing diff table: {e}")

    created_at = datetime.now(timezone.utc).isoformat()
    summary_payload = {
        "summary": diff_note,
        "file_paths": {
            "fused_review_table_csv": fused_review_csv,
            "source_lineage_long_csv": source_lineage_csv,
            "fusion_vs_testset_diff_csv": diff_csv,
            "human_review_summary_json": review_summary_json,
            "human_review_summary_md": review_summary_md,
        },
        "counts": {
            "fused_rows": int(len(fused_df)),
            "review_rows": int(len(wide_df)),
            "lineage_rows": int(len(lineage_df)),
            "diff_rows": int(len(diff_df)),
            "source_datasets_loaded": int(len([d for d in dataset_info if not d["df"].empty])),
            "source_datasets_requested": int(len(dataset_info)),
            "review_attributes": int(len(review_attributes)),
        },
        "warnings": warnings,
        "created_at": created_at,
    }

    try:
        write_json(review_summary_json, summary_payload)
    except Exception:
        pass
    try:
        write_md(review_summary_md, summary_payload)
    except Exception:
        pass


if __name__ == "__main__":
    main()