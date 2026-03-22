import os
import json
import ast
from datetime import datetime, timezone

import pandas as pd


CONTEXT_PAYLOAD = {
    "datasets": [
        "output/runs/20260322_155058_books/normalization/attempt_3/amazon_small.csv",
        "output/runs/20260322_155058_books/normalization/attempt_3/goodreads_small.csv",
        "output/runs/20260322_155058_books/normalization/attempt_3/metabooks_small.csv"
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
        "overall_accuracy": 0.7125,
        "macro_accuracy": 0.7124999999999999,
        "num_evaluated_records": 10,
        "num_evaluated_attributes": 8,
        "total_evaluations": 80,
        "total_correct": 57,
        "publisher_accuracy": 0.6,
        "publisher_count": 10,
        "language_accuracy": 0.9,
        "language_count": 10,
        "publish_year_accuracy": 1.0,
        "publish_year_count": 10,
        "title_accuracy": 0.8,
        "title_count": 10,
        "author_accuracy": 0.8,
        "author_count": 10,
        "genres_accuracy": 0.1,
        "genres_count": 10,
        "page_count_accuracy": 0.5,
        "page_count_count": 10,
        "isbn_clean_accuracy": 1.0,
        "isbn_clean_count": 10
    },
    "auto_diagnostics": {
        "debug_reason_counts": {
            "mismatch": 20,
            "missing_fused_value": 3
        },
        "debug_reason_ratios": {
            "mismatch": 0.8695652173913043,
            "missing_fused_value": 0.13043478260869565
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
        "overall_accuracy": 0.7125
    },
    "integration_diagnostics_report": {},
    "output_paths": {
        "fused_csv": "output/runs/20260322_155058_books/data_fusion/fusion_data.csv",
        "fused_debug_jsonl": "output/runs/20260322_155058_books/data_fusion/debug_fusion_data.jsonl",
        "evaluation_json": "output/runs/20260322_155058_books/pipeline_evaluation/pipeline_evaluation.json",
        "review_summary_json": "output/runs/20260322_155058_books/human_review/human_review_summary.json",
        "review_summary_md": "output/runs/20260322_155058_books/human_review/human_review_summary.md",
        "fused_review_csv": "output/runs/20260322_155058_books/human_review/fused_review_table.csv",
        "source_lineage_csv": "output/runs/20260322_155058_books/human_review/source_lineage_long.csv",
        "diff_csv": "output/runs/20260322_155058_books/human_review/fusion_vs_testset_diff.csv"
    }
}


def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def normalize_missing(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value)
    if text.strip().lower() in {"nan", "none", "null"}:
        return ""
    return text


def read_csv_safe(path, warnings, label):
    if not path or not os.path.exists(path):
        warnings.append(f"Missing {label}: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as e:
        warnings.append(f"Failed to read {label} at {path}: {e}")
        return pd.DataFrame()


def detect_id_columns(df):
    if df is None or df.empty:
        return []
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    priority = ["_id", "id", "eval_id", "entity_id", "record_id", "source_id", "pk", "uuid"]
    found = []
    for p in priority:
        if p in lower_map:
            found.append(lower_map[p])
    for c in cols:
        lc = c.lower()
        if c not in found and (lc == "id" or lc.endswith("_id") or lc.startswith("id_")):
            found.append(c)
    return found


def choose_primary_id_column(df):
    ids = detect_id_columns(df)
    return ids[0] if ids else None


def parse_fusion_sources(value):
    if value is None:
        return []
    if isinstance(value, list):
        raw_items = value
    else:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null", "[]", "{}"}:
            return []
        raw_items = None
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if isinstance(parsed, list):
                    raw_items = parsed
                    break
                if isinstance(parsed, dict):
                    raw_items = [parsed]
                    break
                if isinstance(parsed, str):
                    raw_items = [parsed]
                    break
            except Exception:
                continue
        if raw_items is None:
            if "|" in text:
                raw_items = [x.strip() for x in text.split("|") if x.strip()]
            elif ";" in text:
                raw_items = [x.strip() for x in text.split(";") if x.strip()]
            elif "," in text:
                raw_items = [x.strip() for x in text.split(",") if x.strip()]
            else:
                raw_items = [text]
    parsed_items = []
    for item in raw_items:
        if isinstance(item, dict):
            parsed_items.append(item)
        else:
            parsed_items.append({"raw": normalize_missing(item)})
    return parsed_items


def source_candidates_from_item(item):
    candidates = []
    if not isinstance(item, dict):
        raw = normalize_missing(item)
        if raw:
            candidates.append({"dataset": "", "id": raw})
        return candidates

    dataset_keys = ["dataset", "source_dataset", "table", "file", "path", "dataset_name"]
    id_keys = ["id", "_id", "source_id", "record_id", "entity_id", "pk", "row_id"]

    dataset_vals = []
    id_vals = []

    for k, v in item.items():
        if k in dataset_keys:
            vv = normalize_missing(v)
            if vv:
                dataset_vals.append(vv)
        if k in id_keys or k.lower() in id_keys:
            vv = normalize_missing(v)
            if vv:
                id_vals.append(vv)

    raw = normalize_missing(item.get("raw", ""))
    if raw:
        for sep in ["::", "|", "@", "#"]:
            if sep in raw:
                left, right = raw.split(sep, 1)
                left = left.strip()
                right = right.strip()
                if left:
                    dataset_vals.append(left)
                if right:
                    id_vals.append(right)
                break
        if not id_vals:
            id_vals.append(raw)

    dataset_vals = list(dict.fromkeys(dataset_vals))
    id_vals = list(dict.fromkeys(id_vals))

    if dataset_vals and id_vals:
        for d in dataset_vals:
            for i in id_vals:
                candidates.append({"dataset": d, "id": i})
    elif id_vals:
        for i in id_vals:
            candidates.append({"dataset": "", "id": i})
    elif dataset_vals:
        for d in dataset_vals:
            candidates.append({"dataset": d, "id": ""})
    return candidates


def dataset_aliases(path):
    aliases = set()
    if not path:
        return aliases
    aliases.add(path)
    aliases.add(os.path.abspath(path))
    base = os.path.basename(path)
    aliases.add(base)
    stem, _ = os.path.splitext(base)
    aliases.add(stem)
    return {a for a in aliases if a}


def find_matching_source_row(candidates, dataset_infos):
    for cand in candidates:
        cand_dataset = normalize_missing(cand.get("dataset", ""))
        cand_id = normalize_missing(cand.get("id", ""))
        if not cand_id:
            continue
        for info in dataset_infos:
            if cand_dataset:
                if cand_dataset not in info["aliases"]:
                    continue
            row = info["id_index"].get(cand_id)
            if row is not None:
                return {
                    "source_dataset": info["path"],
                    "source_dataset_label": os.path.basename(info["path"]),
                    "source_id": cand_id,
                    "row": row
                }
    for cand in candidates:
        cand_id = normalize_missing(cand.get("id", ""))
        if not cand_id:
            continue
        matches = []
        for info in dataset_infos:
            row = info["id_index"].get(cand_id)
            if row is not None:
                matches.append((info, row))
        if len(matches) == 1:
            info, row = matches[0]
            return {
                "source_dataset": info["path"],
                "source_dataset_label": os.path.basename(info["path"]),
                "source_id": cand_id,
                "row": row
            }
    return None


def row_get(row, col):
    if row is None or not isinstance(row, dict):
        return ""
    return normalize_missing(row.get(col, ""))


def build_testset_lookup(df, review_attributes, warnings):
    if df is None or df.empty:
        return {}, None
    candidate_ids = detect_id_columns(df)
    preferred = [c for c in review_attributes if c in df.columns and c in candidate_ids]
    if preferred:
        id_col = preferred[0]
    else:
        id_col = candidate_ids[0] if candidate_ids else None
    if not id_col:
        warnings.append("Could not detect ID column in fusion testset.")
        return {}, None
    lookup = {}
    for _, rec in df.iterrows():
        key = normalize_missing(rec.get(id_col, ""))
        if key and key not in lookup:
            lookup[key] = {c: normalize_missing(rec.get(c, "")) for c in df.columns}
    return lookup, id_col


def main():
    warnings = []
    created_at = datetime.now(timezone.utc).isoformat()

    output_paths = CONTEXT_PAYLOAD.get("output_paths", {})
    review_attributes = CONTEXT_PAYLOAD.get("review_attributes", [])

    fused_csv_path = output_paths.get("fused_csv") or "output/data_fusion/fusion_data.csv"
    if not os.path.exists(fused_csv_path):
        fallback = "output/data_fusion/fusion_data.csv"
        if os.path.exists(fallback):
            fused_csv_path = fallback

    fused_review_csv = output_paths.get("fused_review_csv", "output/human_review/fused_review_table.csv")
    source_lineage_csv = output_paths.get("source_lineage_csv", "output/human_review/source_lineage_long.csv")
    diff_csv = output_paths.get("diff_csv", "output/human_review/fusion_vs_testset_diff.csv")
    summary_json_path = output_paths.get("review_summary_json", "output/human_review/human_review_summary.json")
    summary_md_path = output_paths.get("review_summary_md", "output/human_review/human_review_summary.md")

    for p in [fused_review_csv, source_lineage_csv, diff_csv, summary_json_path, summary_md_path]:
        ensure_dir(p)

    fused_df = read_csv_safe(fused_csv_path, warnings, "fused output")
    fused_id_col = choose_primary_id_column(fused_df) if not fused_df.empty else None
    if fused_df.empty:
        warnings.append("Fused output is empty or unavailable.")
    if not fused_id_col and not fused_df.empty:
        warnings.append("Could not detect primary ID column in fused output.")

    dataset_infos = []
    for path in CONTEXT_PAYLOAD.get("datasets", []):
        df = read_csv_safe(path, warnings, f"source dataset {path}")
        id_col = choose_primary_id_column(df) if not df.empty else None
        if not df.empty and not id_col:
            warnings.append(f"Could not detect ID column for source dataset: {path}")
        id_index = {}
        if not df.empty and id_col:
            for _, rec in df.iterrows():
                key = normalize_missing(rec.get(id_col, ""))
                if key and key not in id_index:
                    id_index[key] = {c: normalize_missing(rec.get(c, "")) for c in df.columns}
        dataset_infos.append({
            "path": path,
            "df": df,
            "id_col": id_col,
            "id_index": id_index,
            "aliases": dataset_aliases(path)
        })

    testset_path = CONTEXT_PAYLOAD.get("fusion_testset")
    testset_df = read_csv_safe(testset_path, warnings, "fusion testset") if testset_path else pd.DataFrame()
    testset_lookup, testset_id_col = build_testset_lookup(testset_df, review_attributes, warnings)

    fused_rows_output = []
    lineage_rows = []
    diff_rows = []

    mandatory_columns = []
    for attr in review_attributes:
        mandatory_columns.extend([
            f"{attr}_test",
            f"{attr}_fused",
            f"{attr}_source_1",
            f"{attr}_source_2",
            f"{attr}_source_3"
        ])

    fused_base_cols = []
    if not fused_df.empty:
        if fused_id_col:
            fused_base_cols.append("fused_id")
        if "_fusion_sources" in fused_df.columns:
            fused_base_cols.append("_fusion_sources")

    for _, fused_rec in fused_df.iterrows():
        fused_map = {c: normalize_missing(fused_rec.get(c, "")) for c in fused_df.columns}
        fused_id = normalize_missing(fused_rec.get(fused_id_col, "")) if fused_id_col else ""
        source_items = parse_fusion_sources(fused_map.get("_fusion_sources", ""))
        resolved_sources = []
        seen_pairs = set()

        for item in source_items:
            candidates = source_candidates_from_item(item)
            match = find_matching_source_row(candidates, dataset_infos)
            if match:
                pair = (match["source_dataset"], match["source_id"])
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    resolved_sources.append(match)

        row_out = {}
        if "fused_id" in fused_base_cols:
            row_out["fused_id"] = fused_id
        if "_fusion_sources" in fused_base_cols:
            row_out["_fusion_sources"] = fused_map.get("_fusion_sources", "")

        test_row = testset_lookup.get(fused_id, {}) if fused_id else {}

        for attr in review_attributes:
            row_out[f"{attr}_test"] = row_get(test_row, attr)
            row_out[f"{attr}_fused"] = row_get(fused_map, attr)
            for i in range(3):
                source_val = ""
                if i < len(resolved_sources):
                    source_val = row_get(resolved_sources[i]["row"], attr)
                row_out[f"{attr}_source_{i+1}"] = source_val

        fused_rows_output.append(row_out)

        for src in resolved_sources:
            lineage_row = {
                "fused_id": fused_id,
                "source_id": src["source_id"],
                "source_dataset": src["source_dataset"]
            }
            for attr in review_attributes:
                lineage_row[f"source_{attr}"] = row_get(src["row"], attr)
                lineage_row[f"fused_{attr}"] = row_get(fused_map, attr)
            lineage_rows.append(lineage_row)

        if testset_lookup:
            overlapping = [c for c in review_attributes if c in fused_df.columns and c in testset_df.columns]
            for attr in overlapping:
                fused_val = row_get(fused_map, attr)
                test_val = row_get(test_row, attr)
                if fused_id and (fused_val != test_val):
                    diff_rows.append({
                        "fused_id": fused_id,
                        "attribute": attr,
                        "fused_value": fused_val,
                        "testset_value": test_val,
                        "match": False
                    })

    fused_review_columns = list(dict.fromkeys(fused_base_cols + mandatory_columns))
    fused_review_df = pd.DataFrame(fused_rows_output)
    if fused_review_df.empty:
        fused_review_df = pd.DataFrame(columns=fused_review_columns)
    else:
        for col in fused_review_columns:
            if col not in fused_review_df.columns:
                fused_review_df[col] = ""
        fused_review_df = fused_review_df[fused_review_columns]

    lineage_columns = ["fused_id", "source_id", "source_dataset"]
    for attr in review_attributes:
        lineage_columns.append(f"source_{attr}")
    for attr in review_attributes:
        lineage_columns.append(f"fused_{attr}")
    lineage_df = pd.DataFrame(lineage_rows)
    if lineage_df.empty:
        lineage_df = pd.DataFrame(columns=lineage_columns)
    else:
        for col in lineage_columns:
            if col not in lineage_df.columns:
                lineage_df[col] = ""
        lineage_df = lineage_df[lineage_columns]

    diff_columns = ["fused_id", "attribute", "fused_value", "testset_value", "match"]
    diff_df = pd.DataFrame(diff_rows)
    if diff_df.empty:
        diff_df = pd.DataFrame(columns=diff_columns)
        if not testset_lookup:
            warnings.append("Fusion testset unavailable or unusable; created empty diff CSV.")
    else:
        for col in diff_columns:
            if col not in diff_df.columns:
                diff_df[col] = ""
        diff_df = diff_df[diff_columns]

    fused_review_df.to_csv(fused_review_csv, index=False)
    lineage_df.to_csv(source_lineage_csv, index=False)
    diff_df.to_csv(diff_csv, index=False)

    counts = {
        "fused_rows": int(len(fused_df)),
        "review_rows": int(len(fused_review_df)),
        "lineage_rows": int(len(lineage_df)),
        "diff_rows": int(len(diff_df)),
        "source_datasets": int(len(CONTEXT_PAYLOAD.get("datasets", []))),
        "review_attributes": int(len(review_attributes)),
        "testset_rows": int(len(testset_df)) if not testset_df.empty else 0
    }

    summary_text = "Human-review outputs created."
    if fused_df.empty:
        summary_text = "Human-review outputs created with limited content because fused data was unavailable."

    summary_obj = {
        "summary": summary_text,
        "file_paths": {
            "fused_review_table": fused_review_csv,
            "source_lineage_long": source_lineage_csv,
            "fusion_vs_testset_diff": diff_csv,
            "human_review_summary_json": summary_json_path,
            "human_review_summary_md": summary_md_path
        },
        "counts": counts,
        "warnings": warnings,
        "created_at": created_at
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_obj, f, indent=2, ensure_ascii=False)

    md_lines = [
        "# Human Review Summary",
        "",
        f"**Created at:** {created_at}",
        "",
        f"**Summary:** {summary_text}",
        "",
        "## Files",
        "",
        f"- Fused review table: `{fused_review_csv}`",
        f"- Source lineage long: `{source_lineage_csv}`",
        f"- Fusion vs testset diff: `{diff_csv}`",
        f"- Summary JSON: `{summary_json_path}`",
        "",
        "## Counts",
        ""
    ]
    for k, v in counts.items():
        md_lines.append(f"- {k}: {v}")
    md_lines.extend(["", "## Warnings", ""])
    if warnings:
        for w in warnings:
            md_lines.append(f"- {w}")
    else:
        md_lines.append("- None")
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))


if __name__ == "__main__":
    main()