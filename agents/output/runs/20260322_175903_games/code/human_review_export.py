import os
import re
import json
import ast
import csv
from datetime import datetime
from xml.etree import ElementTree as ET

import pandas as pd


CONTEXT_PAYLOAD = {
    "datasets": [
        "output/runs/20260322_175903_games/normalization/attempt_1/dbpedia.csv",
        "output/runs/20260322_175903_games/normalization/attempt_1/metacritic.csv",
        "output/runs/20260322_175903_games/normalization/attempt_1/sales.csv"
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
        "overall_accuracy": 0.7310924369747899,
        "macro_accuracy": 0.7273809523809525,
        "num_evaluated_records": 15,
        "num_evaluated_attributes": 8,
        "total_evaluations": 119,
        "total_correct": 87,
        "ESRB_accuracy": 0.6666666666666666,
        "ESRB_count": 15,
        "developer_accuracy": 0.6,
        "developer_count": 15,
        "name_accuracy": 0.9333333333333333,
        "name_count": 15,
        "releaseYear_accuracy": 1.0,
        "releaseYear_count": 15,
        "platform_accuracy": 0.8,
        "platform_count": 15,
        "publisher_accuracy": 0.8,
        "publisher_count": 15,
        "criticScore_accuracy": 0.7333333333333333,
        "criticScore_count": 15,
        "userScore_accuracy": 0.2857142857142857,
        "userScore_count": 14
    },
    "auto_diagnostics": {
        "debug_reason_counts": {
            "missing_fused_value": 28,
            "mismatch": 15
        },
        "debug_reason_ratios": {
            "missing_fused_value": 0.6511627906976745,
            "mismatch": 0.3488372093023256
        },
        "evaluation_stage": "validation",
        "evaluation_testset_path": "input/datasets/games/testsets/test_set_fusion.xml",
        "id_alignment": {
            "gold_count": 15,
            "direct_coverage": 0,
            "direct_coverage_ratio": 0.0,
            "mapped_coverage": 15,
            "mapped_coverage_ratio": 1.0,
            "missing_gold_ids": []
        },
        "overall_accuracy": 0.5865384615384616
    },
    "integration_diagnostics_report": {},
    "output_paths": {
        "fused_csv": "output/runs/20260322_175903_games/data_fusion/fusion_data.csv",
        "fused_debug_jsonl": "output/runs/20260322_175903_games/data_fusion/debug_fusion_data.jsonl",
        "evaluation_json": "output/runs/20260322_175903_games/pipeline_evaluation/pipeline_evaluation.json",
        "review_summary_json": "output/runs/20260322_175903_games/human_review/human_review_summary.json",
        "review_summary_md": "output/runs/20260322_175903_games/human_review/human_review_summary.md",
        "fused_review_csv": "output/runs/20260322_175903_games/human_review/fused_review_table.csv",
        "source_lineage_csv": "output/runs/20260322_175903_games/human_review/source_lineage_long.csv",
        "diff_csv": "output/runs/20260322_175903_games/human_review/fusion_vs_testset_diff.csv"
    }
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def safe_str(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def normalize_string(value):
    return safe_str(value).strip()


def lower_set(items):
    return {safe_str(x).strip().lower() for x in items}


def read_csv_flex(path, warnings):
    if not path or not os.path.exists(path):
        warnings.append(f"Missing CSV file: {path}")
        return pd.DataFrame()
    encodings = ["utf-8", "utf-8-sig", "latin1"]
    seps = [",", ";", "\t", "|"]
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(path, dtype=str, sep=sep, engine="python", keep_default_na=False)
                if df is not None and len(df.columns) > 0:
                    return df.fillna("")
            except Exception:
                continue
    try:
        df = pd.read_csv(path, dtype=str, engine="python", keep_default_na=False)
        return df.fillna("")
    except Exception as e:
        warnings.append(f"Failed to read CSV {path}: {e}")
        return pd.DataFrame()


def detect_id_column(df, preferred_names=None):
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    preferred_names = preferred_names or []
    candidates = preferred_names + [
        "_id", "id", "entity_id", "record_id", "eval_id", "fusion_id", "cluster_id", "pk", "uuid"
    ]
    col_map = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in col_map:
            return col_map[c.lower()]
    score = []
    for c in cols:
        lc = c.lower()
        s = 0
        if lc == "id":
            s += 100
        if lc.endswith("_id") or lc.startswith("id_") or "id" in lc:
            s += 50
        series = df[c].astype(str)
        non_empty = series[series.str.strip() != ""]
        unique_ratio = (non_empty.nunique(dropna=True) / max(len(non_empty), 1)) if len(non_empty) else 0
        s += unique_ratio * 25
        if len(non_empty) and non_empty.str.len().mean() < 80:
            s += 5
        score.append((s, c))
    score.sort(reverse=True)
    return score[0][1] if score else None


def parse_fusion_sources_cell(cell):
    if cell is None:
        return []
    if isinstance(cell, list):
        return [safe_str(x) for x in cell if safe_str(x) != ""]
    text = safe_str(cell).strip()
    if text == "":
        return []
    parsed = None
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            break
        except Exception:
            continue
    if isinstance(parsed, list):
        out = []
        for item in parsed:
            if isinstance(item, dict):
                candidate = item.get("id", item.get("_id", item.get("source_id", item.get("record_id", ""))))
                if safe_str(candidate):
                    out.append(safe_str(candidate))
            else:
                out.append(safe_str(item))
        return [x for x in out if x != ""]
    if isinstance(parsed, dict):
        out = []
        for key in ["ids", "source_ids", "sources", "records"]:
            val = parsed.get(key)
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        candidate = item.get("id", item.get("_id", item.get("source_id", item.get("record_id", ""))))
                        if safe_str(candidate):
                            out.append(safe_str(candidate))
                    else:
                        out.append(safe_str(item))
                return [x for x in out if x != ""]
        candidate = parsed.get("id", parsed.get("_id", parsed.get("source_id", parsed.get("record_id", ""))))
        return [safe_str(candidate)] if safe_str(candidate) else []
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if inner == "":
            return []
        parts = [p.strip().strip("'").strip('"') for p in inner.split(",")]
        return [p for p in parts if p]
    if "|" in text:
        return [p.strip() for p in text.split("|") if p.strip()]
    if ";" in text:
        return [p.strip() for p in text.split(";") if p.strip()]
    if "," in text:
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if len(parts) > 1:
            return parts
    return [text]


def dataset_name_from_path(path):
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name or base or "unknown_dataset"


def load_source_datasets(paths, warnings):
    datasets = []
    for path in paths:
        df = read_csv_flex(path, warnings)
        name = dataset_name_from_path(path)
        id_col = detect_id_column(df)
        index_by_id = {}
        if not df.empty and id_col and id_col in df.columns:
            for _, row in df.iterrows():
                rid = normalize_string(row.get(id_col, ""))
                if rid != "" and rid not in index_by_id:
                    index_by_id[rid] = row.to_dict()
        else:
            warnings.append(f"Could not robustly detect ID column for source dataset: {path}")
        datasets.append({
            "path": path,
            "name": name,
            "df": df,
            "id_col": id_col,
            "index_by_id": index_by_id
        })
    return datasets


def extract_possible_source_ids(row):
    possible = []
    for col, val in row.items():
        if safe_str(col).lower() == "_fusion_sources":
            possible.extend(parse_fusion_sources_cell(val))
        elif "source" in safe_str(col).lower() and "id" in safe_str(col).lower():
            possible.extend(parse_fusion_sources_cell(val))
    seen = set()
    out = []
    for x in possible:
        sx = safe_str(x).strip()
        if sx and sx not in seen:
            seen.add(sx)
            out.append(sx)
    return out


def map_source_ids_to_rows(source_ids, source_datasets):
    mapped = []
    for sid in source_ids:
        matched = False
        for ds in source_datasets:
            rec = ds["index_by_id"].get(sid)
            if rec is not None:
                mapped.append({
                    "source_id": sid,
                    "source_dataset": ds["name"],
                    "row": rec
                })
                matched = True
        if not matched:
            mapped.append({
                "source_id": sid,
                "source_dataset": "",
                "row": {}
            })
    return mapped


def try_parse_xml_testset(path, warnings):
    if not path or not os.path.exists(path):
        warnings.append(f"Missing fusion testset: {path}")
        return pd.DataFrame()
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except Exception as e:
        warnings.append(f"Failed to parse XML testset {path}: {e}")
        return pd.DataFrame()

    records = []

    def collect_record(elem):
        record = {}
        attrs = dict(elem.attrib) if elem.attrib else {}
        for k, v in attrs.items():
            record[k] = v
        child_counts = {}
        for child in list(elem):
            tag = child.tag.split("}")[-1]
            text = (child.text or "").strip()
            if list(child):
                nested = {}
                for gc in list(child):
                    gtag = gc.tag.split("}")[-1]
                    gtext = (gc.text or "").strip()
                    if gtag in nested:
                        if not isinstance(nested[gtag], list):
                            nested[gtag] = [nested[gtag]]
                        nested[gtag].append(gtext)
                    else:
                        nested[gtag] = gtext
                if tag in record:
                    child_counts[tag] = child_counts.get(tag, 1) + 1
                    record[f"{tag}_{child_counts[tag]}"] = json.dumps(nested, ensure_ascii=False)
                else:
                    record[tag] = json.dumps(nested, ensure_ascii=False)
            else:
                if tag in record:
                    child_counts[tag] = child_counts.get(tag, 1) + 1
                    record[f"{tag}_{child_counts[tag]}"] = text
                else:
                    record[tag] = text
        if record:
            records.append(record)

    for elem in root.iter():
        children = list(elem)
        if not children:
            continue
        leaf_children = [c for c in children if len(list(c)) == 0]
        ratio = len(leaf_children) / max(len(children), 1)
        if ratio >= 0.5:
            collect_record(elem)

    if not records:
        warnings.append(f"No record-like rows extracted from XML testset: {path}")
        return pd.DataFrame()

    df = pd.DataFrame(records).fillna("")
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def load_testset(path, warnings):
    if not path:
        return pd.DataFrame()
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return read_csv_flex(path, warnings)
    if ext == ".xml":
        return try_parse_xml_testset(path, warnings)
    warnings.append(f"Unsupported testset extension for {path}; attempting CSV read")
    return read_csv_flex(path, warnings)


def build_testset_index(test_df, fused_df, review_attributes, warnings):
    if test_df.empty:
        return {}, None
    preferred = ["eval_id", "_id", "id"]
    test_id_col = detect_id_column(test_df, preferred_names=preferred)
    fused_id_col = detect_id_column(fused_df, preferred_names=preferred)

    test_index = {}
    if test_id_col and test_id_col in test_df.columns:
        for _, row in test_df.iterrows():
            rid = normalize_string(row.get(test_id_col, ""))
            if rid != "":
                test_index[rid] = row.to_dict()

    if test_index:
        return test_index, test_id_col

    overlap = [c for c in review_attributes if c in test_df.columns]
    if overlap:
        for _, row in test_df.iterrows():
            for c in overlap:
                rid = normalize_string(row.get(c, ""))
                if rid != "":
                    test_index[rid] = row.to_dict()
        if test_index:
            warnings.append("Testset ID column was not confidently detected; built loose index from overlapping review attributes.")
            return test_index, None

    warnings.append("Could not build testset index robustly.")
    return {}, test_id_col or fused_id_col


def find_test_row_for_fused(fused_row, test_index, review_attributes):
    if not test_index:
        return {}
    candidates = []
    for key in ["eval_id", "_id", "id"]:
        val = normalize_string(fused_row.get(key, ""))
        if val:
            candidates.append(val)
    for attr in review_attributes:
        val = normalize_string(fused_row.get(attr, ""))
        if val:
            candidates.append(val)
    seen = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if c in test_index:
            return test_index[c]
    return {}


def create_fused_review_table(fused_df, source_datasets, test_index, review_attributes):
    rows = []
    for _, fused_row in fused_df.iterrows():
        fused_dict = fused_row.to_dict()
        source_ids = extract_possible_source_ids(fused_dict)
        mapped_sources = map_source_ids_to_rows(source_ids, source_datasets)

        out = {}
        fused_id_value = normalize_string(
            fused_dict.get("eval_id", fused_dict.get("_id", fused_dict.get("id", "")))
        )
        out["fused_entity_id"] = fused_id_value
        test_row = find_test_row_for_fused(fused_dict, test_index, review_attributes) if test_index else {}

        for attr in review_attributes:
            out[f"{attr}_test"] = normalize_string(test_row.get(attr, "")) if test_row else ""
            out[f"{attr}_fused"] = normalize_string(fused_dict.get(attr, ""))
            for i in range(1, 4):
                out[f"{attr}_source_{i}"] = ""
            for idx, src in enumerate(mapped_sources[:3], start=1):
                out[f"{attr}_source_{idx}"] = normalize_string(src.get("row", {}).get(attr, ""))
        rows.append(out)

    mandatory_columns = ["fused_entity_id"]
    for attr in review_attributes:
        mandatory_columns.extend([
            f"{attr}_test",
            f"{attr}_fused",
            f"{attr}_source_1",
            f"{attr}_source_2",
            f"{attr}_source_3",
        ])
    if rows:
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(columns=mandatory_columns)
    for c in mandatory_columns:
        if c not in df.columns:
            df[c] = ""
    return df[mandatory_columns]


def create_source_lineage_long(fused_df, source_datasets, review_attributes):
    rows = []
    for _, fused_row in fused_df.iterrows():
        fused_dict = fused_row.to_dict()
        fused_id_value = normalize_string(
            fused_dict.get("eval_id", fused_dict.get("_id", fused_dict.get("id", "")))
        )
        source_ids = extract_possible_source_ids(fused_dict)
        mapped_sources = map_source_ids_to_rows(source_ids, source_datasets)

        if not mapped_sources:
            base = {
                "fused_id": fused_id_value,
                "source_id": "",
                "source_dataset": "",
            }
            for attr in review_attributes:
                base[f"source_{attr}"] = ""
                base[f"fused_{attr}"] = normalize_string(fused_dict.get(attr, ""))
            rows.append(base)
            continue

        for src in mapped_sources:
            rec = src.get("row", {}) or {}
            base = {
                "fused_id": fused_id_value,
                "source_id": normalize_string(src.get("source_id", "")),
                "source_dataset": normalize_string(src.get("source_dataset", "")),
            }
            for attr in review_attributes:
                base[f"source_{attr}"] = normalize_string(rec.get(attr, ""))
                base[f"fused_{attr}"] = normalize_string(fused_dict.get(attr, ""))
            rows.append(base)

    columns = ["fused_id", "source_id", "source_dataset"] + \
              [f"source_{a}" for a in review_attributes] + \
              [f"fused_{a}" for a in review_attributes]
    if rows:
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(columns=columns)
    for c in columns:
        if c not in df.columns:
            df[c] = ""
    return df[columns]


def create_diff_table(fused_df, test_index, review_attributes):
    columns = ["fused_id", "test_id", "attribute", "fused_value", "test_value", "is_match"]
    rows = []
    if not test_index:
        return pd.DataFrame(columns=columns)
    for _, fused_row in fused_df.iterrows():
        fused_dict = fused_row.to_dict()
        test_row = find_test_row_for_fused(fused_dict, test_index, review_attributes)
        if not test_row:
            continue
        fused_id = normalize_string(fused_dict.get("eval_id", fused_dict.get("_id", fused_dict.get("id", ""))))
        test_id = normalize_string(test_row.get("eval_id", test_row.get("_id", test_row.get("id", ""))))
        for attr in review_attributes:
            fused_val = normalize_string(fused_dict.get(attr, ""))
            test_val = normalize_string(test_row.get(attr, ""))
            if fused_val == "" and test_val == "":
                continue
            rows.append({
                "fused_id": fused_id,
                "test_id": test_id,
                "attribute": attr,
                "fused_value": fused_val,
                "test_value": test_val,
                "is_match": fused_val == test_val
            })
    if rows:
        return pd.DataFrame(rows, columns=columns)
    return pd.DataFrame(columns=columns)


def write_csv(df, path):
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def write_json(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_text(text, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    warnings_list = []
    created_at = now_iso()

    output_paths = CONTEXT_PAYLOAD.get("output_paths", {})
    fused_csv_path = output_paths.get("fused_csv", "output/data_fusion/fusion_data.csv")
    if not os.path.exists(fused_csv_path):
        fallback = "output/data_fusion/fusion_data.csv"
        if os.path.exists(fallback):
            fused_csv_path = fallback
        else:
            warnings_list.append(f"Fused CSV not found at configured path or fallback: {fused_csv_path} / {fallback}")

    review_dir = os.path.dirname(output_paths.get("review_summary_json", "output/human_review/human_review_summary.json")) or "output/human_review"
    ensure_dir(review_dir)

    fused_review_csv = output_paths.get("fused_review_csv", os.path.join(review_dir, "fused_review_table.csv"))
    source_lineage_csv = output_paths.get("source_lineage_csv", os.path.join(review_dir, "source_lineage_long.csv"))
    diff_csv = output_paths.get("diff_csv", os.path.join(review_dir, "fusion_vs_testset_diff.csv"))
    review_summary_json = output_paths.get("review_summary_json", os.path.join(review_dir, "human_review_summary.json"))
    review_summary_md = output_paths.get("review_summary_md", os.path.join(review_dir, "human_review_summary.md"))

    review_attributes = CONTEXT_PAYLOAD.get("review_attributes", [])

    fused_df = read_csv_flex(fused_csv_path, warnings_list)
    if fused_df.empty:
        warnings_list.append("Fused dataframe is empty or unavailable; outputs will still be created.")

    source_datasets = load_source_datasets(CONTEXT_PAYLOAD.get("datasets", []), warnings_list)
    test_df = load_testset(CONTEXT_PAYLOAD.get("fusion_testset"), warnings_list)
    test_index, test_id_col = build_testset_index(test_df, fused_df, review_attributes, warnings_list)

    fused_review_df = create_fused_review_table(fused_df, source_datasets, test_index, review_attributes)
    source_lineage_df = create_source_lineage_long(fused_df, source_datasets, review_attributes)
    diff_df = create_diff_table(fused_df, test_index, review_attributes)

    write_csv(fused_review_df, fused_review_csv)
    write_csv(source_lineage_df, source_lineage_csv)
    write_csv(diff_df, diff_csv)

    counts = {
        "fused_rows": int(len(fused_df)),
        "review_rows": int(len(fused_review_df)),
        "source_lineage_rows": int(len(source_lineage_df)),
        "diff_rows": int(len(diff_df)),
        "source_datasets_loaded": int(sum(1 for ds in source_datasets if not ds["df"].empty)),
        "source_datasets_requested": int(len(source_datasets)),
        "testset_rows": int(len(test_df)),
        "review_attributes_count": int(len(review_attributes))
    }

    summary_text = "Human review outputs created."
    if test_df.empty:
        summary_text += " Testset unavailable or unreadable; diff file created empty if no matches."
    elif not test_index:
        summary_text += " Testset loaded but ID mapping was weak; diff coverage may be limited."

    summary = {
        "summary": summary_text,
        "file_paths": {
            "fused_review_table_csv": fused_review_csv,
            "source_lineage_long_csv": source_lineage_csv,
            "fusion_vs_testset_diff_csv": diff_csv,
            "human_review_summary_json": review_summary_json,
            "human_review_summary_md": review_summary_md
        },
        "counts": counts,
        "warnings": warnings_list,
        "created_at": created_at
    }
    write_json(summary, review_summary_json)

    md = []
    md.append("# Human Review Summary")
    md.append("")
    md.append(f"- Created at: {created_at}")
    md.append(f"- Summary: {summary_text}")
    md.append("")
    md.append("## Output Files")
    for k, v in summary["file_paths"].items():
        md.append(f"- **{k}**: `{v}`")
    md.append("")
    md.append("## Counts")
    for k, v in counts.items():
        md.append(f"- **{k}**: {v}")
    md.append("")
    md.append("## Warnings")
    if warnings_list:
        for w in warnings_list:
            md.append(f"- {w}")
    else:
        md.append("- None")
    write_text("\n".join(md) + "\n", review_summary_md)


if __name__ == "__main__":
    main()