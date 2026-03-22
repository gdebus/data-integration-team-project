import os
import re
import csv
import json
import ast
from datetime import datetime

import pandas as pd


CONTEXT_PAYLOAD = {
    "datasets": [
        "output/runs/20260322_171209_music/normalization/attempt_3/discogs.csv",
        "output/runs/20260322_171209_music/normalization/attempt_3/lastfm.csv",
        "output/runs/20260322_171209_music/normalization/attempt_3/musicbrainz.csv"
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
        "artist_accuracy": 0.9130434782608695,
        "artist_count": 23,
        "duration_accuracy": 0.8695652173913043,
        "duration_count": 23,
        "label_accuracy": 0.5625,
        "label_count": 16,
        "macro_accuracy": 0.7702294685990337,
        "name_accuracy": 0.9130434782608695,
        "name_count": 23,
        "num_evaluated_attributes": 9,
        "num_evaluated_records": 23,
        "overall_accuracy": 0.7817258883248731,
        "release-country_accuracy": 0.8695652173913043,
        "release-country_count": 23,
        "release-date_accuracy": 0.7391304347826086,
        "release-date_count": 23,
        "total_correct": 154,
        "total_evaluations": 197,
        "tracks_track_duration_accuracy": 0.5,
        "tracks_track_duration_count": 20,
        "tracks_track_name_accuracy": 0.6956521739130435,
        "tracks_track_name_count": 23,
        "tracks_track_position_accuracy": 0.8695652173913043,
        "tracks_track_position_count": 23
    },
    "auto_diagnostics": {
        "debug_reason_counts": {
            "mismatch": 34,
            "missing_fused_value": 10
        },
        "debug_reason_ratios": {
            "mismatch": 0.7727272727272727,
            "missing_fused_value": 0.22727272727272727
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
        "overall_accuracy": 0.7766497461928934
    },
    "integration_diagnostics_report": {},
    "output_paths": {
        "fused_csv": "output/runs/20260322_171209_music/data_fusion/fusion_data.csv",
        "fused_debug_jsonl": "output/runs/20260322_171209_music/data_fusion/debug_fusion_data.jsonl",
        "evaluation_json": "output/runs/20260322_171209_music/pipeline_evaluation/pipeline_evaluation.json",
        "review_summary_json": "output/runs/20260322_171209_music/human_review/human_review_summary.json",
        "review_summary_md": "output/runs/20222_171209_music/human_review/human_review_summary.md",
        "fused_review_csv": "output/runs/20260322_171209_music/human_review/fused_review_table.csv",
        "source_lineage_csv": "output/runs/20260322_171209_music/human_review/source_lineage_long.csv",
        "diff_csv": "output/runs/20260322_171209_music/human_review/fusion_vs_testset_diff.csv"
    }
}


def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def safe_str(v):
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v)


def normalize_value(v):
    s = safe_str(v).strip()
    return s


def read_csv_forgiving(path, warnings):
    if not path or not os.path.exists(path):
        warnings.append(f"Missing CSV file: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    except Exception as e1:
        try:
            return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False, engine="python", on_bad_lines="skip")
        except Exception as e2:
            warnings.append(f"Failed to read CSV {path}: {e1}; fallback failed: {e2}")
            return pd.DataFrame()


def detect_id_column(df):
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    preferred = [
        "_id", "id", "eval_id", "entity_id", "record_id", "source_id", "fused_id", "cluster_id", "uuid"
    ]
    for p in preferred:
        if p in lower_map:
            return lower_map[p]
    for c in cols:
        lc = c.lower()
        if lc.endswith("_id") or lc == "id":
            return c
    uniqueness = []
    n = len(df)
    for c in cols:
        vals = [normalize_value(v) for v in df[c].tolist()]
        non_empty = [v for v in vals if v != ""]
        if not non_empty:
            continue
        unique_ratio = len(set(non_empty)) / max(1, len(non_empty))
        coverage = len(non_empty) / max(1, n)
        score = (1.5 if "id" in c.lower() else 0) + unique_ratio + coverage
        uniqueness.append((score, c))
    if uniqueness:
        uniqueness.sort(reverse=True)
        return uniqueness[0][1]
    return cols[0] if cols else None


def detect_fused_id_column(df):
    if df is None or df.empty:
        return None
    candidates = ["fused_id", "_id", "id", "eval_id", "entity_id", "record_id", "cluster_id"]
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    return detect_id_column(df)


def dataset_name_from_path(path):
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name or "dataset"


def parse_possible_json_or_literal(text):
    if text is None:
        return None
    if isinstance(text, (list, dict)):
        return text
    s = safe_str(text).strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        pass
    return s


def parse_fusion_sources(value):
    parsed = parse_possible_json_or_literal(value)
    out = []

    def add_item(source_id=None, source_dataset=None, raw=None):
        sid = normalize_value(source_id)
        sds = normalize_value(source_dataset)
        out.append({
            "source_id": sid,
            "source_dataset": sds,
            "raw": raw if raw is not None else {"source_id": sid, "source_dataset": sds}
        })

    def walk(obj):
        if obj is None:
            return
        if isinstance(obj, dict):
            sid = None
            sds = None
            for k in ["source_id", "id", "_id", "record_id", "entity_id"]:
                if k in obj:
                    sid = obj.get(k)
                    break
            for k in ["source_dataset", "dataset", "source", "dataset_name"]:
                if k in obj:
                    sds = obj.get(k)
                    break
            if sid is not None or sds is not None:
                add_item(sid, sds, obj)
            else:
                for v in obj.values():
                    walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)
        elif isinstance(obj, str):
            s = obj.strip()
            if not s:
                return
            if "::" in s:
                parts = s.split("::", 1)
                add_item(parts[1], parts[0], obj)
            elif "|" in s:
                parts = s.split("|")
                if len(parts) >= 2:
                    add_item(parts[-1], parts[0], obj)
                else:
                    add_item(parts[0], "", obj)
            elif ":" in s and not re.match(r"^[A-Za-z]:\\", s):
                parts = s.split(":", 1)
                if len(parts) == 2 and parts[0] and parts[1]:
                    add_item(parts[1], parts[0], obj)
                else:
                    add_item(s, "", obj)
            else:
                add_item(s, "", obj)
        else:
            add_item(obj, "", obj)

    walk(parsed)
    dedup = []
    seen = set()
    for item in out:
        key = (item["source_dataset"], item["source_id"])
        if key not in seen:
            seen.add(key)
            dedup.append(item)
    return dedup


def load_source_datasets(paths, warnings):
    datasets = []
    for path in paths:
        df = read_csv_forgiving(path, warnings)
        name = dataset_name_from_path(path)
        id_col = detect_id_column(df) if not df.empty else None
        datasets.append({
            "path": path,
            "name": name,
            "df": df,
            "id_col": id_col
        })
        if df.empty:
            warnings.append(f"Source dataset empty or unreadable: {path}")
        elif not id_col:
            warnings.append(f"Could not confidently detect ID column for source dataset: {path}")
    return datasets


def build_source_indexes(source_datasets):
    by_dataset = {}
    global_index = {}
    for ds in source_datasets:
        name = ds["name"]
        df = ds["df"]
        id_col = ds["id_col"]
        dataset_index = {}
        if not df.empty and id_col and id_col in df.columns:
            for _, row in df.iterrows():
                sid = normalize_value(row.get(id_col, ""))
                if sid != "" and sid not in dataset_index:
                    dataset_index[sid] = row.to_dict()
                if sid != "":
                    global_index.setdefault(sid, []).append((name, row.to_dict()))
        by_dataset[name] = {
            "path": ds["path"],
            "id_col": id_col,
            "rows": dataset_index,
            "columns": list(df.columns) if not df.empty else []
        }
    return by_dataset, global_index


def guess_dataset_for_source_id(source_id, hinted_dataset, by_dataset, global_index):
    sid = normalize_value(source_id)
    hinted = normalize_value(hinted_dataset)
    if hinted and hinted in by_dataset and sid in by_dataset[hinted]["rows"]:
        return hinted, by_dataset[hinted]["rows"].get(sid)
    if sid in global_index:
        matches = global_index[sid]
        if hinted:
            for ds_name, row in matches:
                if ds_name == hinted:
                    return ds_name, row
        return matches[0]
    if hinted and hinted in by_dataset:
        return hinted, None
    return "", None


def parse_xml_testset(path, warnings):
    rows = []
    if not path or not os.path.exists(path):
        warnings.append(f"Missing fusion_testset file: {path}")
        return pd.DataFrame()
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(path)
        root = tree.getroot()
    except Exception as e:
        warnings.append(f"Failed to parse XML testset {path}: {e}")
        return pd.DataFrame()

    def strip_tag(tag):
        return tag.split("}", 1)[-1] if "}" in tag else tag

    record_nodes = []
    for elem in root.iter():
        tag = strip_tag(elem.tag).lower()
        if tag in {"record", "entity", "row", "item", "instance", "release", "entry"}:
            record_nodes.append(elem)
    if not record_nodes:
        record_nodes = list(root)

    for elem in record_nodes:
        row = {}
        attrs = {k: safe_str(v) for k, v in elem.attrib.items()}
        row.update(attrs)
        children = list(elem)
        if children:
            tag_counts = {}
            for child in children:
                ctag = strip_tag(child.tag)
                tag_counts[ctag] = tag_counts.get(ctag, 0) + 1
            repeats = {k for k, v in tag_counts.items() if v > 1}
            seq = {}
            for child in children:
                ctag = strip_tag(child.tag)
                ctext = normalize_value(child.text)
                if ctag in repeats:
                    seq[ctag] = seq.get(ctag, 0) + 1
                    idx = seq[ctag]
                    row[f"{ctag}_{idx}"] = ctext
                else:
                    row[ctag] = ctext
                for gchild in list(child):
                    gtag = strip_tag(gchild.tag)
                    gtext = normalize_value(gchild.text)
                    row[f"{ctag}_{gtag}"] = gtext
                    for attrk, attrv in gchild.attrib.items():
                        row[f"{ctag}_{gtag}_{attrk}"] = safe_str(attrv)
                for attrk, attrv in child.attrib.items():
                    row[f"{ctag}_{attrk}"] = safe_str(attrv)
        text_val = normalize_value(elem.text)
        if text_val and not row:
            row["value"] = text_val
        if row:
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).fillna("")


def build_testset_index(df):
    if df is None or df.empty:
        return None, {}, {}
    id_col = detect_id_column(df)
    idx = {}
    if id_col and id_col in df.columns:
        for _, row in df.iterrows():
            rid = normalize_value(row.get(id_col, ""))
            if rid != "" and rid not in idx:
                idx[rid] = row.to_dict()
    return id_col, idx, {c.lower(): c for c in df.columns}


def map_fused_to_test_row(fused_row, fused_id_col, test_index, test_columns_lower):
    if not test_index:
        return None
    candidate_keys = []
    if fused_id_col and fused_id_col in fused_row:
        candidate_keys.append(normalize_value(fused_row.get(fused_id_col, "")))
    for c in ["eval_id", "_id", "id", "fused_id", "entity_id", "record_id"]:
        if c in fused_row:
            candidate_keys.append(normalize_value(fused_row.get(c, "")))
    seen = set()
    for k in candidate_keys:
        if k and k not in seen:
            seen.add(k)
            if k in test_index:
                return test_index[k]
    for tc in ["eval_id", "_id", "id"]:
        if tc in test_columns_lower:
            real_tc = test_columns_lower[tc]
            for k in candidate_keys:
                if not k:
                    continue
                for test_id, row in test_index.items():
                    if normalize_value(row.get(real_tc, "")) == k:
                        return row
    return None


def main():
    warnings = []
    output_paths = CONTEXT_PAYLOAD.get("output_paths", {})

    fused_csv = output_paths.get("fused_csv") or "output/data_fusion/fusion_data.csv"
    if not os.path.exists(fused_csv):
        fallback = "output/data_fusion/fusion_data.csv"
        if os.path.exists(fallback):
            fused_csv = fallback
        else:
            warnings.append(f"Primary fused CSV not found and fallback missing: {fused_csv} / {fallback}")
            fused_csv = fallback

    fused_review_csv = output_paths.get("fused_review_csv") or "output/human_review/fused_review_table.csv"
    source_lineage_csv = output_paths.get("source_lineage_csv") or "output/human_review/source_lineage_long.csv"
    diff_csv = output_paths.get("diff_csv") or "output/human_review/fusion_vs_testset_diff.csv"
    review_summary_json = output_paths.get("review_summary_json") or "output/human_review/human_review_summary.json"
    review_summary_md = output_paths.get("review_summary_md") or "output/human_review/human_review_summary.md"

    for p in [fused_review_csv, source_lineage_csv, diff_csv, review_summary_json, review_summary_md]:
        ensure_dir(p)

    fused_df = read_csv_forgiving(fused_csv, warnings)
    fused_id_col = detect_fused_id_column(fused_df) if not fused_df.empty else None
    if fused_df.empty:
        warnings.append(f"Fused dataset empty or unreadable: {fused_csv}")

    review_attributes = CONTEXT_PAYLOAD.get("review_attributes", [])
    source_datasets = load_source_datasets(CONTEXT_PAYLOAD.get("datasets", []), warnings)
    by_dataset, global_index = build_source_indexes(source_datasets)

    testset_path = CONTEXT_PAYLOAD.get("fusion_testset")
    test_df = parse_xml_testset(testset_path, warnings) if testset_path else pd.DataFrame()
    test_id_col, test_index, test_columns_lower = build_testset_index(test_df)

    wide_rows = []
    lineage_rows = []
    diff_rows = []

    fused_rows_processed = 0
    source_links_resolved = 0
    source_links_total = 0
    diff_count = 0

    if not fused_df.empty:
        fused_df = fused_df.fillna("")
        for _, fused_row_pd in fused_df.iterrows():
            fused_row = fused_row_pd.to_dict()
            fused_rows_processed += 1
            fused_id = ""
            if fused_id_col and fused_id_col in fused_row:
                fused_id = normalize_value(fused_row.get(fused_id_col, ""))
            if not fused_id:
                for c in ["eval_id", "_id", "id", "fused_id"]:
                    if c in fused_row:
                        fused_id = normalize_value(fused_row.get(c, ""))
                        if fused_id:
                            break

            test_row = map_fused_to_test_row(fused_row, fused_id_col, test_index, test_columns_lower)

            source_entries = parse_fusion_sources(fused_row.get("_fusion_sources", ""))
            source_links_total += len(source_entries)

            resolved_sources = []
            for src in source_entries:
                ds_name, src_row = guess_dataset_for_source_id(
                    src.get("source_id", ""),
                    src.get("source_dataset", ""),
                    by_dataset,
                    global_index
                )
                if src_row is not None:
                    source_links_resolved += 1
                resolved_sources.append({
                    "source_id": normalize_value(src.get("source_id", "")),
                    "source_dataset": ds_name or normalize_value(src.get("source_dataset", "")),
                    "row": src_row or {}
                })

            wide_row = {}
            wide_row["fused_id"] = fused_id

            for attr in review_attributes:
                test_val = normalize_value(test_row.get(attr, "")) if isinstance(test_row, dict) else ""
                fused_val = normalize_value(fused_row.get(attr, ""))
                wide_row[f"{attr}_test"] = test_val
                wide_row[f"{attr}_fused"] = fused_val
                for i in range(1, 4):
                    src_val = ""
                    if i <= len(resolved_sources):
                        src_row = resolved_sources[i - 1]["row"]
                        if isinstance(src_row, dict):
                            src_val = normalize_value(src_row.get(attr, ""))
                    wide_row[f"{attr}_source_{i}"] = src_val
            wide_rows.append(wide_row)

            for src in resolved_sources:
                row = {
                    "fused_id": fused_id,
                    "source_id": src.get("source_id", ""),
                    "source_dataset": src.get("source_dataset", "")
                }
                src_row = src.get("row", {}) if isinstance(src.get("row", {}), dict) else {}
                for attr in review_attributes:
                    row[f"source_{attr}"] = normalize_value(src_row.get(attr, ""))
                    row[f"fused_{attr}"] = normalize_value(fused_row.get(attr, ""))
                lineage_rows.append(row)

            if isinstance(test_row, dict) and test_row:
                for attr in review_attributes:
                    fused_val = normalize_value(fused_row.get(attr, ""))
                    test_val = normalize_value(test_row.get(attr, ""))
                    if fused_val != test_val:
                        diff_rows.append({
                            "fused_id": fused_id,
                            "attribute": attr,
                            "fused_value": fused_val,
                            "test_value": test_val,
                            "diff_type": "missing_fused_value" if fused_val == "" and test_val != "" else "mismatch"
                        })
                        diff_count += 1

    wide_columns = ["fused_id"]
    for attr in review_attributes:
        wide_columns.extend([
            f"{attr}_test",
            f"{attr}_fused",
            f"{attr}_source_1",
            f"{attr}_source_2",
            f"{attr}_source_3"
        ])
    wide_df = pd.DataFrame(wide_rows)
    for c in wide_columns:
        if c not in wide_df.columns:
            wide_df[c] = ""
    wide_df = wide_df[wide_columns]

    lineage_columns = ["fused_id", "source_id", "source_dataset"]
    for attr in review_attributes:
        lineage_columns.append(f"source_{attr}")
    for attr in review_attributes:
        lineage_columns.append(f"fused_{attr}")
    lineage_df = pd.DataFrame(lineage_rows)
    for c in lineage_columns:
        if c not in lineage_df.columns:
            lineage_df[c] = ""
    lineage_df = lineage_df[lineage_columns]

    diff_columns = ["fused_id", "attribute", "fused_value", "test_value", "diff_type"]
    diff_df = pd.DataFrame(diff_rows)
    for c in diff_columns:
        if c not in diff_df.columns:
            diff_df[c] = ""
    diff_df = diff_df[diff_columns]

    if test_df.empty:
        warnings.append("fusion_testset unavailable or unparsable; created empty diff CSV.")

    wide_df.to_csv(fused_review_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    lineage_df.to_csv(source_lineage_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    diff_df.to_csv(diff_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    summary = {
        "summary": "Human-review outputs generated for fused entities, source lineage, and fusion-vs-testset diffs.",
        "file_paths": {
            "fused_review_table": fused_review_csv,
            "source_lineage_long": source_lineage_csv,
            "fusion_vs_testset_diff": diff_csv,
            "human_review_summary_json": review_summary_json,
            "human_review_summary_md": review_summary_md,
            "fused_input_csv": fused_csv,
            "fusion_testset": testset_path or ""
        },
        "counts": {
            "fused_rows_input": int(len(fused_df)) if fused_df is not None else 0,
            "fused_rows_processed": fused_rows_processed,
            "review_table_rows": int(len(wide_df)),
            "source_lineage_rows": int(len(lineage_df)),
            "diff_rows": int(len(diff_df)),
            "source_links_total": int(source_links_total),
            "source_links_resolved": int(source_links_resolved),
            "source_datasets_loaded": int(len(source_datasets)),
            "review_attributes_count": int(len(review_attributes))
        },
        "warnings": warnings,
        "created_at": now_iso()
    }

    with open(review_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    md_lines = [
        "# Human Review Summary",
        "",
        f"- Created at: {summary['created_at']}",
        f"- Summary: {summary['summary']}",
        "",
        "## Files",
    ]
    for k, v in summary["file_paths"].items():
        md_lines.append(f"- **{k}**: `{v}`")
    md_lines.extend([
        "",
        "## Counts",
    ])
    for k, v in summary["counts"].items():
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
    with open(review_summary_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))


if __name__ == "__main__":
    main()