import os
import re
import json
import ast
from datetime import datetime, timezone
from xml.etree import ElementTree as ET

import pandas as pd


CONTEXT_PAYLOAD = {
    "datasets": [
        "output/runs/20260322_102925_companies/normalization/attempt_1/forbes.csv",
        "output/runs/20260322_102925_companies/normalization/attempt_1/dbpedia.csv",
        "output/runs/20260322_102925_companies/normalization/attempt_1/fullcontact.csv"
    ],
    "fusion_testset": "input/datasets/companies/testsets/test_set.xml",
    "review_attributes": [
        "_id",
        "Continent",
        "Market Value",
        "Profits",
        "Rank",
        "Sales",
        "Sector",
        "assets",
        "city",
        "country",
        "founded",
        "id",
        "industry",
        "keypeople_name",
        "name"
    ],
    "evaluation_metrics": {
        "overall_accuracy": 0.48484848484848486,
        "macro_accuracy": 0.46095238095238095,
        "num_evaluated_records": 25,
        "num_evaluated_attributes": 6,
        "total_evaluations": 132,
        "total_correct": 64,
        "keypeople_name_accuracy": 0.2857142857142857,
        "keypeople_name_count": 7,
        "city_accuracy": 0.2,
        "city_count": 25,
        "name_accuracy": 1.0,
        "name_count": 25,
        "assets_accuracy": 0.0,
        "assets_count": 25,
        "country_accuracy": 0.96,
        "country_count": 25,
        "founded_accuracy": 0.32,
        "founded_count": 25
    },
    "auto_diagnostics": {
        "debug_reason_counts": {
            "mismatch": 37,
            "missing_fused_value": 31
        },
        "debug_reason_ratios": {
            "mismatch": 0.5441176470588235,
            "missing_fused_value": 0.45588235294117646
        },
        "evaluation_stage": "validation",
        "evaluation_testset_path": "input/datasets/companies/testsets/validation_set.xml",
        "id_alignment": {
            "gold_count": 25,
            "direct_coverage": 14,
            "direct_coverage_ratio": 0.56,
            "mapped_coverage": 25,
            "mapped_coverage_ratio": 1.0,
            "missing_gold_ids": []
        },
        "fusion_size_comparison": {
            "estimate_path": "output/runs/20260322_102925_companies/pipeline_evaluation/fusion_size_estimate.json",
            "actual": {
                "rows": 12870,
                "unique_ids": 12870
            },
            "comparisons": {
                "blocking": {
                    "expected_rows": 12004,
                    "actual_rows": 12870,
                    "rows_abs_error": 866,
                    "rows_pct_error": 0.07214261912695769,
                    "expected_rows_matched_only": 2013,
                    "expected_rows_singleton_aware": 12004,
                    "rows_pct_error_matched_only": 5.39344262295082,
                    "rows_pct_error_singleton_aware": 0.07214261912695769,
                    "better_variant": "singleton_aware",
                    "expected_unique_ids": 12004,
                    "actual_unique_ids": 12870,
                    "unique_ids_abs_error": 866,
                    "unique_ids_pct_error": 0.07214261912695769,
                    "reasoning": "Estimate is close to actual fused size. Blocking estimate was conservative; downstream matching/fusion retained more entities than expected.",
                    "compared_at": "2026-03-22T09:40:20.230912+00:00"
                },
                "matching": {
                    "expected_rows": 11792,
                    "actual_rows": 12870,
                    "rows_abs_error": 1078,
                    "rows_pct_error": 0.0914179104477612,
                    "expected_rows_matched_only": 2225,
                    "expected_rows_singleton_aware": 11792,
                    "rows_pct_error_matched_only": 4.784269662921348,
                    "rows_pct_error_singleton_aware": 0.0914179104477612,
                    "better_variant": "singleton_aware",
                    "expected_unique_ids": 11792,
                    "actual_unique_ids": 12870,
                    "unique_ids_abs_error": 1078,
                    "unique_ids_pct_error": 0.0914179104477612,
                    "reasoning": "Estimate is close to actual fused size. Matching estimate seems conservative; matcher/fusion retained more clusters than projected.",
                    "compared_at": "2026-03-22T09:40:20.231008+00:00"
                }
            }
        },
        "overall_accuracy": 0.48484848484848486
    },
    "integration_diagnostics_report": {},
    "output_paths": {
        "fused_csv": "output/runs/20260322_102925_companies/data_fusion/fusion_data.csv",
        "fused_debug_jsonl": "output/runs/20260322_102925_companies/data_fusion/debug_fusion_data.jsonl",
        "evaluation_json": "output/runs/20260322_102925_companies/pipeline_evaluation/pipeline_evaluation.json",
        "review_summary_json": "output/runs/20260322_102925_companies/human_review/human_review_summary.json",
        "review_summary_md": "output/runs/20260322_102925_companies/human_review/human_review_summary.md",
        "fused_review_csv": "output/runs/20260322_102925_companies/human_review/fused_review_table.csv",
        "source_lineage_csv": "output/runs/20260322_102925_companies/human_review/source_lineage_long.csv",
        "diff_csv": "output/runs/20260322_102925_companies/human_review/fusion_vs_testset_diff.csv"
    }
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def normalize_text(v):
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).strip()


def safe_read_csv(path, warnings):
    if not path or not os.path.exists(path):
        warnings.append(f"Missing CSV file: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as e:
        warnings.append(f"Failed to read CSV {path}: {e}")
        return pd.DataFrame()


def detect_id_column(df):
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    preferred_exact = ["_id", "id", "ID", "Id"]
    for c in preferred_exact:
        if c in cols:
            return c
    lower_map = {str(c).strip().lower(): c for c in cols}
    for candidate in ["_id", "id", "record_id", "entity_id", "source_id", "uuid"]:
        if candidate in lower_map:
            return lower_map[candidate]
    scored = []
    for c in cols:
        lc = str(c).strip().lower()
        score = 0
        if lc.endswith("id"):
            score += 3
        if lc == "identifier":
            score += 2
        if "id" in lc:
            score += 1
        non_empty = df[c].astype(str).str.strip().replace({"": pd.NA}).dropna()
        uniq_ratio = (non_empty.nunique(dropna=True) / len(non_empty)) if len(non_empty) else 0
        if uniq_ratio > 0.95:
            score += 2
        elif uniq_ratio > 0.75:
            score += 1
        scored.append((score, c))
    scored.sort(reverse=True)
    return scored[0][1] if scored and scored[0][0] > 0 else None


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
    parsers = []
    try:
        parsers.append(json.loads(s))
    except Exception:
        pass
    try:
        parsers.append(ast.literal_eval(s))
    except Exception:
        pass
    for obj in parsers:
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            if "sources" in obj and isinstance(obj["sources"], list):
                return obj["sources"]
            return [obj]
    parts = [p.strip() for p in re.split(r"[|;,]+", s) if p.strip()]
    results = []
    for p in parts:
        if "::" in p:
            a, b = p.split("::", 1)
            results.append({"dataset": a.strip(), "id": b.strip()})
        elif ":" in p:
            a, b = p.split(":", 1)
            results.append({"dataset": a.strip(), "id": b.strip()})
        else:
            results.append({"id": p})
    return results


def source_ref_candidates(ref):
    candidates = []
    if isinstance(ref, dict):
        dataset = None
        for k in ["dataset", "source_dataset", "source", "table", "file", "path", "dataset_name"]:
            if k in ref and normalize_text(ref.get(k)):
                dataset = normalize_text(ref.get(k))
                break
        sid = None
        for k in ["id", "source_id", "_id", "record_id", "entity_id", "pk"]:
            if k in ref and normalize_text(ref.get(k)):
                sid = normalize_text(ref.get(k))
                break
        if sid:
            candidates.append((dataset, sid))
        for k, v in ref.items():
            if isinstance(v, (str, int, float)):
                sv = normalize_text(v)
                if sv and (k.lower().endswith("id") or k.lower() in {"id", "_id", "source_id", "record_id"}):
                    candidates.append((dataset, sv))
    elif isinstance(ref, str):
        for item in parse_fusion_sources(ref):
            candidates.extend(source_ref_candidates(item))
    return candidates


def basename_no_ext(path):
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


def build_source_indexes(dataset_paths, warnings):
    datasets = []
    for path in dataset_paths:
        df = safe_read_csv(path, warnings)
        dataset_name = basename_no_ext(path)
        id_col = detect_id_column(df) if not df.empty else None
        id_index = {}
        if not df.empty and id_col and id_col in df.columns:
            for _, row in df.iterrows():
                key = normalize_text(row.get(id_col))
                if key and key not in id_index:
                    id_index[key] = row.to_dict()
        else:
            warnings.append(f"Could not robustly detect ID column for dataset: {path}")
        datasets.append({
            "path": path,
            "dataset_name": dataset_name,
            "df": df,
            "id_col": id_col,
            "id_index": id_index,
        })
    return datasets


def match_dataset_hint(dataset_hint, datasets):
    if not dataset_hint:
        return None
    hint = normalize_text(dataset_hint).lower()
    hint_base = basename_no_ext(hint)
    for ds in datasets:
        names = {
            normalize_text(ds["dataset_name"]).lower(),
            normalize_text(os.path.basename(ds["path"])).lower(),
            normalize_text(ds["path"]).lower(),
            basename_no_ext(ds["path"]).lower(),
        }
        if hint in names or hint_base in names:
            return ds
        if hint_base and any(hint_base == n or hint_base in n for n in names):
            return ds
    return None


def resolve_source_rows(fused_row, datasets):
    refs = parse_fusion_sources(fused_row.get("_fusion_sources", ""))
    resolved = []
    seen = set()
    for ref in refs:
        candidates = source_ref_candidates(ref)
        found = False
        for dataset_hint, sid in candidates:
            sid = normalize_text(sid)
            if not sid:
                continue
            if dataset_hint:
                ds = match_dataset_hint(dataset_hint, datasets)
                if ds and sid in ds["id_index"]:
                    key = (ds["dataset_name"], sid)
                    if key not in seen:
                        resolved.append((sid, ds["dataset_name"], ds["id_index"][sid]))
                        seen.add(key)
                    found = True
                    break
            else:
                for ds in datasets:
                    if sid in ds["id_index"]:
                        key = (ds["dataset_name"], sid)
                        if key not in seen:
                            resolved.append((sid, ds["dataset_name"], ds["id_index"][sid]))
                            seen.add(key)
                        found = True
                        break
        if not found and isinstance(ref, dict):
            sid = ""
            for k in ["id", "source_id", "_id", "record_id", "entity_id"]:
                sid = normalize_text(ref.get(k))
                if sid:
                    break
            ds_hint = ""
            for k in ["dataset", "source_dataset", "source", "table", "file", "path", "dataset_name"]:
                ds_hint = normalize_text(ref.get(k))
                if ds_hint:
                    break
            key = (ds_hint or "", sid or "")
            if key not in seen and (sid or ds_hint):
                resolved.append((sid, ds_hint, {}))
                seen.add(key)
    return resolved[:3], refs


def load_testset(path, warnings):
    if not path or not os.path.exists(path):
        warnings.append(f"Missing fusion_testset file: {path}")
        return pd.DataFrame()
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return safe_read_csv(path, warnings)
    if ext == ".json":
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                return pd.DataFrame(obj)
            if isinstance(obj, dict):
                if "records" in obj and isinstance(obj["records"], list):
                    return pd.DataFrame(obj["records"])
                return pd.DataFrame([obj])
        except Exception as e:
            warnings.append(f"Failed to parse JSON testset {path}: {e}")
            return pd.DataFrame()
    if ext == ".xml":
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            records = []

            def strip_ns(tag):
                return tag.split("}", 1)[-1] if "}" in tag else tag

            def collect_flat(elem, prefix="", out=None):
                if out is None:
                    out = {}
                children = list(elem)
                text = normalize_text(elem.text)
                attrs = {f"{prefix}@{k}" if prefix else f"@{k}": normalize_text(v) for k, v in elem.attrib.items()}
                out.update(attrs)
                if not children:
                    tag = strip_ns(elem.tag)
                    key = prefix or tag
                    if text:
                        out[key] = text
                    return out
                counts = {}
                for child in children:
                    ctag = strip_ns(child.tag)
                    counts[ctag] = counts.get(ctag, 0) + 1
                repeated = any(v > 1 for v in counts.values())
                if repeated and not text:
                    direct_leaf_children = all(len(list(c)) == 0 for c in children)
                    if direct_leaf_children:
                        candidate = {}
                        candidate.update(attrs)
                        for child in children:
                            ctag = strip_ns(child.tag)
                            candidate[ctag] = normalize_text(child.text)
                            for ak, av in child.attrib.items():
                                candidate[f"{ctag}@{ak}"] = normalize_text(av)
                        if len(candidate) > 1:
                            records.append(candidate)
                            return out
                for child in children:
                    ctag = strip_ns(child.tag)
                    child_prefix = f"{prefix}.{ctag}" if prefix else ctag
                    collect_flat(child, child_prefix, out)
                return out

            collect_flat(root, "", {})
            if not records:
                for elem in root.iter():
                    children = list(elem)
                    if not children:
                        continue
                    row = {}
                    simple = True
                    for child in children:
                        if list(child):
                            simple = False
                            break
                        row[strip_ns(child.tag)] = normalize_text(child.text)
                        for ak, av in child.attrib.items():
                            row[f"{strip_ns(child.tag)}@{ak}"] = normalize_text(av)
                    if simple and row:
                        for ak, av in elem.attrib.items():
                            row[f"@{ak}"] = normalize_text(av)
                        records.append(row)
                dedup = []
                seen = set()
                for r in records:
                    key = json.dumps(r, sort_keys=True, ensure_ascii=False)
                    if key not in seen:
                        seen.add(key)
                        dedup.append(r)
                records = dedup
            return pd.DataFrame(records)
        except Exception as e:
            warnings.append(f"Failed to parse XML testset {path}: {e}")
            return pd.DataFrame()
    warnings.append(f"Unsupported testset extension for {path}; creating empty testset dataframe.")
    return pd.DataFrame()


def build_testset_index(test_df):
    if test_df.empty:
        return None, {}
    id_col = detect_id_column(test_df)
    idx = {}
    if id_col and id_col in test_df.columns:
        for _, row in test_df.iterrows():
            key = normalize_text(row.get(id_col))
            if key and key not in idx:
                idx[key] = row.to_dict()
    return id_col, idx


def choose_fused_id_column(fused_df):
    return detect_id_column(fused_df) or "_id"


def main():
    warnings = []
    output_paths = CONTEXT_PAYLOAD.get("output_paths", {})
    review_attributes = CONTEXT_PAYLOAD.get("review_attributes", [])

    fused_csv_path = output_paths.get("fused_csv") or "output/data_fusion/fusion_data.csv"
    if not os.path.exists(fused_csv_path):
        fallback = "output/data_fusion/fusion_data.csv"
        if os.path.exists(fallback):
            fused_csv_path = fallback

    review_dir = os.path.dirname(output_paths.get("review_summary_json", "output/human_review/human_review_summary.json")) or "output/human_review"
    ensure_dir(review_dir)

    fused_review_csv = output_paths.get("fused_review_csv", os.path.join(review_dir, "fused_review_table.csv"))
    source_lineage_csv = output_paths.get("source_lineage_csv", os.path.join(review_dir, "source_lineage_long.csv"))
    diff_csv = output_paths.get("diff_csv", os.path.join(review_dir, "fusion_vs_testset_diff.csv"))
    summary_json = output_paths.get("review_summary_json", os.path.join(review_dir, "human_review_summary.json"))
    summary_md = output_paths.get("review_summary_md", os.path.join(review_dir, "human_review_summary.md"))

    fused_df = safe_read_csv(fused_csv_path, warnings)
    fused_id_col = choose_fused_id_column(fused_df) if not fused_df.empty else "_id"
    if fused_df.empty:
        warnings.append("Fused dataframe is empty or unavailable.")

    source_datasets = build_source_indexes(CONTEXT_PAYLOAD.get("datasets", []), warnings)
    test_df = load_testset(CONTEXT_PAYLOAD.get("fusion_testset"), warnings)
    test_id_col, test_index = build_testset_index(test_df)

    wide_columns = [fused_id_col]
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

    fused_records_processed = 0
    fused_with_sources = 0
    mapped_testset_rows = 0

    for _, row in fused_df.iterrows():
        fused_records_processed += 1
        fused_row = row.to_dict()
        fused_id = normalize_text(fused_row.get(fused_id_col))
        resolved_sources, raw_refs = resolve_source_rows(fused_row, source_datasets)
        if resolved_sources:
            fused_with_sources += 1

        test_row = test_index.get(fused_id, {}) if fused_id else {}
        if test_row:
            mapped_testset_rows += 1

        out_row = {fused_id_col: fused_id}
        for attr in review_attributes:
            out_row[f"{attr}_test"] = normalize_text(test_row.get(attr, ""))
            out_row[f"{attr}_fused"] = normalize_text(fused_row.get(attr, ""))
            for i in range(1, 4):
                out_row[f"{attr}_source_{i}"] = ""
            for i, (_, _, src_row) in enumerate(resolved_sources[:3], start=1):
                out_row[f"{attr}_source_{i}"] = normalize_text(src_row.get(attr, "")) if src_row else ""
        wide_rows.append(out_row)

        for sid, sdataset, src_row in resolved_sources:
            lineage = {
                "fused_id": fused_id,
                "source_id": normalize_text(sid),
                "source_dataset": normalize_text(sdataset),
            }
            for attr in review_attributes:
                lineage[f"source__{attr}"] = normalize_text(src_row.get(attr, "")) if src_row else ""
                lineage[f"fused__{attr}"] = normalize_text(fused_row.get(attr, ""))
            lineage_rows.append(lineage)

        if test_row:
            overlap_attrs = [c for c in review_attributes if c in fused_df.columns and c in test_df.columns]
            for attr in overlap_attrs:
                fused_val = normalize_text(fused_row.get(attr, ""))
                test_val = normalize_text(test_row.get(attr, ""))
                if fused_val != test_val:
                    diff_rows.append({
                        "fused_id": fused_id,
                        "attribute": attr,
                        "fused_value": fused_val,
                        "testset_value": test_val,
                        "is_diff": True,
                    })

    wide_df = pd.DataFrame(wide_rows, columns=wide_columns)
    if wide_df.empty:
        wide_df = pd.DataFrame(columns=wide_columns)

    lineage_columns = ["fused_id", "source_id", "source_dataset"]
    for attr in review_attributes:
        lineage_columns.append(f"source__{attr}")
    for attr in review_attributes:
        lineage_columns.append(f"fused__{attr}")
    lineage_df = pd.DataFrame(lineage_rows, columns=lineage_columns)
    if lineage_df.empty:
        lineage_df = pd.DataFrame(columns=lineage_columns)

    diff_columns = ["fused_id", "attribute", "fused_value", "testset_value", "is_diff"]
    diff_df = pd.DataFrame(diff_rows, columns=diff_columns)
    if diff_df.empty:
        diff_df = pd.DataFrame(columns=diff_columns)

    if test_df.empty:
        warnings.append("Testset unavailable or empty; created empty diff CSV.")
    elif not test_id_col:
        warnings.append("Could not detect testset ID column robustly; diff output may be empty or incomplete.")
    elif mapped_testset_rows == 0:
        warnings.append("No fused rows were directly mapped to testset rows by detected ID column.")

    for path in [fused_review_csv, source_lineage_csv, diff_csv, summary_json, summary_md]:
        ensure_dir(os.path.dirname(path) or ".")

    wide_df.to_csv(fused_review_csv, index=False)
    lineage_df.to_csv(source_lineage_csv, index=False)
    diff_df.to_csv(diff_csv, index=False)

    created_at = datetime.now(timezone.utc).isoformat()
    summary = {
        "summary": "Human-review outputs generated for fused entities, source lineage, and fusion-vs-testset diffs.",
        "file_paths": {
            "fused_review_table": fused_review_csv,
            "source_lineage_long": source_lineage_csv,
            "fusion_vs_testset_diff": diff_csv,
            "human_review_summary_json": summary_json,
            "human_review_summary_md": summary_md,
            "fused_csv_used": fused_csv_path,
            "fusion_testset_used": CONTEXT_PAYLOAD.get("fusion_testset", "")
        },
        "counts": {
            "fused_rows_input": int(len(fused_df)),
            "fused_rows_review_output": int(len(wide_df)),
            "source_lineage_rows": int(len(lineage_df)),
            "diff_rows": int(len(diff_df)),
            "fused_rows_with_resolved_sources": int(fused_with_sources),
            "testset_rows": int(len(test_df)),
            "mapped_testset_rows": int(mapped_testset_rows),
            "source_datasets_loaded": int(len(source_datasets))
        },
        "warnings": warnings,
        "created_at": created_at
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    md_lines = [
        "# Human Review Summary",
        "",
        f"- Created at: {created_at}",
        f"- Fused rows input: {len(fused_df)}",
        f"- Review rows written: {len(wide_df)}",
        f"- Source lineage rows: {len(lineage_df)}",
        f"- Diff rows: {len(diff_df)}",
        f"- Testset rows: {len(test_df)}",
        f"- Mapped testset rows: {mapped_testset_rows}",
        "",
        "## Files",
        f"- Fused review table: `{fused_review_csv}`",
        f"- Source lineage long: `{source_lineage_csv}`",
        f"- Fusion vs testset diff: `{diff_csv}`",
        f"- Summary JSON: `{summary_json}`",
        "",
        "## Warnings",
    ]
    if warnings:
        md_lines.extend([f"- {w}" for w in warnings])
    else:
        md_lines.append("- None")
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))


if __name__ == "__main__":
    main()