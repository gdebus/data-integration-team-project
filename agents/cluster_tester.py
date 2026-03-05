import pandas as pd
import numpy as np
import json
import os
import shutil
from typing import Dict, Any, List, Optional, Tuple

from PyDI.entitymatching import EntityMatchingEvaluator
from PyDI.io import load_xml, load_parquet, load_csv


def load_dataset(path):
    """Loads a dataset from a given path, supporting parquet, csv, and xml."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        df = load_parquet(path)
    elif ext == ".csv":
        df = load_csv(path)
    elif ext == ".xml":
        df = load_xml(path, nested_handling="aggregate")
    else:
        raise ValueError(f"Unsupported format: {ext}. Supported: .csv, .parquet, .xml")
    return df


class ClusterTester:
    """
    Analyzes entity clusters and recommends specific PyDI post-processing steps.
    """

    def __init__(
        self,
        llm,
        output_dir: str = "output/cluster-evaluation",
        verbose: bool = True,
        min_small_cluster_ratio: float = 0.75,
        max_large_cluster_ratio: float = 0.05,
        max_healthy_cluster_size: int = 15,
        max_one_to_many_ratio: float = 0.2,
        ambiguous_match_margin: float = 0.05,
        ambiguous_ratio_threshold: float = 0.2,
        target_small_cluster_ratio: float = 0.9,
        target_large_cluster_ratio: float = 0.005,
        target_one_to_many_ratio: float = 0.03,
        dataset_dir: str = "output/schema-matching",
        dataset_paths: Optional[Dict[str, str]] = None,
        id_columns: Optional[Dict[str, str]] = None,
        max_edges_per_cluster: int = 1500,
        large_cluster_min_size: int = 6,
        max_large_clusters_reported: int = 5,
        max_hubs_reported: int = 5,
        max_feature_drivers: int = 5,
        max_attribute_reports: int = 5,
    ):
        self.llm = llm
        self.output_dir = output_dir
        self.verbose = verbose
        self.min_small_cluster_ratio = min_small_cluster_ratio
        self.max_large_cluster_ratio = max_large_cluster_ratio
        self.max_healthy_cluster_size = max_healthy_cluster_size
        self.max_one_to_many_ratio = max_one_to_many_ratio
        self.ambiguous_match_margin = ambiguous_match_margin
        self.ambiguous_ratio_threshold = ambiguous_ratio_threshold
        self.target_small_cluster_ratio = target_small_cluster_ratio
        self.target_large_cluster_ratio = target_large_cluster_ratio
        self.target_one_to_many_ratio = target_one_to_many_ratio
        self.dataset_dir = dataset_dir
        self.dataset_paths = dataset_paths or {}
        self.id_columns = id_columns or {}
        self.max_edges_per_cluster = max_edges_per_cluster
        self.large_cluster_min_size = large_cluster_min_size
        self.max_large_clusters_reported = max_large_clusters_reported
        self.max_hubs_reported = max_hubs_reported
        self.max_feature_drivers = max_feature_drivers
        self.max_attribute_reports = max_attribute_reports
        self._dataset_cache: Dict[str, pd.DataFrame] = {}
        # Clean and recreate the output directory for a fresh run
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def _json_sanitize(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._json_sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._json_sanitize(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        try:
            if pd.isna(obj):
                return None
        except Exception:
            pass
        return obj

    def _analyze_cluster_health(self, cluster_stats: pd.DataFrame) -> Dict[str, Any]:
        """Compute simple health metrics for a cluster size distribution."""
        size_freq = {}
        for _, row in cluster_stats.iterrows():
            size = int(row["cluster_size"])
            freq = int(row["frequency"])
            size_freq[size] = size_freq.get(size, 0) + freq

        total_clusters = sum(size_freq.values())
        if total_clusters == 0:
            return {
                "total_clusters": 0,
                "max_cluster_size": 0,
                "small_cluster_ratio": 0.0,
                "large_cluster_ratio": 0.0,
                "is_long_tail": True,
            }

        max_cluster_size = max(size_freq.keys())
        small_clusters = sum(freq for size, freq in size_freq.items() if size <= 2)
        large_clusters = sum(freq for size, freq in size_freq.items() if size >= 6)

        small_cluster_ratio = small_clusters / total_clusters
        large_cluster_ratio = large_clusters / total_clusters

        is_long_tail = (
            small_cluster_ratio >= self.min_small_cluster_ratio
            and large_cluster_ratio <= self.max_large_cluster_ratio
            and max_cluster_size <= self.max_healthy_cluster_size
        )

        return {
            "total_clusters": total_clusters,
            "max_cluster_size": max_cluster_size,
            "small_cluster_ratio": round(small_cluster_ratio, 4),
            "large_cluster_ratio": round(large_cluster_ratio, 4),
            "is_long_tail": is_long_tail,
        }

    def _get_recommendation(
        self,
        cluster_stats: pd.DataFrame,
        duplicate_stats: Dict[str, Any],
        score_stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Summarize cluster health and flag if matcher configuration should be adjusted."""
        health = self._analyze_cluster_health(cluster_stats)
        health["one_to_many_ratio"] = duplicate_stats.get("one_to_many_ratio", 0.0)
        health["max_degree"] = duplicate_stats.get("max_degree", 0)
        health["ambiguous_ratio"] = duplicate_stats.get("ambiguous_ratio", 0.0)

        is_one_to_many_ok = health["one_to_many_ratio"] <= self.max_one_to_many_ratio
        is_ambiguous = health["ambiguous_ratio"] >= self.ambiguous_ratio_threshold
        has_large_clusters = (
            health["large_cluster_ratio"] > self.max_large_cluster_ratio
            or health["max_cluster_size"] > self.max_healthy_cluster_size
        )
        meets_target = (
            health["small_cluster_ratio"] >= self.target_small_cluster_ratio
            and health["large_cluster_ratio"] <= self.target_large_cluster_ratio
            and health["one_to_many_ratio"] <= self.target_one_to_many_ratio
        )

        if (
            health["is_long_tail"] and is_one_to_many_ok and not is_ambiguous
        ) or meets_target:
            diagnosis = (
                "The cluster distribution shows a healthy long-tail pattern with mostly "
                "small clusters and very few large clusters."
            )
            return {
                "diagnosis": diagnosis,
                "recommended_strategy": "None",
                "recommended_action": "None",
                "parameters": {},
                "health_metrics": health,
                "duplicate_stats": duplicate_stats,
                "score_stats": score_stats,
                "adjustment_signals": {
                    "large_cluster_ratio": health["large_cluster_ratio"],
                    "one_to_many_ratio": health["one_to_many_ratio"],
                    "ambiguous_ratio": health["ambiguous_ratio"],
                },
                "meets_target": meets_target,
                "is_healthy": True,
            }

        if is_ambiguous:
            diagnosis = (
                "Many entities have multiple close-scoring matches. This indicates ambiguous "
                "duplicates and suggests matcher weights or thresholds need refinement."
            )
        elif has_large_clusters:
            diagnosis = (
                "Large clusters suggest transitive over-grouping. Matching configuration "
                "should be tightened or reweighted to reduce hub formation."
            )
        else:
            diagnosis = (
                "Many entities map to multiple counterparts, indicating duplicate assignments. "
                "Matcher weights likely overemphasize broad attributes."
            )

        return {
            "diagnosis": diagnosis,
            "recommended_strategy": "None",
            "recommended_action": "update_matching_config",
            "parameters": {},
            "health_metrics": health,
            "duplicate_stats": duplicate_stats,
            "score_stats": score_stats,
            "adjustment_signals": {
                "large_cluster_ratio": health["large_cluster_ratio"],
                "one_to_many_ratio": health["one_to_many_ratio"],
                "ambiguous_ratio": health["ambiguous_ratio"],
                "max_cluster_size": health["max_cluster_size"],
            },
            "meets_target": meets_target,
            "is_healthy": False,
        }

    def _parse_dataset_names_from_correspondence(
        self, path: str
    ) -> Tuple[Optional[str], Optional[str]]:
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem.startswith("correspondences_"):
            stem = stem[len("correspondences_") :]

        # If explicit dataset_paths are provided, try to match by longest names
        if self.dataset_paths:
            names = list(self.dataset_paths.keys())
            matches = [name for name in names if name in stem]
            if len(matches) >= 2:
                matches.sort(key=len, reverse=True)
                left = matches[0]
                right = next((m for m in matches[1:] if m != left), None)
                return left, right

        if "_" in stem:
            left, right = stem.rsplit("_", 1)
            return left, right

        return None, None

    def _resolve_dataset(
        self, name: Optional[str]
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        if not name:
            return None, None

        if name in self._dataset_cache:
            return self._dataset_cache[name], self._resolve_id_column(
                self._dataset_cache[name], name
            )

        path = self.dataset_paths.get(name)
        if not path:
            for ext in (".csv", ".parquet", ".xml"):
                candidate = os.path.join(self.dataset_dir, f"{name}{ext}")
                if os.path.exists(candidate):
                    path = candidate
                    break

        if not path or not os.path.exists(path):
            return None, None

        df = load_dataset(path)
        self._dataset_cache[name] = df
        return df, self._resolve_id_column(df, name)

    def _resolve_id_column(
        self, df: pd.DataFrame, dataset_name: Optional[str]
    ) -> Optional[str]:
        if dataset_name and dataset_name in self.id_columns:
            return self.id_columns[dataset_name]

        for candidate in ("id", "ID", "Id", "record_id", "recordId"):
            if candidate in df.columns:
                return candidate

        return df.columns[0] if len(df.columns) else None

    def _common_attribute_columns(
        self,
        left_df: Optional[pd.DataFrame],
        right_df: Optional[pd.DataFrame],
        left_id_col: Optional[str],
        right_id_col: Optional[str],
    ) -> List[str]:
        if left_df is None or right_df is None:
            return []
        return [
            col
            for col in left_df.columns
            if col in right_df.columns and col not in (left_id_col, right_id_col)
        ]

    def _compute_hub_attribute_overlap(
        self,
        hub_id: Any,
        hub_side: str,
        edges: pd.DataFrame,
        left_df: Optional[pd.DataFrame],
        right_df: Optional[pd.DataFrame],
        left_id_col: Optional[str],
        right_id_col: Optional[str],
    ) -> List[Dict[str, Any]]:
        if (
            left_df is None
            or right_df is None
            or left_id_col not in left_df.columns
            or right_id_col not in right_df.columns
        ):
            return []

        if hub_side == "id1":
            hub_df = left_df
            hub_id_col = left_id_col
            other_df = right_df
            other_id_col = right_id_col
            other_ids = edges.loc[edges["id1"] == hub_id, "id2"]
        else:
            hub_df = right_df
            hub_id_col = right_id_col
            other_df = left_df
            other_id_col = left_id_col
            other_ids = edges.loc[edges["id2"] == hub_id, "id1"]

        hub_rows = hub_df[hub_df[hub_id_col] == hub_id]
        if hub_rows.empty:
            return []
        hub_row = hub_rows.iloc[0]

        other_subset = other_df[other_df[other_id_col].isin(other_ids)]
        if other_subset.empty:
            return []

        common_cols = self._common_attribute_columns(
            left_df, right_df, left_id_col, right_id_col
        )
        overlaps = []
        for col in common_cols:
            hub_val = hub_row.get(col)
            if pd.isna(hub_val):
                continue
            series = other_subset[col].dropna()
            if series.empty:
                continue
            if pd.api.types.is_numeric_dtype(series):
                series_num = pd.to_numeric(series, errors="coerce")
                hub_num = pd.to_numeric(pd.Series([hub_val]), errors="coerce").iloc[0]
                if pd.isna(hub_num):
                    continue
                diff = (series_num - hub_num).abs()
                denom = series_num.abs().replace(0, pd.NA)
                rel = (diff / denom).dropna()
                within_5pct = float((rel <= 0.05).mean()) if not rel.empty else 0.0
                overlaps.append(
                    {
                        "attribute": col,
                        "hub_value": hub_val,
                        "type": "numeric",
                        "pct_within_5pct": round(within_5pct, 4),
                        "coverage": round(float(series_num.notna().mean()), 4),
                    }
                )
            else:
                hub_norm = str(hub_val).lower().strip()
                series_norm = series.astype(str).str.lower().str.strip()
                match_rate = float((series_norm == hub_norm).mean())
                overlaps.append(
                    {
                        "attribute": col,
                        "hub_value": hub_val,
                        "type": "text",
                        "match_rate": round(match_rate, 4),
                        "coverage": round(float(series.notna().mean()), 4),
                    }
                )

        overlaps.sort(
            key=lambda x: x.get("match_rate", x.get("pct_within_5pct", 0.0)),
            reverse=True,
        )
        return overlaps[: self.max_attribute_reports]

    def _build_clusters(self, correspondences: pd.DataFrame) -> List[Dict[str, Any]]:
        if "id1" not in correspondences.columns or "id2" not in correspondences.columns:
            return []

        parent: Dict[Any, Any] = {}

        def find(node):
            while parent.get(node, node) != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return parent.get(node, node)

        def union(a, b):
            ra = find(a)
            rb = find(b)
            if ra != rb:
                parent[rb] = ra

        for row in correspondences[["id1", "id2"]].itertuples(index=False):
            left = ("id1", row[0])
            right = ("id2", row[1])
            if left not in parent:
                parent[left] = left
            if right not in parent:
                parent[right] = right
            union(left, right)

        clusters: Dict[Any, Dict[str, Any]] = {}
        for idx, row in correspondences[["id1", "id2"]].iterrows():
            left = ("id1", row["id1"])
            root = find(left)
            cluster = clusters.setdefault(
                root, {"id1": set(), "id2": set(), "edge_indices": []}
            )
            cluster["id1"].add(row["id1"])
            cluster["id2"].add(row["id2"])
            cluster["edge_indices"].append(idx)

        return [
            {
                "id1": info["id1"],
                "id2": info["id2"],
                "edges": correspondences.loc[info["edge_indices"]],
            }
            for info in clusters.values()
        ]

    def _summarize_large_clusters(
        self,
        correspondences: pd.DataFrame,
        left_df: Optional[pd.DataFrame],
        right_df: Optional[pd.DataFrame],
        left_id_col: Optional[str],
        right_id_col: Optional[str],
    ) -> List[Dict[str, Any]]:
        clusters = self._build_clusters(correspondences)
        if not clusters:
            return []

        def cluster_size(c):
            return len(c["id1"]) + len(c["id2"])

        large_clusters = [
            c for c in clusters if cluster_size(c) >= self.large_cluster_min_size
        ]
        large_clusters.sort(key=cluster_size, reverse=True)
        summaries = []

        for idx, cluster in enumerate(
            large_clusters[: self.max_large_clusters_reported], start=1
        ):
            edges = cluster["edges"]
            if len(edges) > self.max_edges_per_cluster:
                edges = edges.sample(self.max_edges_per_cluster, random_state=42)
            size_left = len(cluster["id1"])
            size_right = len(cluster["id2"])
            total_edges = len(edges)
            density = (
                total_edges / (size_left * size_right)
                if size_left and size_right
                else 0.0
            )

            left_counts = (
                edges["id1"].value_counts()
                if "id1" in edges.columns
                else pd.Series(dtype=int)
            )
            right_counts = (
                edges["id2"].value_counts()
                if "id2" in edges.columns
                else pd.Series(dtype=int)
            )

            def top_hubs(counts: pd.Series) -> List[Dict[str, Any]]:
                hubs = []
                for entity, count in counts.head(self.max_hubs_reported).items():
                    avg_score = (
                        float(edges.loc[edges[counts.name] == entity, "score"].mean())
                        if "score" in edges.columns
                        else None
                    )
                    hubs.append(
                        {
                            "id": str(entity),
                            "matches": int(count),
                            "avg_score": (
                                round(avg_score, 4) if avg_score is not None else None
                            ),
                        }
                    )
                return hubs

            left_counts.name = "id1"
            right_counts.name = "id2"

            top_left = top_hubs(left_counts)
            top_right = top_hubs(right_counts)
            common_cols = self._common_attribute_columns(
                left_df, right_df, left_id_col, right_id_col
            )

            for hub in top_left:
                hub["attribute_overlap"] = self._compute_hub_attribute_overlap(
                    hub["id"],
                    "id1",
                    edges,
                    left_df,
                    right_df,
                    left_id_col,
                    right_id_col,
                )
            for hub in top_right:
                hub["attribute_overlap"] = self._compute_hub_attribute_overlap(
                    hub["id"],
                    "id2",
                    edges,
                    left_df,
                    right_df,
                    left_id_col,
                    right_id_col,
                )

            hub_explanations = []
            for hub in top_left:
                overlaps = hub.get("attribute_overlap", [])
                if overlaps:
                    top_overlap = overlaps[0]
                    overlap_rate = top_overlap.get(
                        "match_rate", top_overlap.get("pct_within_5pct", 0.0)
                    )
                    if overlap_rate >= 0.8:
                        hub_explanations.append(
                            {
                                "side": "left",
                                "id": hub["id"],
                                "attribute": top_overlap.get("attribute"),
                                "overlap_rate": overlap_rate,
                            }
                        )
            for hub in top_right:
                overlaps = hub.get("attribute_overlap", [])
                if overlaps:
                    top_overlap = overlaps[0]
                    overlap_rate = top_overlap.get(
                        "match_rate", top_overlap.get("pct_within_5pct", 0.0)
                    )
                    if overlap_rate >= 0.8:
                        hub_explanations.append(
                            {
                                "side": "right",
                                "id": hub["id"],
                                "attribute": top_overlap.get("attribute"),
                                "overlap_rate": overlap_rate,
                            }
                        )

            ambiguous_ratio = (
                self._compute_ambiguous_ratio(edges)
                if "score" in edges.columns
                else 0.0
            )

            score_stats = {}
            if "score" in edges.columns and not edges["score"].dropna().empty:
                score_series = edges["score"].dropna()
                score_stats = {
                    "mean": round(float(score_series.mean()), 4),
                    "std": round(float(score_series.std()), 4),
                    "min": round(float(score_series.min()), 4),
                    "max": round(float(score_series.max()), 4),
                }

            reasons = []
            if density >= 0.6:
                reasons.append(
                    "Dense bipartite structure suggests overly broad blocking or highly generic attributes."
                )
            if left_counts.max() if len(left_counts) else 0 >= max(3, size_right * 0.6):
                reasons.append(
                    "A left-side hub entity matches many right entities, indicating ambiguity."
                )
            if (
                right_counts.max()
                if len(right_counts)
                else 0 >= max(3, size_left * 0.6)
            ):
                reasons.append(
                    "A right-side hub entity matches many left entities, indicating ambiguity."
                )
            if (
                score_stats
                and score_stats.get("std", 0) <= 0.05
                and score_stats.get("mean", 0) >= 0.8
            ):
                reasons.append(
                    "Scores are uniformly high, suggesting a dominant attribute or overly permissive matching."
                )
            if not reasons:
                reasons.append(
                    "Cluster size is large with mixed match quality; review dominant attributes and hub entities."
                )

            attribute_agreement = []
            if (
                left_df is not None
                and right_df is not None
                and left_id_col in left_df.columns
                and right_id_col in right_df.columns
            ):
                left_subset = left_df[left_df[left_id_col].isin(cluster["id1"])]
                right_subset = right_df[right_df[right_id_col].isin(cluster["id2"])]
                if not left_subset.empty and not right_subset.empty:
                    left_subset = left_subset.copy()
                    right_subset = right_subset.copy()
                    left_subset["__id1"] = left_subset[left_id_col]
                    right_subset["__id2"] = right_subset[right_id_col]

                    merged = edges.merge(
                        left_subset, left_on="id1", right_on="__id1", how="left"
                    )
                    merged = merged.merge(
                        right_subset,
                        left_on="id2",
                        right_on="__id2",
                        how="left",
                        suffixes=("_left", "_right"),
                    )

                    for col in common_cols:
                        left_col = f"{col}_left"
                        right_col = f"{col}_right"
                        if (
                            left_col not in merged.columns
                            or right_col not in merged.columns
                        ):
                            continue
                        series_left = merged[left_col]
                        series_right = merged[right_col]
                        both_present = series_left.notna() & series_right.notna()
                        if both_present.sum() == 0:
                            continue
                        if pd.api.types.is_numeric_dtype(
                            series_left
                        ) and pd.api.types.is_numeric_dtype(series_right):
                            left_num = pd.to_numeric(
                                series_left[both_present], errors="coerce"
                            )
                            right_num = pd.to_numeric(
                                series_right[both_present], errors="coerce"
                            )
                            valid = left_num.notna() & right_num.notna()
                            if valid.sum() == 0:
                                continue
                            diff = (left_num[valid] - right_num[valid]).abs()
                            mean_diff = float(diff.mean())
                            denom = right_num[valid].abs().replace(0, pd.NA)
                            rel = (diff / denom).dropna()
                            within_5pct = (
                                float((rel <= 0.05).mean()) if not rel.empty else 0.0
                            )
                            attribute_agreement.append(
                                {
                                    "attribute": col,
                                    "type": "numeric",
                                    "mean_abs_diff": round(mean_diff, 4),
                                    "pct_within_5pct": round(within_5pct, 4),
                                    "coverage": round(float(valid.mean()), 4),
                                }
                            )
                        else:
                            left_norm = (
                                series_left[both_present]
                                .astype(str)
                                .str.lower()
                                .str.strip()
                            )
                            right_norm = (
                                series_right[both_present]
                                .astype(str)
                                .str.lower()
                                .str.strip()
                            )
                            match_rate = float((left_norm == right_norm).mean())
                            attribute_agreement.append(
                                {
                                    "attribute": col,
                                    "type": "text",
                                    "match_rate": round(match_rate, 4),
                                    "coverage": round(float(both_present.mean()), 4),
                                }
                            )
                    attribute_agreement.sort(
                        key=lambda x: x.get(
                            "match_rate", x.get("pct_within_5pct", 0.0)
                        ),
                        reverse=True,
                    )
                    attribute_agreement = attribute_agreement[
                        : self.max_attribute_reports
                    ]

            dominant_attributes = [
                attr
                for attr in attribute_agreement
                if (attr.get("match_rate") or attr.get("pct_within_5pct", 0.0)) >= 0.7
                and attr.get("coverage", 0) >= 0.3
            ][: self.max_attribute_reports]

            if dominant_attributes:
                reasons.append(
                    "High agreement on specific attributes suggests shared identifiers."
                )
            if hub_explanations:
                reasons.append(
                    "Hub entities share a dominant attribute value across many matches."
                )

            summaries.append(
                {
                    "cluster_rank": idx,
                    "size_left": size_left,
                    "size_right": size_right,
                    "total_nodes": size_left + size_right,
                    "total_edges": total_edges,
                    "density": round(density, 4),
                    "top_left_hubs": top_left,
                    "top_right_hubs": top_right,
                    "ambiguous_ratio": round(ambiguous_ratio, 4),
                    "score_stats": score_stats,
                    "attribute_agreement": attribute_agreement,
                    "dominant_attributes": dominant_attributes,
                    "hub_explanations": hub_explanations,
                    "suspected_reasons": reasons,
                }
            )

        return summaries

    def _compute_multi_match_entities(
        self, correspondences: pd.DataFrame
    ) -> Dict[str, Any]:
        result = {"left": {}, "right": {}}
        if "id1" in correspondences.columns:
            counts = correspondences["id1"].value_counts()
            multi = counts[counts > 1]
            result["left"] = {
                "count": int(len(multi)),
                "top": [
                    {"id": str(k), "matches": int(v)}
                    for k, v in multi.head(self.max_hubs_reported).items()
                ],
            }
        if "id2" in correspondences.columns:
            counts = correspondences["id2"].value_counts()
            multi = counts[counts > 1]
            result["right"] = {
                "count": int(len(multi)),
                "top": [
                    {"id": str(k), "matches": int(v)}
                    for k, v in multi.head(self.max_hubs_reported).items()
                ],
            }
        return result

    def _deep_cluster_analysis(
        self,
        correspondences: pd.DataFrame,
        left_df: Optional[pd.DataFrame],
        right_df: Optional[pd.DataFrame],
        left_id_col: Optional[str],
        right_id_col: Optional[str],
    ) -> Dict[str, Any]:
        if correspondences.empty:
            return {"note": "No correspondences available for deep analysis."}

        return {
            "multi_match_entities": self._compute_multi_match_entities(correspondences),
            "large_clusters": self._summarize_large_clusters(
                correspondences,
                left_df,
                right_df,
                left_id_col,
                right_id_col,
            ),
        }

    def _compute_duplicate_stats(self, correspondences: pd.DataFrame) -> Dict[str, Any]:
        id1_counts = (
            correspondences["id1"].value_counts()
            if "id1" in correspondences.columns
            else pd.Series(dtype=int)
        )
        id2_counts = (
            correspondences["id2"].value_counts()
            if "id2" in correspondences.columns
            else pd.Series(dtype=int)
        )

        id1_one_to_many = float((id1_counts > 1).mean()) if len(id1_counts) else 0.0
        id2_one_to_many = float((id2_counts > 1).mean()) if len(id2_counts) else 0.0
        one_to_many_ratio = max(id1_one_to_many, id2_one_to_many)

        max_degree = 0
        if len(id1_counts) or len(id2_counts):
            max_degree = max(
                id1_counts.max() if len(id1_counts) else 0,
                id2_counts.max() if len(id2_counts) else 0,
            )

        ambiguous_ratio = 0.0
        if "score" in correspondences.columns and not correspondences.empty:
            ambiguous_ratio = self._compute_ambiguous_ratio(correspondences)

        return {
            "one_to_many_ratio": round(one_to_many_ratio, 4),
            "id1_one_to_many_ratio": round(id1_one_to_many, 4),
            "id2_one_to_many_ratio": round(id2_one_to_many, 4),
            "max_degree": int(max_degree),
            "ambiguous_ratio": round(ambiguous_ratio, 4),
        }

    def _compute_ambiguous_ratio(self, correspondences: pd.DataFrame) -> float:
        def ratio_for_side(col: str) -> float:
            if col not in correspondences.columns:
                return 0.0
            scores = pd.to_numeric(correspondences["score"], errors="coerce")
            grouped = (
                correspondences.assign(score_num=scores)
                .dropna(subset=["score_num"])
                .groupby(col)["score_num"]
                .apply(list)
            )
            ambiguous = 0
            candidates = 0
            for scores in grouped:
                if len(scores) < 2:
                    continue
                candidates += 1
                top_two = sorted(scores, reverse=True)[:2]
                if (top_two[0] - top_two[1]) <= self.ambiguous_match_margin:
                    ambiguous += 1
            return ambiguous / candidates if candidates else 0.0

        return max(ratio_for_side("id1"), ratio_for_side("id2"))

    def _compute_score_stats(self, correspondences: pd.DataFrame) -> Dict[str, Any]:
        if "score" not in correspondences.columns or correspondences.empty:
            return {}
        scores = pd.to_numeric(correspondences["score"], errors="coerce").dropna()
        if scores.empty:
            return {}
        return {
            "mean": round(float(scores.mean()), 4),
            "p50": round(float(scores.quantile(0.5)), 4),
            "p75": round(float(scores.quantile(0.75)), 4),
            "p90": round(float(scores.quantile(0.9)), 4),
        }

    def run(self, correspondences_paths: List[str]) -> Dict[str, Any]:
        """
        Loads correspondences, analyzes cluster quality, and returns an aggregated report with recommendations.
        """
        if not correspondences_paths:
            if self.verbose:
                print("No correspondence files provided for cluster analysis.")
            return {}

        all_reports = {}
        print("\n")
        print("===================")
        print(" CLUSTER ANALYSIS")
        print("===================")

        for path in correspondences_paths:
            if not os.path.exists(path):
                print(
                    f"    [!] Warning: Correspondences file not found at: {path}. Skipping."
                )
                continue

            correspondences = pd.read_csv(path)

            if self.verbose:
                print(f"[*] Analyzing clusters from {os.path.basename(path)}")

            left_name, right_name = self._parse_dataset_names_from_correspondence(path)
            left_df, left_id_col = self._resolve_dataset(left_name)
            right_df, right_id_col = self._resolve_dataset(right_name)

            duplicate_stats = self._compute_duplicate_stats(correspondences)
            score_stats = self._compute_score_stats(correspondences)

            file_output_dir = os.path.join(
                self.output_dir, os.path.splitext(os.path.basename(path))[0]
            )
            os.makedirs(file_output_dir, exist_ok=True)

            cluster_dist_df = EntityMatchingEvaluator.create_cluster_size_distribution(
                correspondences=correspondences, out_dir=file_output_dir
            )

            deep_analysis = self._deep_cluster_analysis(
                correspondences,
                left_df,
                right_df,
                left_id_col,
                right_id_col,
            )

            if cluster_dist_df.empty:
                if self.verbose:
                    print("    No clusters found to analyze.")
                recommendation = {
                    "diagnosis": "No clusters were generated from the correspondences.",
                    "recommended_strategy": "None",
                    "recommended_action": "None",
                    "adjustment_signals": {},
                }
            else:
                health = self._analyze_cluster_health(cluster_dist_df)
                if self.verbose:
                    print(f"    Total clusters: {health['total_clusters']}")
                    print(f"    Max cluster size: {health['max_cluster_size']}")
                    print("    Cluster Distribution:")
                    print(cluster_dist_df.to_string(index=False))
                    print(
                        f"    Small cluster ratio: {health['small_cluster_ratio']:.2%}"
                    )
                    print(
                        f"    Large cluster ratio: {health['large_cluster_ratio']:.2%}"
                    )
                    print(
                        f"    One-to-many ratio: {duplicate_stats.get('one_to_many_ratio', 0.0):.2%}"
                    )
                    if score_stats:
                        print(f"    Score stats: {score_stats}")

                recommendation = self._get_recommendation(
                    cluster_dist_df, duplicate_stats, score_stats
                )

                # If deep analysis found large clusters, force a config-update recommendation
                if (
                    deep_analysis.get("large_clusters")
                    and recommendation.get("is_healthy")
                    and not recommendation.get("meets_target")
                ):
                    recommendation["is_healthy"] = False
                    recommendation["recommended_action"] = "update_matching_config"
                    recommendation["diagnosis"] = (
                        "Deep analysis found large clusters; update matcher weights to reduce hub formation."
                    )

                if self.verbose:
                    print(f"    Diagnosis: {recommendation.get('diagnosis')}")
                    if recommendation.get("recommended_action"):
                        print(
                            f"    Recommended Action: {recommendation.get('recommended_action')}"
                        )
                    if recommendation.get("parameters"):
                        print(f"    Parameters: {recommendation.get('parameters')}")

            report = {
                "diagnosis": recommendation.get("diagnosis"),
                "recommended_strategy": recommendation.get("recommended_strategy"),
                "recommended_action": recommendation.get("recommended_action"),
                "parameters": recommendation.get("parameters"),
                "error": recommendation.get("error"),
                "health_metrics": recommendation.get("health_metrics"),
                "duplicate_stats": recommendation.get("duplicate_stats"),
                "score_stats": recommendation.get("score_stats"),
                "adjustment_signals": recommendation.get("adjustment_signals"),
                "meets_target": recommendation.get("meets_target"),
                "is_healthy": recommendation.get("is_healthy"),
                "cluster_distribution": cluster_dist_df.to_dict("records"),
                "deep_analysis": deep_analysis,
            }
            all_reports[os.path.basename(path)] = report

        if all_reports:
            unhealthy_files = [
                name
                for name, report in all_reports.items()
                if report.get("recommended_action") not in (None, "None")
                and not report.get("meets_target")
            ]
            recommendations = [
                report.get("recommended_action")
                for report in all_reports.values()
                if isinstance(report, dict)
                and report.get("recommended_action") not in (None, "None")
            ]
            overall_recommendation = (
                "update_matching_config" if recommendations else "None"
            )
            overall_parameters = {}
            overall_diagnosis = (
                "One or more correspondence files show a non-long-tail distribution."
                if unhealthy_files
                else "All correspondence files show healthy long-tail distributions."
            )
            all_reports["_overall"] = {
                "diagnosis": overall_diagnosis,
                "recommended_strategy": "None",
                "recommended_action": overall_recommendation,
                "parameters": overall_parameters,
                "unhealthy_files": unhealthy_files,
                "target_thresholds": {
                    "small_cluster_ratio": self.target_small_cluster_ratio,
                    "large_cluster_ratio": self.target_large_cluster_ratio,
                    "one_to_many_ratio": self.target_one_to_many_ratio,
                },
            }

        all_reports = self._json_sanitize(all_reports)

        report_path = os.path.join(self.output_dir, "cluster_analysis_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(all_reports, f, indent=4, ensure_ascii=False)

        if self.verbose:
            print(f"\n    Aggregated cluster analysis report saved to {report_path}")

        return all_reports
