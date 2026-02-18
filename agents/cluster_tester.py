import pandas as pd
import json
import os
import shutil
from typing import Dict, Any, List

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
        compact_output: bool = True,
        min_small_cluster_ratio: float = 0.75,
        max_large_cluster_ratio: float = 0.05,
        max_healthy_cluster_size: int = 15,
        max_one_to_many_ratio: float = 0.2,
    ):
        self.llm = llm
        self.output_dir = output_dir
        self.verbose = verbose
        self.compact_output = compact_output
        self.min_small_cluster_ratio = min_small_cluster_ratio
        self.max_large_cluster_ratio = max_large_cluster_ratio
        self.max_healthy_cluster_size = max_healthy_cluster_size
        self.max_one_to_many_ratio = max_one_to_many_ratio
        # Clean and recreate the output directory for a fresh run
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

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
        self, cluster_stats: pd.DataFrame, one_to_many_ratio: float
    ) -> Dict[str, Any]:
        """Recommend matching threshold adjustments when distribution is not long-tail."""
        health = self._analyze_cluster_health(cluster_stats)
        health["one_to_many_ratio"] = round(one_to_many_ratio, 4)

        is_one_to_many_ok = one_to_many_ratio <= self.max_one_to_many_ratio
        if health["is_long_tail"] and is_one_to_many_ok:
            diagnosis = (
                "The cluster distribution shows a healthy long-tail pattern with mostly "
                "small clusters and very few large clusters."
            )
            return {
                "diagnosis": diagnosis,
                "recommended_strategy": "None",
                "health_metrics": health,
                "is_healthy": True,
            }

        if not is_one_to_many_ok:
            diagnosis = (
                "Many entities map to multiple counterparts (one-to-many), which suggests "
                "ambiguous correspondences and over-matching."
            )
        else:
            diagnosis = (
                "The cluster distribution is not long-tail and indicates potential many-to-many "
                "matches or over-clustering."
            )

        return {
            "diagnosis": diagnosis,
            "recommended_strategy": "AdjustMatchingConfig",
            "parameters": {
                "threshold_delta": 0.05,
                "threshold_cap": 0.95,
                "reason": "Increase match thresholds to reduce over-matching and large clusters.",
            },
            "health_metrics": health,
            "is_healthy": False,
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
            id1_one_to_many = (
                float((id1_counts > 1).mean()) if len(id1_counts) > 0 else 0.0
            )
            id2_one_to_many = (
                float((id2_counts > 1).mean()) if len(id2_counts) > 0 else 0.0
            )
            one_to_many_ratio = max(id1_one_to_many, id2_one_to_many)

            file_output_dir = os.path.join(
                self.output_dir, os.path.splitext(os.path.basename(path))[0]
            )
            os.makedirs(file_output_dir, exist_ok=True)

            cluster_dist_df = EntityMatchingEvaluator.create_cluster_size_distribution(
                correspondences=correspondences, out_dir=file_output_dir
            )

            if cluster_dist_df.empty:
                if self.verbose:
                    print("    No clusters found to analyze.")
                recommendation = {
                    "diagnosis": "No clusters were generated from the correspondences.",
                    "recommended_strategy": "None",
                }
            else:
                health = self._analyze_cluster_health(cluster_dist_df)

                if self.verbose:
                    if self.compact_output:
                        print(
                            "    Metrics: "
                            f"clusters={health['total_clusters']}, "
                            f"max_size={health['max_cluster_size']}, "
                            f"small={health['small_cluster_ratio']:.2%}, "
                            f"large={health['large_cluster_ratio']:.2%}, "
                            f"one_to_many={one_to_many_ratio:.2%}"
                        )
                    else:
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
                        print(f"    One-to-many ratio: {one_to_many_ratio:.2%}")

                recommendation = self._get_recommendation(
                    cluster_dist_df, one_to_many_ratio
                )

                if self.verbose:
                    print(
                        f"    Recommended Strategy: "
                        f"{recommendation.get('recommended_strategy')}"
                    )
                    if not self.compact_output:
                        print(f"    Diagnosis: {recommendation.get('diagnosis')}")
                        if recommendation.get("parameters"):
                            print(f"    Parameters: {recommendation.get('parameters')}")

            report = {
                "diagnosis": recommendation.get("diagnosis"),
                "recommended_strategy": recommendation.get("recommended_strategy"),
                "parameters": recommendation.get("parameters"),
                "error": recommendation.get("error"),
                "health_metrics": recommendation.get("health_metrics"),
                "is_healthy": recommendation.get("is_healthy"),
                "one_to_many_ratio": round(one_to_many_ratio, 4),
                "cluster_distribution": cluster_dist_df.to_dict("records"),
            }
            all_reports[os.path.basename(path)] = report

        if all_reports:
            unhealthy_files = [
                name
                for name, report in all_reports.items()
                if report.get("recommended_strategy") not in (None, "None")
            ]
            overall_recommendation = (
                "AdjustMatchingConfig" if unhealthy_files else "None"
            )
            overall_diagnosis = (
                "One or more correspondence files show a non-long-tail distribution."
                if unhealthy_files
                else "All correspondence files show healthy long-tail distributions."
            )
            all_reports["_overall"] = {
                "diagnosis": overall_diagnosis,
                "recommended_strategy": overall_recommendation,
                "parameters": (
                    {
                        "threshold_delta": 0.05,
                        "threshold_cap": 0.95,
                        "reason": "Increase match thresholds to reduce over-matching and large clusters.",
                    }
                    if unhealthy_files
                    else {}
                ),
                "unhealthy_files": unhealthy_files,
            }

        report_path = os.path.join(self.output_dir, "cluster_analysis_report.json")
        with open(report_path, "w") as f:
            json.dump(all_reports, f, indent=4)

        if self.verbose:
            overall = all_reports.get("_overall", {}) if isinstance(all_reports, dict) else {}
            if isinstance(overall, dict) and overall:
                print(
                    "    Overall: "
                    f"recommended_strategy={overall.get('recommended_strategy', 'None')}, "
                    f"unhealthy_files={len(overall.get('unhealthy_files', []))}"
                )
            print(f"\n    Aggregated cluster analysis report saved to {report_path}")

        return all_reports
