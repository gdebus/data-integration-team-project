import pandas as pd
import json
import os
import shutil
from typing import Dict, Any, List

from PyDI.entitymatching import EntityMatchingEvaluator
from PyDI.io import load_xml, load_parquet, load_csv
from langchain_core.messages import SystemMessage, HumanMessage


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
        self, llm, output_dir: str = "output/cluster-evaluation", verbose: bool = True
    ):
        self.llm = llm
        self.output_dir = output_dir
        self.verbose = verbose
        # Clean and recreate the output directory for a fresh run
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_llm_recommendation(
        self,
        cluster_stats: pd.DataFrame,
        total_clusters: int,
        max_cluster_size: int,
        large_cluster_examples: Dict[str, List] = None,
    ) -> Dict[str, Any]:
        """Prompts LLM to diagnose cluster issues and recommend a specific PyDI post-processing function."""

        system_prompt = """
        You are a world-class Data Integration expert specializing in the PyDI library. Your task is to analyze entity clustering results and recommend a concrete, programmatic solution using a specific PyDI post-clustering function if necessary.

        **CONTEXT:**
        The PyDI library offers several post-clustering functions in `PyDI.entitymatching` to refine correspondences:
        - `MaximumBipartiteMatching`: A more optimal way to enforce a 1-to-1 constraint by finding the matching with the maximum total similarity score.
        - `HierarchicalClusterer`: Can be used to split large, messy clusters by re-clustering them based on a distance metric and a threshold.
        - `StableMatching`: Enforces a 1-to-1 constraint by finding a stable matching based on ranked preferences, which is useful when the order of preference for matches is more important than a global similarity score.

        **YOUR TASK:**
        Based on the provided cluster statistics and record examples, you must:
        1.  **Diagnose the Problem:** Concisely state the most likely clustering issue (e.g., "over-matching", "under-clustering", "healthy distribution").
        2.  **Recommend a Strategy:** Recommend ONE specific PyDI function to apply. Your options are:
            - "MaximumBipartiteMatching"
            - "HierarchicalClusterer"
            - "StableMatching"
            - "None" (if the clusters are healthy and no action is needed).
        3.  **Provide Parameters:** If you recommend a strategy, provide the necessary parameters as a JSON object.
            - For `HierarchicalClusterer`, you MUST provide a `linkage_method` (e.g., 'single', 'average', 'complete') and a `threshold` (float between 0.0 and 1.0).
            - For all others, provide an empty JSON object `{}`.

        **RESPONSE FORMAT:**
        You must respond with ONLY a JSON object containing three keys: `diagnosis`, `recommended_strategy`, and `parameters`.

        **Example 1 (Healthy Clusters):**
        ```json
        {
          "diagnosis": "The cluster distribution shows a healthy long tail with a reasonable number of size-2 clusters and very few large clusters. The matching appears balanced.",
          "recommended_strategy": "None",
          "parameters": {}
        }
        ```

        **Example 2 (Optimal 1-to-1 Matching Needed):**
        ```json
        {
          "diagnosis": "The clusters contain complex many-to-many relationships where other approach might fail to find the best overall set of pairs. The similarity scores suggest that a globally optimal 1-to-1 assignment is required to maximize the quality of the final matches.",
          "recommended_strategy": "MaximumBipartiteMatching",
          "parameters": {}
        }
        ```

        **Example 3 (Messy Large Clusters):**
        ```json
        {
          "diagnosis": "A giant cluster of size 50 was found. The records within it show some similarity but are not all perfect matches, suggesting a need for refinement. The average similarity is likely low.",
          "recommended_strategy": "HierarchicalClusterer",
          "parameters": {
            "linkage_method": "average",
            "threshold": 0.85
          }
        }
        ```
        
        **Example 4 (Preference-based 1-to-1 Matching):**
        ```json
        {
          "diagnosis": "The clusters show many-to-many relationships, but the matching quality depends on ranked preferences rather than just similarity scores. A stable matching is needed to ensure that no two entities would prefer each other over their current match.",
          "recommended_strategy": "StableMatching",
          "parameters": {}
        }
        ```
        """

        large_cluster_section = ""
        if large_cluster_examples:
            large_cluster_section = f"""
            **Examples from Large Clusters:**
            {json.dumps(large_cluster_examples, indent=2)}
            """

        human_content = f"""
        Analyze the following clustering results and provide your recommendation.

        **Cluster Statistics:**
        - Total clusters: {total_clusters}
        - Maximum cluster size: {max_cluster_size}
        - Distribution:
          {cluster_stats.to_string()}
        
        {large_cluster_section}
        
        Provide your diagnosis and recommendation as a JSON object.
        """

        response = self.llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=human_content)]
        )
        response_text = (
            response.content if hasattr(response, "content") else str(response)
        )

        try:
            # Clean the response to get only the JSON
            json_str = response_text[
                response_text.find("{") : response_text.rfind("}") + 1
            ]
            return json.loads(json_str)
        except (json.JSONDecodeError, IndexError):
            return {
                "diagnosis": "Could not parse LLM response.",
                "recommended_strategy": "None",
                "parameters": {},
                "error": f"LLM response was not valid JSON: {response_text}",
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
                    "parameters": {},
                }
            else:
                total_clusters = int(cluster_dist_df["frequency"].sum())
                max_cluster_size = int(cluster_dist_df["cluster_size"].max())

                if self.verbose:
                    print(f"    Total clusters: {total_clusters}")
                    print(f"    Max cluster size: {max_cluster_size}")
                    print("    Cluster Distribution:")
                    print(cluster_dist_df.to_string(index=False))

                recommendation = self._get_llm_recommendation(
                    cluster_dist_df, total_clusters, max_cluster_size
                )

                if self.verbose:
                    print(f"    Diagnosis: {recommendation.get('diagnosis')}")
                    print(
                        f"    Recommended Strategy: {recommendation.get('recommended_strategy')}"
                    )
                    if recommendation.get("parameters"):
                        print(f"    Parameters: {recommendation.get('parameters')}")

            report = {
                "diagnosis": recommendation.get("diagnosis"),
                "recommended_strategy": recommendation.get("recommended_strategy"),
                "parameters": recommendation.get("parameters"),
                "error": recommendation.get("error"),
                "cluster_distribution": cluster_dist_df.to_dict("records"),
            }
            all_reports[os.path.basename(path)] = report

        report_path = os.path.join(self.output_dir, "cluster_analysis_report.json")
        with open(report_path, "w") as f:
            json.dump(all_reports, f, indent=4)

        if self.verbose:
            print(f"\n    Aggregated cluster analysis report saved to {report_path}")

        return all_reports
