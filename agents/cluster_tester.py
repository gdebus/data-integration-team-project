from PyDI.entitymatching import EntityMatchingEvaluator
import pandas as pd
import json
import os
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Dict, Any, List


class ClusterTester:
    """
    Analyzes clusters from entity matching correspondences to assess quality.
    """

    def __init__(
        self, llm, output_dir: str = "output/cluster-evaluation", verbose: bool = True
    ):
        self.llm = llm
        self.output_dir = output_dir
        self.verbose = verbose
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_llm_analysis(
        self, cluster_stats: pd.DataFrame, total_clusters: int, max_cluster_size: int
    ) -> Dict[str, str]:
        """Prompts the LLM with cluster statistics to get an analysis."""

        system_prompt = """
        You are a Quality Control Inspector for a data integration pipeline. Your task is to analyze the quality of entity clusters and provide a detailed analysis.
        
        You will be given cluster size distribution statistics. Based on these stats, you must:
        1. Provide a concise analysis explaining the quality of the clusters. Describe how the clusters look and whether they indicate good or bad matching (e.g., over-matching, under-matching, or balanced).
        2. Provide concrete suggestions for what to fix in the matching logic if the quality is poor (e.g., "increase matching threshold", "add more restrictive comparators for noisy columns like 'address'"). If the quality is good, suggestions can be empty.

        **Criteria Explanation:**
        -   **The Long Tail:** A good distribution often has ~42% of clusters at size 2, with frequency decreasing as size increases. Clusters of size 10+ should be rare (~13% or less). A high percentage of clusters of size 1 means no overlap was found for many records.
        -   **Giant Clusters:** One or more single clusters containing a very large number of entities is a red flag for matching errors (over-matching or transitive closure issues).
        -   **Too many small clusters:** A very high percentage of clusters of size 1 or 2 might indicate under-matching (thresholds too high, restrictive comparators).
        
        Respond with ONLY a JSON OBJECT with "analysis" and "suggestions" keys.
        Example for a good case:
        {
          "analysis": "The cluster distribution shows a healthy long tail. Over 45% of clusters are of size 2, and large clusters are rare, which indicates precise matching.",
          "suggestions": ""
        }
        Example for a bad case:
        {
          "analysis": "There are very few small clusters (only 10% are size 2) and a large number of medium-sized clusters (size 5-10). This indicates over-matching. A giant cluster of size 500 is also present.",
          "suggestions": "The matching threshold is likely too low. Increase the threshold in the RuleBasedMatcher. The giant cluster may be due to a false positive match on a common attribute; investigate comparators for columns with low cardinality."
        }
        """

        human_content = f"""
        Here are the cluster size distribution statistics:
        
        Total number of clusters: {total_clusters}
        Maximum cluster size found: {max_cluster_size}
        
        Distribution:
        {cluster_stats.to_string()}
        
        Based on these statistics, provide your analysis as a JSON object.
        """

        response = self.llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=human_content)]
        )

        response_text = (
            response.content if hasattr(response, "content") else str(response)
        )

        try:
            # Simple parsing of JSON from potentially messy LLM output
            json_str = response_text[
                response_text.find("{") : response_text.rfind("}") + 1
            ]
            parsed_json = json.loads(json_str)
            return parsed_json

        except (json.JSONDecodeError, IndexError):
            return {
                "analysis": "Could not parse LLM response.",
                "suggestions": "LLM response was not valid JSON. The raw response was: "
                + response_text,
            }

    def run(self, correspondences_paths: List[str]) -> Dict[str, Any]:
        """
        Loads correspondences from multiple files, analyzes cluster quality for each,
        and returns an aggregated report.
        """
        if not correspondences_paths:
            if self.verbose:
                print("No correspondence files provided for cluster analysis.")
            return {}

        all_reports = {}

        for path in correspondences_paths:
            if not os.path.exists(path):
                print(
                    f"    [!] Warning: Correspondences file not found at: {path}. Skipping."
                )
                continue

            correspondences = pd.read_csv(path)

            if self.verbose:
                print(f"[*] Analyzing clusters from {path}")

            filename = os.path.basename(path)
            file_output_dir = os.path.join(
                self.output_dir, os.path.splitext(filename)[0]
            )
            os.makedirs(file_output_dir, exist_ok=True)

            cluster_dist_df = EntityMatchingEvaluator.create_cluster_size_distribution(
                correspondences=correspondences, out_dir=file_output_dir
            )

            if cluster_dist_df.empty:
                if self.verbose:
                    print("    No clusters found to analyze.")
                report = {
                    "analysis": "No clusters found.",
                    "suggestions": "",
                    "total_clusters": 0,
                    "max_cluster_size": 0,
                    "cluster_distribution": [],
                }
            else:
                total_clusters = int(cluster_dist_df["frequency"].sum())
                max_cluster_size = int(cluster_dist_df["cluster_size"].max())

                if self.verbose:
                    print(f"    Total clusters: {total_clusters}")
                    print(f"    Max cluster size: {max_cluster_size}")
                    print("    Cluster Distribution:")
                    print(cluster_dist_df)

                analysis_result = self._get_llm_analysis(
                    cluster_dist_df, total_clusters, max_cluster_size
                )

                if self.verbose:
                    print(
                        f"    Analysis for {filename}: {analysis_result.get('analysis')}"
                    )

                report = {
                    "analysis": analysis_result.get("analysis"),
                    "suggestions": analysis_result.get("suggestions"),
                    "total_clusters": total_clusters,
                    "max_cluster_size": max_cluster_size,
                    "cluster_distribution": cluster_dist_df.to_dict("records"),
                }

            all_reports[filename] = report

        report_path = os.path.join(self.output_dir, "cluster_analysis_report.json")
        with open(report_path, "w") as f:
            json.dump(all_reports, f, indent=2)

        if self.verbose:
            print(f"    Aggregated cluster analysis report saved to {report_path}")

        return all_reports
