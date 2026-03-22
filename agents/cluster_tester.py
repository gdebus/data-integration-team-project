"""Cluster quality analyser for entity-matching correspondences.

Produces rich, per-pair diagnostics that the pipeline prompt can surface
as *advisory evidence* — the LLM decides which post-clustering algorithm
(if any) to apply, guided by concrete metrics rather than a hardcoded rule.

Output: a JSON report saved to ``output/cluster-evaluation/cluster_analysis_report.json``
with per-file and overall sections containing:
  - cluster size distribution & health metrics
  - score statistics (mean, std, percentiles, gap analysis)
  - ambiguity metrics (one-to-many ratio, ambiguity ratio)
  - sample problematic clusters (largest components, most ambiguous entities)
  - per-pair strategy evidence with all 5 PyDI post-clustering options rated
"""

import json
import os
import shutil
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from PyDI.entitymatching import EntityMatchingEvaluator


# ---------------------------------------------------------------------------
# Strategy catalogue — every PyDI post-clustering option with its trade-offs
# ---------------------------------------------------------------------------
STRATEGY_CATALOGUE: Dict[str, Dict[str, str]] = {
    "MaximumBipartiteMatching": {
        "description": "Finds the maximum-weight one-to-one matching via the Hungarian algorithm.",
        "best_for": "High one-to-many ratio; need strict 1:1 correspondences while maximising total score.",
        "trade_off": "Drops valid matches when true cardinality is many-to-many.",
    },
    "StableMatching": {
        "description": "Gale-Shapley stable matching producing a stable one-to-one assignment.",
        "best_for": "Moderate one-to-many with preference-ordering semantics; avoids unstable swaps.",
        "trade_off": "May sacrifice global optimality for stability; proposer-side bias.",
    },
    "GreedyOneToOneMatchingAlgorithm": {
        "description": "Greedy one-to-one: iteratively picks the highest-scoring unmatched pair.",
        "best_for": "Fast one-to-one pruning when scores are well-separated (high score gap).",
        "trade_off": "Locally optimal; may miss better global assignments when scores are close.",
    },
    "HierarchicalClusterer": {
        "description": "Agglomerative clustering with configurable linkage and threshold.",
        "best_for": "Many-to-many relationships; grouping related entities into variable-size clusters.",
        "trade_off": "Does not enforce 1:1; requires tuning linkage_mode and min_similarity.",
    },
    "ConnectedComponentClusterer": {
        "description": "Groups all transitively connected entities into clusters.",
        "best_for": "Exploring the raw cluster structure; baseline before refinement.",
        "trade_off": "No pruning — every transitive link stays, producing large clusters if matching is noisy.",
    },
    "None": {
        "description": "No post-clustering. Use raw matcher output directly.",
        "best_for": "Already clean 1:1 correspondences with healthy long-tail distribution.",
        "trade_off": "Passes through any noise or many-to-many relationships from the matcher.",
    },
}


def _safe_percentile(series: pd.Series, q: float) -> float:
    """Compute a percentile, returning 0.0 on empty series."""
    if series.empty:
        return 0.0
    return float(np.percentile(series.dropna(), q))


# ---------------------------------------------------------------------------
# Core analysis helpers (pure functions — no LLM, no side-effects)
# ---------------------------------------------------------------------------

def compute_cluster_health(
    cluster_stats: pd.DataFrame,
    *,
    min_small_ratio: float = 0.75,
    max_large_ratio: float = 0.05,
    max_healthy_size: int = 15,
) -> Dict[str, Any]:
    """Compute cluster-size-distribution health metrics."""
    size_freq: Dict[int, int] = {}
    for _, row in cluster_stats.iterrows():
        size = int(row["cluster_size"])
        freq = int(row["frequency"])
        size_freq[size] = size_freq.get(size, 0) + freq

    total = sum(size_freq.values())
    if total == 0:
        return {
            "total_clusters": 0,
            "max_cluster_size": 0,
            "small_cluster_ratio": 0.0,
            "large_cluster_ratio": 0.0,
            "is_long_tail": True,
            "size_distribution": {},
        }

    max_size = max(size_freq.keys())
    small = sum(f for s, f in size_freq.items() if s <= 2)
    large = sum(f for s, f in size_freq.items() if s >= 6)
    small_ratio = small / total
    large_ratio = large / total

    is_long_tail = (
        small_ratio >= min_small_ratio
        and large_ratio <= max_large_ratio
        and max_size <= max_healthy_size
    )

    return {
        "total_clusters": total,
        "max_cluster_size": max_size,
        "small_cluster_ratio": round(small_ratio, 4),
        "large_cluster_ratio": round(large_ratio, 4),
        "is_long_tail": is_long_tail,
        "size_distribution": {str(k): v for k, v in sorted(size_freq.items())},
    }


def compute_score_metrics(correspondences: pd.DataFrame) -> Dict[str, Any]:
    """Derive score statistics and gap analysis from a correspondence DF."""
    if "score" not in correspondences.columns or correspondences["score"].dropna().empty:
        return {"available": False}

    scores = correspondences["score"].dropna().astype(float)
    return {
        "available": True,
        "count": int(len(scores)),
        "mean": round(float(scores.mean()), 4),
        "std": round(float(scores.std()), 4),
        "min": round(float(scores.min()), 4),
        "p25": round(_safe_percentile(scores, 25), 4),
        "median": round(float(scores.median()), 4),
        "p75": round(_safe_percentile(scores, 75), 4),
        "max": round(float(scores.max()), 4),
        # Gap between p75 and p25 — large gap means scores are well-separated
        "iqr": round(_safe_percentile(scores, 75) - _safe_percentile(scores, 25), 4),
        # Fraction of scores below 0.5 — potential low-confidence matches
        "low_confidence_ratio": round(float((scores < 0.5).mean()), 4),
    }


def compute_ambiguity_metrics(correspondences: pd.DataFrame) -> Dict[str, Any]:
    """Quantify one-to-many and ambiguity in correspondences."""
    result: Dict[str, Any] = {}

    for id_col in ("id1", "id2"):
        if id_col not in correspondences.columns:
            result[id_col] = {"one_to_many_ratio": 0.0, "max_fanout": 0}
            continue
        counts = correspondences[id_col].value_counts()
        if counts.empty:
            result[id_col] = {"one_to_many_ratio": 0.0, "max_fanout": 0}
            continue
        multi = int((counts > 1).sum())
        result[id_col] = {
            "one_to_many_ratio": round(float(multi / len(counts)), 4),
            "max_fanout": int(counts.max()),
            "mean_fanout": round(float(counts.mean()), 2),
            "entities_with_multiple_matches": multi,
            "total_unique_entities": int(len(counts)),
        }

    # Composite one-to-many: max across both sides
    r1 = result.get("id1", {}).get("one_to_many_ratio", 0.0)
    r2 = result.get("id2", {}).get("one_to_many_ratio", 0.0)
    result["one_to_many_ratio"] = round(max(r1, r2), 4)

    # Ambiguity ratio: for entities with >1 match, how close are the top-2 scores?
    if "score" in correspondences.columns:
        ambiguous_count = 0
        checked = 0
        for id_col in ("id1", "id2"):
            if id_col not in correspondences.columns:
                continue
            grouped = correspondences.groupby(id_col)["score"]
            for _, group_scores in grouped:
                if len(group_scores) < 2:
                    continue
                checked += 1
                top2 = group_scores.nlargest(2).values
                gap = float(top2[0] - top2[1])
                if gap < 0.1:  # top-2 scores within 0.1 → ambiguous
                    ambiguous_count += 1
        result["ambiguity_ratio"] = round(ambiguous_count / checked, 4) if checked > 0 else 0.0
        result["ambiguous_entities"] = ambiguous_count
    else:
        result["ambiguity_ratio"] = None
        result["ambiguous_entities"] = None

    return result


def extract_sample_clusters(
    correspondences: pd.DataFrame,
    *,
    n_largest: int = 3,
    n_most_ambiguous: int = 3,
) -> Dict[str, Any]:
    """Extract sample problematic clusters for human/LLM inspection."""
    samples: Dict[str, Any] = {"largest_components": [], "most_ambiguous": []}

    # --- Largest connected components ---
    for id_col in ("id1", "id2"):
        if id_col not in correspondences.columns:
            continue
        counts = correspondences[id_col].value_counts()
        top_ids = counts.nlargest(n_largest).index.tolist()
        for entity_id in top_ids:
            mask = correspondences[id_col] == entity_id
            component = correspondences[mask]
            entry = {
                "entity_id": str(entity_id),
                "side": id_col,
                "fanout": int(len(component)),
            }
            if "score" in component.columns:
                scores = component["score"].dropna().astype(float)
                entry["scores"] = sorted(scores.tolist(), reverse=True)
                entry["score_range"] = [round(float(scores.min()), 4), round(float(scores.max()), 4)]
            samples["largest_components"].append(entry)

    # Deduplicate by entity_id and keep only the top N overall
    seen = set()
    unique_largest = []
    for e in sorted(samples["largest_components"], key=lambda x: x["fanout"], reverse=True):
        key = (e["entity_id"], e["side"])
        if key not in seen:
            seen.add(key)
            unique_largest.append(e)
    samples["largest_components"] = unique_largest[:n_largest]

    # --- Most ambiguous entities (smallest top-2 score gap) ---
    if "score" in correspondences.columns:
        ambig_entries = []
        for id_col in ("id1", "id2"):
            if id_col not in correspondences.columns:
                continue
            grouped = correspondences.groupby(id_col)["score"]
            for eid, group_scores in grouped:
                if len(group_scores) < 2:
                    continue
                top2 = group_scores.nlargest(2).values
                gap = float(top2[0] - top2[1])
                ambig_entries.append({
                    "entity_id": str(eid),
                    "side": id_col,
                    "top_scores": [round(float(s), 4) for s in top2],
                    "score_gap": round(gap, 4),
                    "fanout": int(len(group_scores)),
                })
        ambig_entries.sort(key=lambda x: x["score_gap"])
        samples["most_ambiguous"] = ambig_entries[:n_most_ambiguous]

    return samples


def rate_strategies(
    health: Dict[str, Any],
    ambiguity: Dict[str, Any],
    score_metrics: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Rate each post-clustering strategy based on evidence.

    Returns a dict of strategy_name -> {rating, rationale} where rating is
    one of: "recommended", "viable", "unsuitable".
    """
    one_to_many = ambiguity.get("one_to_many_ratio", 0.0)
    ambiguity_ratio = ambiguity.get("ambiguity_ratio")
    is_long_tail = health.get("is_long_tail", True)
    max_size = health.get("max_cluster_size", 1)
    large_ratio = health.get("large_cluster_ratio", 0.0)
    scores_available = score_metrics.get("available", False)
    iqr = score_metrics.get("iqr", 0.0) if scores_available else None

    ratings: Dict[str, Dict[str, Any]] = {}

    # --- None (no post-clustering) ---
    if is_long_tail and one_to_many <= 0.05:
        ratings["None"] = {"rating": "recommended", "rationale": "Distribution is healthy long-tail with negligible one-to-many."}
    elif is_long_tail and one_to_many <= 0.15:
        ratings["None"] = {"rating": "viable", "rationale": "Mostly healthy but some one-to-many relationships exist."}
    else:
        ratings["None"] = {"rating": "unsuitable", "rationale": f"One-to-many ratio ({one_to_many:.1%}) or cluster structure indicates noise."}

    # --- MaximumBipartiteMatching ---
    if one_to_many > 0.10:
        ratings["MaximumBipartiteMatching"] = {
            "rating": "recommended",
            "rationale": f"High one-to-many ({one_to_many:.1%}); MBM maximises total score while enforcing 1:1.",
        }
    elif one_to_many > 0.05:
        ratings["MaximumBipartiteMatching"] = {"rating": "viable", "rationale": "Moderate one-to-many; MBM would clean up some noise."}
    else:
        ratings["MaximumBipartiteMatching"] = {"rating": "viable", "rationale": "Low one-to-many; MBM is safe but may not change much."}

    # --- StableMatching ---
    if one_to_many > 0.10 and ambiguity_ratio is not None and ambiguity_ratio > 0.3:
        ratings["StableMatching"] = {
            "rating": "recommended",
            "rationale": f"High ambiguity ({ambiguity_ratio:.1%}) with one-to-many ({one_to_many:.1%}); stability avoids preference cycles.",
        }
    elif one_to_many > 0.05:
        ratings["StableMatching"] = {"rating": "viable", "rationale": "Moderate one-to-many; stable matching is a reasonable alternative to MBM."}
    else:
        ratings["StableMatching"] = {"rating": "viable", "rationale": "Low one-to-many makes stable matching unnecessary but harmless."}

    # --- GreedyOneToOneMatchingAlgorithm ---
    if iqr is not None and iqr > 0.15 and one_to_many > 0.05:
        ratings["GreedyOneToOneMatchingAlgorithm"] = {
            "rating": "recommended",
            "rationale": f"Scores are well-separated (IQR={iqr:.3f}) making greedy selection reliable.",
        }
    elif one_to_many > 0.05:
        ratings["GreedyOneToOneMatchingAlgorithm"] = {"rating": "viable", "rationale": "Greedy 1:1 is fast but may miss globally better assignments."}
    else:
        ratings["GreedyOneToOneMatchingAlgorithm"] = {"rating": "viable", "rationale": "Low one-to-many; greedy selection has little to prune."}

    # --- HierarchicalClusterer ---
    if max_size > 15 or large_ratio > 0.10:
        ratings["HierarchicalClusterer"] = {
            "rating": "recommended",
            "rationale": f"Large clusters detected (max={max_size}, large_ratio={large_ratio:.1%}); hierarchical clustering can split over-merged groups.",
        }
    elif not is_long_tail:
        ratings["HierarchicalClusterer"] = {"rating": "viable", "rationale": "Non-long-tail distribution; hierarchical clustering may improve structure."}
    else:
        ratings["HierarchicalClusterer"] = {"rating": "viable", "rationale": "Healthy distribution; hierarchical clustering adds complexity without clear benefit."}

    # --- ConnectedComponentClusterer ---
    ratings["ConnectedComponentClusterer"] = {
        "rating": "viable",
        "rationale": "Useful as a diagnostic baseline; rarely the best final choice for noisy data.",
    }

    return ratings


def build_evidence_summary(
    health: Dict[str, Any],
    ambiguity: Dict[str, Any],
    score_metrics: Dict[str, Any],
    strategy_ratings: Dict[str, Dict[str, Any]],
) -> str:
    """Build a human-readable evidence summary for the pipeline prompt."""
    lines = []

    # Health
    lines.append(f"Clusters: {health['total_clusters']}, max_size={health['max_cluster_size']}, "
                  f"small_ratio={health['small_cluster_ratio']:.1%}, large_ratio={health['large_cluster_ratio']:.1%}, "
                  f"long_tail={'yes' if health['is_long_tail'] else 'NO'}")

    # Ambiguity
    lines.append(f"One-to-many ratio: {ambiguity['one_to_many_ratio']:.1%}")
    if ambiguity.get("ambiguity_ratio") is not None:
        lines.append(f"Ambiguity ratio (top-2 gap < 0.1): {ambiguity['ambiguity_ratio']:.1%}")

    # Scores
    if score_metrics.get("available"):
        lines.append(f"Scores: mean={score_metrics['mean']:.3f}, std={score_metrics['std']:.3f}, "
                      f"IQR={score_metrics['iqr']:.3f}, low_conf_ratio={score_metrics['low_confidence_ratio']:.1%}")

    # Strategy ratings
    recommended = [name for name, r in strategy_ratings.items() if r["rating"] == "recommended"]
    if recommended:
        lines.append(f"Recommended strategies: {', '.join(recommended)}")
    else:
        lines.append("No strategy strongly recommended — distribution appears healthy.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main ClusterTester class
# ---------------------------------------------------------------------------

class ClusterTester:
    """Analyses entity-matching correspondences and produces per-pair
    strategy evidence for the pipeline LLM.

    Unlike the previous version, this class:
    - Does NOT accept or use an LLM — analysis is deterministic
    - Rates ALL 5 PyDI post-clustering algorithms per pair (not just MBM)
    - Computes score-based metrics (ambiguity, gap, percentiles)
    - Extracts sample problematic clusters for LLM inspection
    - Saves a rich JSON report for human review
    """

    def __init__(
        self,
        output_dir: str = "output/cluster-evaluation",
        verbose: bool = True,
        # Health thresholds
        min_small_cluster_ratio: float = 0.75,
        max_large_cluster_ratio: float = 0.05,
        max_healthy_cluster_size: int = 15,
    ):
        self.output_dir = output_dir
        self.verbose = verbose
        self.min_small_cluster_ratio = min_small_cluster_ratio
        self.max_large_cluster_ratio = max_large_cluster_ratio
        self.max_healthy_cluster_size = max_healthy_cluster_size

        # Clean and recreate output directory
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def _analyze_pair(self, path: str) -> Dict[str, Any]:
        """Full analysis of a single correspondence file."""
        correspondences = pd.read_csv(path)
        basename = os.path.basename(path)

        if self.verbose:
            print(f"[*] Analyzing: {basename} ({len(correspondences)} correspondences)")

        # --- Cluster size distribution (via PyDI) ---
        file_output_dir = os.path.join(
            self.output_dir, os.path.splitext(basename)[0]
        )
        os.makedirs(file_output_dir, exist_ok=True)

        cluster_dist_df = EntityMatchingEvaluator.create_cluster_size_distribution(
            correspondences=correspondences, out_dir=file_output_dir
        )

        if cluster_dist_df.empty:
            if self.verbose:
                print("    No clusters found.")
            return {
                "diagnosis": "No clusters were generated from the correspondences.",
                "is_healthy": None,
                "health_metrics": {},
                "score_metrics": {"available": False},
                "ambiguity_metrics": {},
                "strategy_ratings": {"None": {"rating": "recommended", "rationale": "No clusters to refine."}},
                "evidence_summary": "No clusters generated — nothing to analyze.",
                "sample_clusters": {},
                "cluster_distribution": [],
                "num_correspondences": int(len(correspondences)),
            }

        # --- Compute all metrics ---
        health = compute_cluster_health(
            cluster_dist_df,
            min_small_ratio=self.min_small_cluster_ratio,
            max_large_ratio=self.max_large_cluster_ratio,
            max_healthy_size=self.max_healthy_cluster_size,
        )
        score_metrics = compute_score_metrics(correspondences)
        ambiguity = compute_ambiguity_metrics(correspondences)
        samples = extract_sample_clusters(correspondences)
        strategy_ratings = rate_strategies(health, ambiguity, score_metrics)
        evidence = build_evidence_summary(health, ambiguity, score_metrics, strategy_ratings)

        # Determine best recommendation
        recommended = [name for name, r in strategy_ratings.items() if r["rating"] == "recommended"]
        # Prefer MBM among recommended, else first recommended, else None
        if "MaximumBipartiteMatching" in recommended:
            best = "MaximumBipartiteMatching"
        elif recommended:
            best = recommended[0]
        else:
            best = "None"

        is_healthy = best == "None"

        if self.verbose:
            print(f"    Health: long_tail={'yes' if health['is_long_tail'] else 'NO'}, "
                  f"clusters={health['total_clusters']}, max_size={health['max_cluster_size']}")
            print(f"    One-to-many: {ambiguity['one_to_many_ratio']:.1%}")
            if score_metrics.get("available"):
                print(f"    Scores: mean={score_metrics['mean']:.3f}, IQR={score_metrics['iqr']:.3f}")
            if ambiguity.get("ambiguity_ratio") is not None:
                print(f"    Ambiguity ratio: {ambiguity['ambiguity_ratio']:.1%}")
            print(f"    Best strategy: {best} ({'healthy' if is_healthy else 'needs refinement'})")
            if recommended and best != "None":
                print(f"    All recommended: {', '.join(recommended)}")

        return {
            "is_healthy": is_healthy,
            "recommended_strategy": best,
            "health_metrics": health,
            "score_metrics": score_metrics,
            "ambiguity_metrics": ambiguity,
            "strategy_ratings": strategy_ratings,
            "evidence_summary": evidence,
            "sample_clusters": samples,
            "cluster_distribution": cluster_dist_df.to_dict("records"),
            "num_correspondences": int(len(correspondences)),
        }

    def run(self, correspondences_paths: List[str]) -> Dict[str, Any]:
        """Analyze all correspondence files and produce an aggregated report.

        Returns a dict with per-file reports keyed by filename, plus an
        ``_overall`` summary.
        """
        if not correspondences_paths:
            if self.verbose:
                print("No correspondence files provided for cluster analysis.")
            return {}

        all_reports: Dict[str, Any] = {}
        print()

        for path in correspondences_paths:
            if not os.path.exists(path):
                print(f"    [!] Warning: Not found: {path}. Skipping.")
                continue
            try:
                report = self._analyze_pair(path)
                all_reports[os.path.basename(path)] = report
            except Exception as e:
                print(f"    [!] Error analyzing {path}: {e}")
                all_reports[os.path.basename(path)] = {
                    "error": str(e),
                    "is_healthy": None,
                    "recommended_strategy": "None",
                }

        # --- Build _overall summary ---
        if all_reports:
            unhealthy = [
                name for name, r in all_reports.items()
                if isinstance(r, dict) and r.get("recommended_strategy") not in (None, "None")
            ]

            # Aggregate strategy ratings across all pairs
            aggregated_ratings: Dict[str, List[str]] = {}
            for name, r in all_reports.items():
                if not isinstance(r, dict):
                    continue
                for strategy, rating_info in r.get("strategy_ratings", {}).items():
                    if strategy not in aggregated_ratings:
                        aggregated_ratings[strategy] = []
                    aggregated_ratings[strategy].append(rating_info.get("rating", "viable"))

            # A strategy is "overall recommended" if recommended in any pair
            overall_strategy_ratings: Dict[str, str] = {}
            for strategy, ratings_list in aggregated_ratings.items():
                if "recommended" in ratings_list:
                    overall_strategy_ratings[strategy] = "recommended"
                elif "unsuitable" in ratings_list:
                    overall_strategy_ratings[strategy] = "mixed"
                else:
                    overall_strategy_ratings[strategy] = "viable"

            # Pick overall recommended strategy
            if not unhealthy:
                overall_rec = "None"
                overall_diagnosis = "All correspondence files show healthy long-tail distributions."
            else:
                # Prefer whichever strategy is recommended in the most pairs
                rec_counts: Dict[str, int] = {}
                for name, r in all_reports.items():
                    if not isinstance(r, dict):
                        continue
                    rec = r.get("recommended_strategy")
                    if rec and rec != "None":
                        rec_counts[rec] = rec_counts.get(rec, 0) + 1
                overall_rec = max(rec_counts, key=rec_counts.get) if rec_counts else "MaximumBipartiteMatching"
                overall_diagnosis = (
                    f"{len(unhealthy)} of {len(all_reports)} pairs show cluster issues. "
                    f"Most frequently recommended: {overall_rec}."
                )

            # Collect all evidence summaries
            evidence_lines = []
            for name, r in all_reports.items():
                if isinstance(r, dict) and r.get("evidence_summary"):
                    evidence_lines.append(f"--- {name} ---\n{r['evidence_summary']}")

            all_reports["_overall"] = {
                "diagnosis": overall_diagnosis,
                "recommended_strategy": overall_rec,
                "strategy_ratings": overall_strategy_ratings,
                "unhealthy_files": unhealthy,
                "all_evidence": "\n\n".join(evidence_lines),
            }

            if self.verbose:
                print(f"\n    Overall: recommended={overall_rec}, "
                      f"unhealthy={len(unhealthy)}/{len(all_reports) - 1}")  # -1 for _overall

        # --- Save report ---
        report_path = os.path.join(self.output_dir, "cluster_analysis_report.json")
        with open(report_path, "w") as f:
            json.dump(all_reports, f, indent=2, default=str)

        if self.verbose:
            print(f"\n    Report saved to {report_path}")

        # Also save a human-readable summary
        summary_path = os.path.join(self.output_dir, "cluster_analysis_summary.txt")
        with open(summary_path, "w") as f:
            f.write("CLUSTER ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            for name, r in all_reports.items():
                if name == "_overall":
                    continue
                if not isinstance(r, dict):
                    continue
                f.write(f"File: {name}\n")
                f.write(f"  Correspondences: {r.get('num_correspondences', '?')}\n")
                f.write(f"  Healthy: {r.get('is_healthy', '?')}\n")
                f.write(f"  Recommended: {r.get('recommended_strategy', 'None')}\n")
                if r.get("evidence_summary"):
                    for line in r["evidence_summary"].split("\n"):
                        f.write(f"  {line}\n")
                # Strategy ratings table
                if r.get("strategy_ratings"):
                    f.write("  Strategy ratings:\n")
                    for strat, info in r["strategy_ratings"].items():
                        f.write(f"    {strat:40s} {info['rating']:14s} {info['rationale']}\n")
                # Sample clusters
                samples = r.get("sample_clusters", {})
                if samples.get("largest_components"):
                    f.write("  Largest components:\n")
                    for comp in samples["largest_components"]:
                        scores_str = ""
                        if comp.get("scores"):
                            scores_str = f" scores={[round(s, 3) for s in comp['scores'][:5]]}"
                        f.write(f"    {comp['side']}={comp['entity_id']} fanout={comp['fanout']}{scores_str}\n")
                if samples.get("most_ambiguous"):
                    f.write("  Most ambiguous entities:\n")
                    for amb in samples["most_ambiguous"]:
                        f.write(f"    {amb['side']}={amb['entity_id']} top_scores={amb['top_scores']} gap={amb['score_gap']:.4f}\n")
                f.write("\n")

            # Overall
            overall = all_reports.get("_overall", {})
            if overall:
                f.write("=" * 50 + "\n")
                f.write("OVERALL\n")
                f.write(f"  Diagnosis: {overall.get('diagnosis', '')}\n")
                f.write(f"  Recommended: {overall.get('recommended_strategy', 'None')}\n")
                if overall.get("strategy_ratings"):
                    f.write("  Aggregated strategy ratings:\n")
                    for strat, rating in overall["strategy_ratings"].items():
                        f.write(f"    {strat:40s} {rating}\n")

        if self.verbose:
            print(f"    Summary saved to {summary_path}")

        return all_reports
