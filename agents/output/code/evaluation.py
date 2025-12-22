
import pandas as pd
import json

from PyDI.io import load_xml
from PyDI.fusion import (
    DataFusionStrategy,
    longest_string,
    union,
    most_recent,
    prefer_higher_trust,
)
from PyDI.fusion import (
    DataFusionEvaluator,
    tokenized_match,
    year_only_match,
    set_equality_match,
    numeric_tolerance_match,
)

import numpy as np
import re


# ----------------------------
# Helpers (reuse fusion logic)
# ----------------------------

def normalize_country(c):
    if pd.isna(c):
        return c
    c = str(c).strip()
    mapping = {
        "uk": "United Kingdom of Great Britain and Northern Ireland",
        "u.k.": "United Kingdom of Great Britain and Northern Ireland",
        "united kingdom": "United Kingdom of Great Britain and Northern Ireland",
        "england": "United Kingdom of Great Britain and Northern Ireland",
    }
    key = c.lower()
    return mapping.get(key, c)


# Source trust scores: musicbrainz > discogs > lastfm
source_trust = {
    "musicbrainz": 0.9,
    "discogs": 0.8,
    "lastfm": 0.6,
}


def prefer_musicbrainz_then_discogs(values, provenance):
    return prefer_higher_trust(values, provenance, trust_scores=source_trust)


def fuse_release_country(values, provenance):
    fused = prefer_higher_trust(values, provenance, trust_scores=source_trust)
    return normalize_country(fused)


def fuse_duration(values, provenance):
    chosen = prefer_higher_trust(values, provenance, trust_scores=source_trust)
    return chosen


# ----------------------------
# Load fused output & gold set
# ----------------------------

fused = pd.read_csv("output/data_fusion/fusion_rb_standard_blocker.csv")

fusion_test_set = load_xml(
    "input/testsets/music/test_set.xml",
    name="fusion_test_set",
    nested_handling="aggregate",
)

# ----------------------------
# Recreate the fusion strategy
# ----------------------------

strategy = DataFusionStrategy("music_release_fusion_strategy")

# Attribute fusers (must match integration pipeline)
strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("artist", prefer_musicbrainz_then_discogs)
strategy.add_attribute_fuser("release-date", most_recent)
strategy.add_attribute_fuser("release-country", fuse_release_country)
strategy.add_attribute_fuser("duration", fuse_duration)
strategy.add_attribute_fuser("label", prefer_musicbrainz_then_discogs)
strategy.add_attribute_fuser("genre", union)
strategy.add_attribute_fuser("tracks_track_name", union)
strategy.add_attribute_fuser("tracks_track_position", union)
strategy.add_attribute_fuser("tracks_track_duration", union)

# ----------------------------
# Configure evaluation functions
# ----------------------------

strategy.add_evaluation_function("name", tokenized_match)
strategy.add_evaluation_function("artist", tokenized_match)
strategy.add_evaluation_function("duration", numeric_tolerance_match)
strategy.add_evaluation_function("release-date", year_only_match)
strategy.add_evaluation_function("release-country", tokenized_match)
strategy.add_evaluation_function("label", tokenized_match)
strategy.add_evaluation_function("tracks_track_name", set_equality_match)

# ----------------------------
# Run evaluation
# ----------------------------

evaluator = DataFusionEvaluator(
    strategy,
    debug=True,
    debug_file="output/pipeline_evaluation/debug_fusion_eval.jsonl",
    debug_format="json",
)

evaluation_results = evaluator.evaluate(
    fused_df=fused,
    fused_id_column="_id",
    gold_df=fusion_test_set,
    gold_id_column="id",
)

# ----------------------------
# Write required JSON output
# ----------------------------

evaluation_output = "output/pipeline_evaluation/pipeline_evaluation.json"
with open(evaluation_output, "w") as f:
    json.dump(evaluation_results, f, indent=4)

# ----------------------------
# Print structured metrics
# ----------------------------

# The exact structure of evaluation_results depends on PyDI, but commonly:
# {
#   "overall": {...},
#   "per_attribute": {...},
#   ...
# }
print("=== Overall Evaluation ===")
overall = evaluation_results.get("overall", {})
for k, v in overall.items():
    print(f"{k}: {v}")

print("\n=== Per-Attribute Evaluation ===")
per_attr = evaluation_results.get("per_attribute", {})
for attr, metrics in per_attr.items():
    print(f"\nAttribute: {attr}")
    for m, val in metrics.items():
        print(f"  {m}: {val}")
