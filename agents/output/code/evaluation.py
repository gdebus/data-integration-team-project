
import pandas as pd
import json

from PyDI.io import load_xml
from PyDI.fusion import (
    DataFusionStrategy,
    longest_string,
    union,
    DataFusionEvaluator,
)
from PyDI.fusion import (
    tokenized_match,
    year_only_match,
    set_equality_match,
    numeric_tolerance_match,
)


# -------------------------------------------------
# Recreate the fusion strategy used in the pipeline
# -------------------------------------------------

def prefer_discogs_then_musicbrainz(values_with_provenance):
    priority = ["discogs", "musicbrainz"]
    for source in priority:
        for v, prov in values_with_provenance:
            if prov == source and v not in (None, "", []):
                return v
    return longest_string([v for v, _ in values_with_provenance])


def prefer_musicbrainz_then_discogs(values_with_provenance):
    priority = ["musicbrainz", "discogs"]
    for source in priority:
        for v, prov in values_with_provenance:
            if prov == source and v not in (None, "", []):
                return v
    return longest_string([v for v, _ in values_with_provenance])


def average_duration(values_with_provenance):
    nums = []
    for v, _ in values_with_provenance:
        if v in (None, "", []):
            continue
        try:
            num = float(v)
            if num > 0:
                nums.append(num)
        except Exception:
            continue
    if not nums:
        return longest_string([v for v, _ in values_with_provenance])
    return str(int(round(sum(nums) / len(nums))))


def prefer_discogs_tracks_then_union(values_with_provenance):
    ordered_sources = ["discogs", "musicbrainz", "lastfm"]
    for src in ordered_sources:
        vals = [
            v
            for v, prov in values_with_provenance
            if prov == src and v not in (None, "", [])
        ]
        if vals:
            return vals[0]
    raw_vals = [v for v, _ in values_with_provenance]
    return union(raw_vals)


def build_fusion_strategy() -> DataFusionStrategy:
    strategy = DataFusionStrategy("music_release_fusion_strategy")

    # Core fields
    strategy.add_attribute_fuser("name", prefer_discogs_then_musicbrainz)
    strategy.add_attribute_fuser("artist", prefer_discogs_then_musicbrainz)
    strategy.add_attribute_fuser("release-date", prefer_musicbrainz_then_discogs)
    strategy.add_attribute_fuser("release-country", prefer_musicbrainz_then_discogs)
    strategy.add_attribute_fuser("label", prefer_discogs_then_musicbrainz)
    strategy.add_attribute_fuser("genre", prefer_discogs_then_musicbrainz)
    strategy.add_attribute_fuser("duration", average_duration)

    # Track-related fields
    strategy.add_attribute_fuser("tracks_track_name", prefer_discogs_tracks_then_union)
    strategy.add_attribute_fuser(
        "tracks_track_position", prefer_discogs_then_musicbrainz
    )
    strategy.add_attribute_fuser(
        "tracks_track_duration", prefer_musicbrainz_then_discogs
    )

    # Attach evaluation functions matching the music test set
    strategy.add_evaluation_function("name", tokenized_match)
    strategy.add_evaluation_function("artist", tokenized_match)
    strategy.add_evaluation_function("duration", numeric_tolerance_match)
    strategy.add_evaluation_function("release-date", year_only_match)
    strategy.add_evaluation_function("release-country", tokenized_match)
    strategy.add_evaluation_function("label", tokenized_match)
    strategy.add_evaluation_function("tracks_track_name", set_equality_match)

    return strategy


def main():
    # -------------------------
    # Load fused output & gold
    # -------------------------
    fused = pd.read_csv(
        "output/data_fusion/fusion_rb_standard_blocker.csv"
    )

    fusion_test_set = load_xml(
        "input/testsets/music/test_set.xml",
        name="fusion_test_set",
        nested_handling="aggregate",
    )

    # --------------------------------
    # Build strategy & evaluator
    # --------------------------------
    strategy = build_fusion_strategy()

    evaluator = DataFusionEvaluator(
        strategy,
        debug=True,
        debug_file="output/pipeline_evaluation/debug_fusion_eval.jsonl",
        debug_format="json",
    )

    # -------------------------
    # Run evaluation
    # -------------------------
    evaluation_results = evaluator.evaluate(
        fused_df=fused,
        fused_id_column="_id",   # ID column in fused output
        gold_df=fusion_test_set,
        gold_id_column="id",     # ID column in gold test set
    )

    # -------------------------
    # Print structured metrics
    # -------------------------
    # You can adapt this to print specific keys as needed
    print(json.dumps(evaluation_results, indent=4))

    # -------------------------
    # Write output JSON (required)
    # -------------------------
    evaluation_output = "output/pipeline_evaluation/pipeline_evaluation.json"
    with open(evaluation_output, "w") as f:
        json.dump(evaluation_results, f, indent=4)


if __name__ == "__main__":
    main()
