import os
import sys
import unittest

import pandas as pd

AGENTS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if AGENTS_ROOT not in sys.path:
    sys.path.insert(0, AGENTS_ROOT)

from helpers.normalization_orchestrator import _compute_shadow_normalization_precheck
from helpers.normalization_orchestrator import _run_directive_ablation_precheck


class NormalizationShadowPrecheckTests(unittest.TestCase):
    def test_blocks_when_no_overlap_and_no_pressure(self):
        def loader(_path):
            return pd.DataFrame({"id": ["a"], "name": ["x"]})

        out = _compute_shadow_normalization_precheck(
            dataset_paths=["a.csv"],
            load_dataset_fn=loader,
            directives={"list_columns": ["tracks_track_name"]},
            metrics={"name_accuracy": 0.95, "name_count": 10},
            auto_diagnostics={"debug_reason_ratios": {}},
        )
        self.assertFalse(out["allow"])

    def test_allows_when_overlap_and_pressure_are_high(self):
        def loader(_path):
            return pd.DataFrame({"id": ["a"], "tracks_track_name": ["a|b"], "genre": ["x|y"]})

        out = _compute_shadow_normalization_precheck(
            dataset_paths=["a.csv"],
            load_dataset_fn=loader,
            directives={"list_columns": ["tracks_track_name", "genre"], "text_columns": ["tracks_track_name"]},
            metrics={"tracks_track_name_accuracy": 0.10, "tracks_track_name_count": 25, "genre_accuracy": 0.20, "genre_count": 25},
            auto_diagnostics={"debug_reason_ratios": {"list_format_mismatch": 0.7, "value_mismatch": 0.2}},
        )
        self.assertTrue(out["allow"])

    def test_ablation_selects_positive_subset(self):
        def loader(_path):
            return pd.DataFrame({"id": ["a"], "tracks_track_name": ["a|b"], "genre": ["x|y"]})

        out = _run_directive_ablation_precheck(
            dataset_paths=["a.csv"],
            load_dataset_fn=loader,
            directives={
                "list_columns": ["tracks_track_name", "genre"],
                "lowercase_columns": ["label"],
                "country_columns": [],
                "text_columns": ["tracks_track_name"],
            },
            metrics={"tracks_track_name_accuracy": 0.10, "tracks_track_name_count": 20},
            auto_diagnostics={"debug_reason_ratios": {"list_format_mismatch": 0.8}},
        )
        self.assertIn("selected_keys", out)
        self.assertTrue(len(out["selected_keys"]) >= 1)


if __name__ == "__main__":
    unittest.main()
