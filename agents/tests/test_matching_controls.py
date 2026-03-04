import os
import sys
import tempfile
import unittest

import pandas as pd

AGENTS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if AGENTS_ROOT not in sys.path:
    sys.path.insert(0, AGENTS_ROOT)

from matching_tester import MatchingTester


class MatchingControlTests(unittest.TestCase):
    def _write_csv(self, path: str, rows):
        pd.DataFrame(rows).to_csv(path, index=False)

    def test_blocking_quality_gate_skips_pair(self):
        with tempfile.TemporaryDirectory() as td:
            left_path = os.path.join(td, "left.csv")
            right_path = os.path.join(td, "right.csv")
            gold_path = os.path.join(td, "gold.csv")
            self._write_csv(left_path, [{"id": "l1", "name": "a"}])
            self._write_csv(right_path, [{"id": "r1", "name": "a"}])
            self._write_csv(gold_path, [{"id1": "l1", "id2": "r1", "label": 1}])

            tester = MatchingTester(
                llm=None,
                datasets=[left_path, right_path],
                matching_testsets={("left", "right"): gold_path},
                blocking_config={
                    "pc_threshold": 0.9,
                    "blocking_strategies": {
                        "left_right": {"pair_completeness": 0.2, "strategy": "token_blocking", "columns": ["name"]}
                    },
                },
                verbose=False,
                matcher_mode="rule_based",
            )
            skip, info = tester._should_skip_pair_for_blocking_quality("left", "right")
            self.assertTrue(skip)
            self.assertAlmostEqual(info["threshold"], 0.9)

    def test_proxy_sampling_caps_candidate_rows(self):
        with tempfile.TemporaryDirectory() as td:
            left_path = os.path.join(td, "left.csv")
            right_path = os.path.join(td, "right.csv")
            gold_path = os.path.join(td, "gold.csv")
            self._write_csv(left_path, [{"id": "l1", "name": "a"}])
            self._write_csv(right_path, [{"id": "r1", "name": "a"}])
            self._write_csv(gold_path, [{"id1": "l1", "id2": "r1", "label": 1}])
            tester = MatchingTester(
                llm=None,
                datasets=[left_path, right_path],
                matching_testsets={("left", "right"): gold_path},
                blocking_config={},
                verbose=False,
            )
            candidates = pd.DataFrame({"id1": range(1000), "id2": range(1000)})
            sampled = tester._sample_candidate_pairs(candidates, max_rows=123)
            self.assertEqual(len(sampled), 123)

    def test_confidence_stop_triggers_on_low_variance(self):
        decision = MatchingTester._confidence_stop_for_f1(
            f1_history=[0.10, 0.102, 0.101, 0.1005],
            best_f1=0.102,
            target_f1=0.75,
            epsilon=0.003,
        )
        self.assertTrue(decision["stop"])


if __name__ == "__main__":
    unittest.main()
