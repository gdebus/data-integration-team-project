import os
import sys
import tempfile
import unittest

import pandas as pd

AGENTS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if AGENTS_ROOT not in sys.path:
    sys.path.insert(0, AGENTS_ROOT)

from blocking_tester import BlockingTester


class BlockingGuardrailTests(unittest.TestCase):
    def _write_csv(self, path: str, rows):
        pd.DataFrame(rows).to_csv(path, index=False)

    def test_guardrails_enforce_strategy_and_columns(self):
        with tempfile.TemporaryDirectory() as td:
            left = os.path.join(td, "left.csv")
            right = os.path.join(td, "right.csv")
            gold = os.path.join(td, "gold.csv")
            self._write_csv(left, [{"id": "l1", "name": "A", "genre": "x"}])
            self._write_csv(right, [{"id": "r1", "name": "A", "genre": "x"}])
            self._write_csv(gold, [{"id1": "l1", "id2": "r1"}])
            tester = BlockingTester(
                llm=None,
                datasets=[left, right],
                blocking_testsets={("left", "right"): gold},
                verbose=False,
            )
            analysis = tester._analyze_columns_for_pair("left", "right", "id")
            guardrails = tester._build_strategy_guardrails(analysis)
            out = tester._apply_strategy_guardrails(
                {"strategy": "ngram_blocking", "columns": ["missing_col"], "ngram_size": 2},
                guardrails,
                analysis["common_columns"],
            )
            self.assertIn(out["strategy"], guardrails["allowed_strategies"])
            self.assertTrue(out["columns"])

    def test_confidence_stop_for_pc(self):
        decision = BlockingTester._confidence_stop_for_pc(
            pc_history=[0.1, 0.101, 0.1015, 0.1012],
            best_pc=0.1015,
            threshold=0.9,
        )
        self.assertTrue(decision["stop"])


if __name__ == "__main__":
    unittest.main()
