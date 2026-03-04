import json
import os
import sys
import tempfile
import unittest

AGENTS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if AGENTS_ROOT not in sys.path:
    sys.path.insert(0, AGENTS_ROOT)

from helpers.run_report import write_agent_run_report


class RunReportTests(unittest.TestCase):
    def test_report_is_generated(self):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "output")
            os.makedirs(os.path.join(out, "results"), exist_ok=True)
            os.makedirs(os.path.join(out, "blocking-evaluation"), exist_ok=True)
            os.makedirs(os.path.join(out, "matching-evaluation"), exist_ok=True)
            os.makedirs(os.path.join(out, "normalization", "attempt_1"), exist_ok=True)
            os.makedirs(os.path.join(out, "profile"), exist_ok=True)

            results_file = os.path.join(out, "results", "run_20260303_120000.json")
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "evaluation_metrics": {"overall_accuracy": 0.7},
                        "sealed_final_test_metrics": {"overall_accuracy": 0.8},
                        "token_usage": {"total_tokens": 1000, "total_cost": 0.12},
                    },
                    f,
                )
            with open(os.path.join(out, "blocking-evaluation", "blocking_config.json"), "w", encoding="utf-8") as f:
                json.dump({"blocking_strategies": {"a_b": {"strategy": "semantic_similarity", "pair_completeness": 0.9}}}, f)
            with open(os.path.join(out, "matching-evaluation", "matching_config.json"), "w", encoding="utf-8") as f:
                json.dump({"matching_strategies": {"a_b": {"f1": 0.8}}}, f)
            with open(os.path.join(out, "normalization", "attempt_1", "normalization_report.json"), "w", encoding="utf-8") as f:
                json.dump({"status": "success", "failure_tags": ["ok"]}, f)

            report = write_agent_run_report(output_root=out, results_file=results_file)
            self.assertTrue(os.path.exists(report))
            with open(report, "r", encoding="utf-8") as f:
                body = f.read()
            self.assertIn("Agent Run Report", body)
            self.assertIn("Blocking Decisions", body)


if __name__ == "__main__":
    unittest.main()
