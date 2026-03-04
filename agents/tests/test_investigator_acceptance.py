import os
import sys
import unittest

AGENTS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if AGENTS_ROOT not in sys.path:
    sys.path.insert(0, AGENTS_ROOT)

from helpers.investigator_acceptance import (
    build_normalization_gate_request,
    create_pending_normalization_acceptance,
    evaluate_pending_normalization_acceptance,
)


class InvestigatorAcceptanceTests(unittest.TestCase):
    def test_rejects_when_delta_below_threshold(self):
        state = {
            "datasets": ["output/normalization/attempt_1/a.csv", "output/normalization/attempt_1/b.csv"],
            "original_datasets": ["output/schema-matching/a.csv", "output/schema-matching/b.csv"],
            "evaluation_attempts": 1,
        }
        gate = build_normalization_gate_request(state=state, baseline_accuracy=0.60, min_delta=0.01)
        pending = create_pending_normalization_acceptance(
            gate_request=gate,
            normalized_datasets=["output/normalization/attempt_2/a.csv", "output/normalization/attempt_2/b.csv"],
            normalization_attempt=2,
            evaluation_attempt=1,
        )
        verdict = evaluate_pending_normalization_acceptance(
            pending,
            current_accuracy=0.603,
            current_evaluation_attempt=2,
        )
        self.assertEqual(verdict["status"], "rejected")
        self.assertTrue(verdict["decision_ready"])
        self.assertGreater(len(verdict["rollback_datasets"]), 0)

    def test_accepts_when_delta_meets_threshold(self):
        state = {"datasets": ["a.csv"], "original_datasets": ["orig_a.csv"], "evaluation_attempts": 4}
        gate = build_normalization_gate_request(state=state, baseline_accuracy=0.70, min_delta=0.005)
        pending = create_pending_normalization_acceptance(
            gate_request=gate,
            normalized_datasets=["norm_a.csv"],
            normalization_attempt=3,
            evaluation_attempt=4,
        )
        verdict = evaluate_pending_normalization_acceptance(
            pending,
            current_accuracy=0.706,
            current_evaluation_attempt=5,
            current_total_evaluations=200,
            current_total_correct=141,
        )
        self.assertEqual(verdict["status"], "accepted")
        self.assertTrue(verdict["decision_ready"])
        self.assertEqual(verdict["rollback_datasets"], [])

    def test_confidence_adjusted_delta_can_reject(self):
        state = {
            "datasets": ["a.csv"],
            "original_datasets": ["orig_a.csv"],
            "evaluation_attempts": 1,
            "evaluation_metrics": {"total_evaluations": 20, "total_correct": 10},
        }
        gate = build_normalization_gate_request(state=state, baseline_accuracy=0.50, min_delta=0.01)
        pending = create_pending_normalization_acceptance(
            gate_request=gate,
            normalized_datasets=["norm_a.csv"],
            normalization_attempt=1,
            evaluation_attempt=1,
        )
        verdict = evaluate_pending_normalization_acceptance(
            pending,
            current_accuracy=0.55,
            current_evaluation_attempt=2,
            current_total_evaluations=20,
            current_total_correct=11,
        )
        self.assertEqual(verdict["status"], "rejected")
        self.assertTrue(verdict["confidence_supported"])

    def test_stays_pending_before_new_eval_attempt(self):
        state = {"datasets": ["a.csv"], "evaluation_attempts": 2}
        gate = build_normalization_gate_request(state=state, baseline_accuracy=0.50, min_delta=0.01)
        pending = create_pending_normalization_acceptance(
            gate_request=gate,
            normalized_datasets=["norm_a.csv"],
            normalization_attempt=1,
            evaluation_attempt=2,
        )
        verdict = evaluate_pending_normalization_acceptance(
            pending,
            current_accuracy=0.53,
            current_evaluation_attempt=2,
        )
        self.assertEqual(verdict["status"], "pending")
        self.assertFalse(verdict["decision_ready"])

    def test_rejects_when_heldout_proxy_drops_too_much(self):
        state = {
            "datasets": ["a.csv"],
            "evaluation_attempts": 1,
            "normalization_directives": {"list_columns": ["tracks_track_name"]},
            "evaluation_metrics": {
                "overall_accuracy": 0.70,
                "tracks_track_name_accuracy": 0.40,
                "tracks_track_name_count": 20,
                "genre_accuracy": 0.92,
                "genre_count": 20,
                "label_accuracy": 0.90,
                "label_count": 20,
            },
        }
        gate = build_normalization_gate_request(state=state, baseline_accuracy=0.70, min_delta=0.002)
        pending = create_pending_normalization_acceptance(
            gate_request=gate,
            normalized_datasets=["norm_a.csv"],
            normalization_attempt=1,
            evaluation_attempt=1,
        )
        verdict = evaluate_pending_normalization_acceptance(
            pending,
            current_accuracy=0.705,
            current_evaluation_attempt=2,
            current_metrics={
                "overall_accuracy": 0.705,
                "tracks_track_name_accuracy": 0.50,
                "tracks_track_name_count": 20,
                "genre_accuracy": 0.82,
                "genre_count": 20,
                "label_accuracy": 0.80,
                "label_count": 20,
            },
            current_total_evaluations=100,
            current_total_correct=71,
        )
        self.assertEqual(verdict["status"], "rejected")
        self.assertFalse(verdict["heldout_proxy_ok"])

    def test_accepts_when_focus_improves_and_heldout_stable(self):
        state = {
            "datasets": ["a.csv"],
            "evaluation_attempts": 1,
            "normalization_directives": {"list_columns": ["tracks_track_name"]},
            "evaluation_metrics": {
                "overall_accuracy": 0.70,
                "tracks_track_name_accuracy": 0.40,
                "tracks_track_name_count": 20,
                "genre_accuracy": 0.92,
                "genre_count": 20,
                "label_accuracy": 0.90,
                "label_count": 20,
            },
        }
        gate = build_normalization_gate_request(state=state, baseline_accuracy=0.70, min_delta=0.002)
        pending = create_pending_normalization_acceptance(
            gate_request=gate,
            normalized_datasets=["norm_a.csv"],
            normalization_attempt=1,
            evaluation_attempt=1,
        )
        verdict = evaluate_pending_normalization_acceptance(
            pending,
            current_accuracy=0.712,
            current_evaluation_attempt=2,
            current_metrics={
                "overall_accuracy": 0.712,
                "tracks_track_name_accuracy": 0.52,
                "tracks_track_name_count": 20,
                "genre_accuracy": 0.915,
                "genre_count": 20,
                "label_accuracy": 0.902,
                "label_count": 20,
            },
            current_total_evaluations=100,
            current_total_correct=72,
        )
        self.assertEqual(verdict["status"], "accepted")
        self.assertTrue(verdict["key_attrs_ok"])
        self.assertTrue(verdict["heldout_proxy_ok"])


if __name__ == "__main__":
    unittest.main()
