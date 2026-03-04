import os
import sys
import unittest

AGENTS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if AGENTS_ROOT not in sys.path:
    sys.path.insert(0, AGENTS_ROOT)

from helpers.investigator_learning import (
    apply_learning_decay_for_drift,
    default_learning_state,
    learning_routing_signals,
    record_learning_checkpoint,
    update_learning_from_observation,
)


class InvestigatorLearningTests(unittest.TestCase):
    def test_updates_only_after_eval_attempt_advances(self):
        learning = default_learning_state()
        learning = record_learning_checkpoint(
            learning,
            decision="normalization_node",
            accuracy=0.50,
            evaluation_attempt=1,
            dataset_signature="a|b",
        )
        learning, obs = update_learning_from_observation(
            learning,
            current_accuracy=0.55,
            current_eval_attempt=1,
            dataset_signature="a|b",
        )
        self.assertIsNone(obs)
        self.assertEqual(learning["routing"]["normalization_node"]["count"], 0)

        learning, obs = update_learning_from_observation(
            learning,
            current_accuracy=0.55,
            current_eval_attempt=2,
            dataset_signature="a|b",
        )
        self.assertIsNotNone(obs)
        self.assertAlmostEqual(obs["gain"], 0.05, places=6)
        self.assertEqual(learning["routing"]["normalization_node"]["count"], 1)
        self.assertEqual(learning["routing"]["normalization_node"]["wins"], 1)

    def test_clears_pending_on_dataset_switch(self):
        learning = default_learning_state()
        learning = record_learning_checkpoint(
            learning,
            decision="pipeline_adaption",
            accuracy=0.61,
            evaluation_attempt=3,
            dataset_signature="music",
        )
        learning, obs = update_learning_from_observation(
            learning,
            current_accuracy=0.64,
            current_eval_attempt=4,
            dataset_signature="games",
        )
        self.assertIsNone(obs)
        self.assertEqual(learning.get("pending_observation"), {})

    def test_learning_signals_activate_regret(self):
        learning = default_learning_state()
        learning["routing"]["normalization_node"] = {"count": 3, "ema_gain": -0.02, "wins": 0, "losses": 3, "last_gain": -0.01}
        learning["routing"]["pipeline_adaption"] = {"count": 3, "ema_gain": 0.03, "wins": 3, "losses": 0, "last_gain": 0.02}
        signals = learning_routing_signals(learning)
        self.assertTrue(signals["regret_active"])
        self.assertLess(signals["learning_bias"], 0.0)

    def test_applies_decay_on_context_change(self):
        learning = default_learning_state()
        learning["routing"]["normalization_node"] = {"count": 10, "ema_gain": 0.05, "wins": 7, "losses": 3, "last_gain": 0.01}
        learning["last_context_key"] = "old"
        out = apply_learning_decay_for_drift(learning, "new")
        self.assertLess(out["routing"]["normalization_node"]["ema_gain"], 0.05)
        self.assertLess(out["routing"]["normalization_node"]["count"], 10)
        self.assertEqual(out["last_context_key"], "new")
        self.assertGreaterEqual(out.get("drift_events", 0), 1)


if __name__ == "__main__":
    unittest.main()
