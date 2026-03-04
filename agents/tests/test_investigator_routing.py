import os
import sys
import unittest

AGENTS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if AGENTS_ROOT not in sys.path:
    sys.path.insert(0, AGENTS_ROOT)

from helpers.investigator_routing import compute_normalization_routing


class InvestigatorRoutingTests(unittest.TestCase):
    def _signals(self, **overrides):
        base = {
            "norm": {"count": 2, "ema_gain": 0.01, "wins": 1, "losses": 1, "last_gain": 0.01},
            "pipe": {"count": 2, "ema_gain": 0.01, "wins": 1, "losses": 1, "last_gain": 0.01},
            "learning_bias": 0.0,
            "regret_active": False,
            "recent_stall": False,
            "normalization_stall_streak": 0,
        }
        base.update(overrides)
        return base

    def test_routes_to_normalization_on_strong_signal(self):
        result = compute_normalization_routing(
            overall_acc=0.55,
            attempts=1,
            max_attempts=3,
            llm_assessment={"needs_normalization": True, "confidence": 0.9},
            fallback_needed=True,
            diagnostics={"debug_reason_ratios": {"missing_fused_value": 0.4}},
            action_plan=[],
            learning_signals=self._signals(),
            probe_results={"normalization_pressure": 0.7},
            force_skip_normalization=False,
        )
        self.assertTrue(result["route_to_normalization"])

    def test_blocks_on_force_skip(self):
        result = compute_normalization_routing(
            overall_acc=0.40,
            attempts=1,
            max_attempts=3,
            llm_assessment={"needs_normalization": True, "confidence": 0.99},
            fallback_needed=True,
            diagnostics={"debug_reason_ratios": {"missing_fused_value": 0.9}},
            action_plan=[],
            learning_signals=self._signals(),
            probe_results={"normalization_pressure": 1.0},
            force_skip_normalization=True,
        )
        self.assertFalse(result["route_to_normalization"])
        self.assertEqual(result["components"].get("forced_skip_normalization"), 1.0)

    def test_regret_blocks_normalization_when_score_not_extreme(self):
        signals = self._signals(regret_active=True, learning_bias=-1.0, recent_stall=True, normalization_stall_streak=2)
        result = compute_normalization_routing(
            overall_acc=0.58,
            attempts=1,
            max_attempts=3,
            llm_assessment={"needs_normalization": True, "confidence": 0.6},
            fallback_needed=False,
            diagnostics={"debug_reason_ratios": {}},
            action_plan=[],
            learning_signals=signals,
            probe_results={"normalization_pressure": 0.1},
            force_skip_normalization=False,
        )
        self.assertFalse(result["route_to_normalization"])
        self.assertTrue(result["blocked_by_learning_regret"])

    def test_objective_pressure_contributes_to_score(self):
        result = compute_normalization_routing(
            overall_acc=0.65,
            attempts=1,
            max_attempts=3,
            llm_assessment={"needs_normalization": False, "confidence": 0.0},
            fallback_needed=False,
            diagnostics={"debug_reason_ratios": {}},
            action_plan=[],
            learning_signals=self._signals(),
            probe_results={"normalization_pressure": 0.0},
            route_objective={"composite_score": 0.30, "worst_attribute_pressure": 0.8},
            force_skip_normalization=False,
        )
        self.assertIn("objective_low_score", result["components"])
        self.assertIn("objective_worst_attribute_pressure", result["components"])


if __name__ == "__main__":
    unittest.main()
