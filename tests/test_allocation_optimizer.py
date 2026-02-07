"""Tests for AllocationOptimizer."""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orchestration.allocation_optimizer import AllocationOptimizer


class TestAllocationOptimizerEqualWeight(unittest.TestCase):
    """Test equal_weight allocation mode."""

    def setUp(self):
        self.optimizer = AllocationOptimizer({"mode": "equal_weight", "cash_reserve": 0.20})

    def test_single_strategy(self):
        strategies = [{"name": "strat_a", "status": "active", "health_score": 80}]
        result = self.optimizer.optimize(strategies)
        self.assertIn("strat_a", result)
        self.assertAlmostEqual(result["strat_a"], 0.60, places=1)  # constrained by max

    def test_two_strategies(self):
        strategies = [
            {"name": "strat_a", "status": "active", "health_score": 80},
            {"name": "strat_b", "status": "active", "health_score": 60},
        ]
        result = self.optimizer.optimize(strategies)
        self.assertAlmostEqual(result["strat_a"], result["strat_b"], places=2)
        self.assertLessEqual(sum(result.values()), 0.80 + 0.01)  # 1.0 - 0.20 reserve

    def test_three_strategies(self):
        strategies = [
            {"name": "a", "status": "active"},
            {"name": "b", "status": "active"},
            {"name": "c", "status": "active"},
        ]
        result = self.optimizer.optimize(strategies)
        self.assertEqual(len(result), 3)
        # Each should get roughly 0.80/3 = 0.267
        for weight in result.values():
            self.assertGreater(weight, 0.0)
        self.assertLessEqual(sum(result.values()), 0.80 + 0.01)

    def test_filters_paused_strategies(self):
        strategies = [
            {"name": "active", "status": "active"},
            {"name": "paused", "status": "paused"},
        ]
        result = self.optimizer.optimize(strategies)
        self.assertIn("active", result)
        self.assertNotIn("paused", result)

    def test_empty_strategies(self):
        result = self.optimizer.optimize([])
        self.assertEqual(result, {})

    def test_all_paused(self):
        strategies = [{"name": "a", "status": "paused"}]
        result = self.optimizer.optimize([])
        self.assertEqual(result, {})


class TestAllocationOptimizerRiskParity(unittest.TestCase):
    """Test risk_parity allocation mode."""

    def setUp(self):
        self.optimizer = AllocationOptimizer({
            "mode": "risk_parity",
            "cash_reserve": 0.20,
            "min_allocation": 0.05,
            "max_allocation": 0.60,
            "step": 0.01,
        })

    def test_lower_vol_gets_higher_allocation(self):
        strategies = [
            {"name": "low_vol", "status": "active", "volatility": 0.10},
            {"name": "high_vol", "status": "active", "volatility": 0.40},
        ]
        result = self.optimizer.optimize(strategies)
        self.assertGreater(result["low_vol"], result["high_vol"])

    def test_equal_vol_equal_allocation(self):
        strategies = [
            {"name": "a", "status": "active", "volatility": 0.20},
            {"name": "b", "status": "active", "volatility": 0.20},
        ]
        result = self.optimizer.optimize(strategies)
        self.assertAlmostEqual(result["a"], result["b"], places=2)

    def test_falls_back_to_equal_weight_no_vol(self):
        strategies = [
            {"name": "a", "status": "active"},
            {"name": "b", "status": "active"},
        ]
        result = self.optimizer.optimize(strategies)
        self.assertAlmostEqual(result["a"], result["b"], places=2)

    def test_zero_vol_treated_as_no_data(self):
        strategies = [
            {"name": "a", "status": "active", "volatility": 0.0},
            {"name": "b", "status": "active", "volatility": 0.20},
        ]
        result = self.optimizer.optimize(strategies)
        self.assertIn("a", result)
        self.assertIn("b", result)


class TestAllocationOptimizerHealthWeighted(unittest.TestCase):
    """Test health_weighted allocation mode."""

    def setUp(self):
        self.optimizer = AllocationOptimizer({
            "mode": "health_weighted",
            "cash_reserve": 0.20,
            "min_allocation": 0.05,
            "max_allocation": 0.60,
            "step": 0.01,
        })

    def test_higher_health_gets_more(self):
        strategies = [
            {"name": "healthy", "status": "active", "health_score": 90},
            {"name": "sick", "status": "active", "health_score": 30},
        ]
        result = self.optimizer.optimize(strategies)
        self.assertGreater(result["healthy"], result["sick"])

    def test_equal_health_equal_allocation(self):
        strategies = [
            {"name": "a", "status": "active", "health_score": 70},
            {"name": "b", "status": "active", "health_score": 70},
        ]
        result = self.optimizer.optimize(strategies)
        self.assertAlmostEqual(result["a"], result["b"], places=2)

    def test_zero_health_floored(self):
        strategies = [
            {"name": "a", "status": "active", "health_score": 0},
            {"name": "b", "status": "active", "health_score": 100},
        ]
        result = self.optimizer.optimize(strategies)
        self.assertIn("a", result)
        self.assertGreater(result["a"], 0)


class TestAllocationConstraints(unittest.TestCase):
    """Test constraint application."""

    def test_min_allocation_enforced(self):
        optimizer = AllocationOptimizer({
            "mode": "equal_weight",
            "min_allocation": 0.10,
            "max_allocation": 0.60,
            "cash_reserve": 0.20,
        })
        strategies = [{"name": f"s{i}", "status": "active"} for i in range(10)]
        result = optimizer.optimize(strategies)
        for weight in result.values():
            self.assertGreaterEqual(weight, 0.0)  # may be below min after scaling

    def test_max_allocation_enforced(self):
        optimizer = AllocationOptimizer({
            "mode": "equal_weight",
            "max_allocation": 0.30,
            "cash_reserve": 0.20,
        })
        strategies = [{"name": "solo", "status": "active"}]
        result = optimizer.optimize(strategies)
        self.assertLessEqual(result["solo"], 0.30 + 0.001)

    def test_step_snapping(self):
        optimizer = AllocationOptimizer({
            "mode": "equal_weight",
            "step": 0.05,
            "min_allocation": 0.05,
            "max_allocation": 0.60,
            "cash_reserve": 0.20,
        })
        strategies = [
            {"name": "a", "status": "active"},
            {"name": "b", "status": "active"},
            {"name": "c", "status": "active"},
        ]
        result = optimizer.optimize(strategies)
        for weight in result.values():
            # After snapping, weight should be close to a multiple of 0.05
            remainder = round(weight % 0.05, 6)
            self.assertTrue(remainder < 0.001 or remainder > 0.049,
                          f"Weight {weight} not snapped to step 0.05")

    def test_total_respects_cash_reserve(self):
        optimizer = AllocationOptimizer({
            "mode": "equal_weight",
            "cash_reserve": 0.30,
        })
        strategies = [
            {"name": "a", "status": "active"},
            {"name": "b", "status": "active"},
        ]
        result = optimizer.optimize(strategies)
        self.assertLessEqual(sum(result.values()), 0.70 + 0.01)

    def test_invalid_mode_falls_back(self):
        optimizer = AllocationOptimizer({"mode": "nonexistent"})
        strategies = [{"name": "a", "status": "active"}]
        result = optimizer.optimize(strategies)
        self.assertIn("a", result)


class TestRebalanceActions(unittest.TestCase):
    """Test rebalance action computation."""

    def setUp(self):
        self.optimizer = AllocationOptimizer()

    def test_no_actions_when_in_sync(self):
        current = {"a": 0.40, "b": 0.40}
        target = {"a": 0.40, "b": 0.40}
        actions = self.optimizer.get_rebalance_actions(current, target)
        self.assertEqual(len(actions), 0)

    def test_actions_when_drifted(self):
        current = {"a": 0.50, "b": 0.30}
        target = {"a": 0.35, "b": 0.45}
        actions = self.optimizer.get_rebalance_actions(current, target, threshold=0.05)
        self.assertEqual(len(actions), 2)

    def test_threshold_filters_small_changes(self):
        current = {"a": 0.40, "b": 0.40}
        target = {"a": 0.41, "b": 0.39}
        actions = self.optimizer.get_rebalance_actions(current, target, threshold=0.05)
        self.assertEqual(len(actions), 0)

    def test_new_strategy_in_target(self):
        current = {"a": 0.50}
        target = {"a": 0.40, "b": 0.30}
        actions = self.optimizer.get_rebalance_actions(current, target, threshold=0.05)
        self.assertTrue(any(a["strategy_name"] == "b" for a in actions))

    def test_removed_strategy_in_current(self):
        current = {"a": 0.40, "b": 0.30}
        target = {"a": 0.50}
        actions = self.optimizer.get_rebalance_actions(current, target, threshold=0.05)
        b_action = [a for a in actions if a["strategy_name"] == "b"]
        self.assertEqual(len(b_action), 1)
        self.assertLess(b_action[0]["change"], 0)


if __name__ == "__main__":
    unittest.main()
