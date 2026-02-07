"""Tests for Phase 4 OODA integration: AI decision engine, allocation, A/B tests."""

import sys
import os
import asyncio
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orchestration.ooda import OODAOrchestrator


def run_async(coro):
    """Helper to run async tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestOODASafetyRules(unittest.TestCase):
    """Test that safety rules always take priority."""

    def _make_orchestrator(self, **kwargs):
        strategy_manager = MagicMock()
        strategy_manager.get_active_strategies.return_value = ["strat_a"]
        strategy_manager._strategies = {
            "strat_a": {"strategy": MagicMock(), "status": "active"},
        }

        health_scorer = MagicMock()
        regime_classifier = MagicMock()
        timescale = MagicMock()
        timescale.query_candles.return_value = []
        timescale.insert_decision.return_value = None
        timescale.insert_system_event.return_value = None

        redis_store = MagicMock()
        redis_store.set_market_regime.return_value = None

        return OODAOrchestrator(
            strategy_manager=strategy_manager,
            health_scorer=health_scorer,
            regime_classifier=regime_classifier,
            timescale=timescale,
            redis_store=redis_store,
            config={"strategy_name": "strat_a"},
            **kwargs,
        )

    def test_safety_overrides_ai(self):
        """Safety rules should fire even when AI says no_action."""
        ai_engine = MagicMock()
        ai_engine.decide.return_value = None  # AI says no action

        ooda = self._make_orchestrator(ai_decision_engine=ai_engine)

        metrics = {"health": {"health_score": 10, "raw_metrics": {"max_drawdown": 1}}}
        regime = {"regime": "trending_up"}
        assessment = {"suitable": True}

        decision = ooda._decide(metrics, regime, assessment, "hourly")
        self.assertIsNotNone(decision)
        self.assertEqual(decision["action"], "pause_strategy")
        self.assertEqual(decision["source"], "safety_rules")

    def test_safety_health_low(self):
        ooda = self._make_orchestrator()
        metrics = {"health": {"health_score": 15, "raw_metrics": {"max_drawdown": 2}}}
        decision = ooda._safety_rules(metrics, {"suitable": True}, {})
        self.assertEqual(decision["action"], "pause_strategy")

    def test_safety_backtest_decay(self):
        ooda = self._make_orchestrator()
        metrics = {
            "health": {"health_score": 80, "raw_metrics": {"max_drawdown": 2}},
            "backtest_comparison": {"decay_pct": 60},
        }
        decision = ooda._safety_rules(metrics, {"suitable": True}, {})
        self.assertEqual(decision["action"], "pause_strategy")

    def test_safety_max_drawdown(self):
        ooda = self._make_orchestrator()
        metrics = {
            "health": {"health_score": 80, "raw_metrics": {"max_drawdown": 7.0}},
        }
        decision = ooda._safety_rules(metrics, {"suitable": True}, {})
        self.assertEqual(decision["action"], "adjust_risk")

    def test_safety_portfolio_dd(self):
        sm = MagicMock()
        sm.get_active_strategies.return_value = ["a", "b"]
        ooda = self._make_orchestrator()
        ooda.strategy_manager = sm

        metrics = {
            "health": {"health_score": 80, "raw_metrics": {"max_drawdown": 2}},
            "portfolio": {"portfolio_max_dd": 20},
            "per_strategy_health": {
                "a": {"health_score": 30},
                "b": {"health_score": 70},
            },
        }
        decision = ooda._safety_rules(metrics, {"suitable": True}, {})
        self.assertEqual(decision["action"], "pause_strategy")
        self.assertEqual(decision["strategy_name"], "a")

    def test_no_safety_issue(self):
        ooda = self._make_orchestrator()
        metrics = {"health": {"health_score": 80, "raw_metrics": {"max_drawdown": 2}}}
        decision = ooda._safety_rules(metrics, {"suitable": True}, {})
        self.assertIsNone(decision)


class TestOODAAIDecision(unittest.TestCase):
    """Test AI decision engine integration in _decide()."""

    def _make_orchestrator(self, ai_result=None):
        sm = MagicMock()
        sm.get_active_strategies.return_value = ["strat_a"]
        sm._strategies = {"strat_a": {"strategy": MagicMock(), "status": "active"}}

        ai_engine = MagicMock()
        ai_engine.decide.return_value = ai_result

        return OODAOrchestrator(
            strategy_manager=sm,
            health_scorer=MagicMock(),
            regime_classifier=MagicMock(),
            timescale=MagicMock(),
            redis_store=MagicMock(),
            config={"strategy_name": "strat_a"},
            ai_decision_engine=ai_engine,
        )

    def test_ai_decision_used_when_available(self):
        ai_decision = {
            "action": "adjust_allocation",
            "strategy_name": "strat_a",
            "reason": "AI thinks so",
            "hypothesis": "Better performance",
            "confidence": 0.8,
            "source": "ai_engine",
        }
        ooda = self._make_orchestrator(ai_result=ai_decision)

        metrics = {"health": {"health_score": 80, "raw_metrics": {"max_drawdown": 2}}}
        decision = ooda._decide(metrics, {"regime": "trending_up"}, {"suitable": True}, "hourly")
        self.assertEqual(decision["source"], "ai_engine")
        self.assertEqual(decision["action"], "adjust_allocation")

    def test_falls_back_to_rules_when_ai_returns_none(self):
        ooda = self._make_orchestrator(ai_result=None)

        metrics = {"health": {"health_score": 80, "raw_metrics": {"max_drawdown": 2}}}
        decision = ooda._decide(metrics, {"regime": "volatile"}, {"suitable": False, "reason": "bad regime"}, "hourly")
        self.assertEqual(decision["action"], "adjust_risk")
        self.assertEqual(decision["source"], "rules")

    def test_ai_exception_falls_back(self):
        sm = MagicMock()
        sm.get_active_strategies.return_value = ["strat_a"]
        sm._strategies = {"strat_a": {"strategy": MagicMock(), "status": "active"}}

        ai_engine = MagicMock()
        ai_engine.decide.side_effect = Exception("API error")

        ooda = OODAOrchestrator(
            strategy_manager=sm,
            health_scorer=MagicMock(),
            regime_classifier=MagicMock(),
            timescale=MagicMock(),
            redis_store=MagicMock(),
            config={"strategy_name": "strat_a"},
            ai_decision_engine=ai_engine,
        )

        metrics = {"health": {"health_score": 80, "raw_metrics": {"max_drawdown": 2}}}
        # Should not raise, just fall back
        decision = ooda._decide(metrics, {"regime": "ranging"}, {"suitable": True}, "hourly")
        # No safety or AI trigger, rules return None
        self.assertIsNone(decision)


class TestOODAActAllocations(unittest.TestCase):
    """Test _act() with allocation and promotion actions."""

    def _make_orchestrator(self):
        sm = MagicMock()
        sm.pause_strategy = AsyncMock()
        sm.resume_strategy = AsyncMock()
        sm.update_allocation = MagicMock()

        return OODAOrchestrator(
            strategy_manager=sm,
            health_scorer=MagicMock(),
            regime_classifier=MagicMock(),
            timescale=MagicMock(),
            redis_store=MagicMock(),
        )

    def test_act_adjust_allocation_executes(self):
        ooda = self._make_orchestrator()
        decision = {
            "action": "adjust_allocation",
            "strategy_name": "strat_a",
            "recommended_allocation": 0.3,
        }
        result = run_async(ooda._act(decision))
        self.assertTrue(result["executed"])
        self.assertEqual(result["new_allocation"], 0.3)
        ooda.strategy_manager.update_allocation.assert_called_once_with("strat_a", 0.3)

    def test_act_rebalance(self):
        ooda = self._make_orchestrator()
        decision = {
            "action": "rebalance",
            "rebalance_actions": [
                {"strategy_name": "a", "target_weight": 0.4},
                {"strategy_name": "b", "target_weight": 0.3},
            ],
            "target_allocations": {"a": 0.4, "b": 0.3},
        }
        result = run_async(ooda._act(decision))
        self.assertTrue(result["executed"])
        self.assertEqual(result["strategies_rebalanced"], 2)
        self.assertEqual(ooda.strategy_manager.update_allocation.call_count, 2)

    def test_act_promote_strategy(self):
        ab_manager = MagicMock()
        ab_manager.check_test = AsyncMock(return_value={
            "recommendation": "promote",
            "meets_criteria": True,
        })
        ab_manager.promote = AsyncMock()

        ooda = self._make_orchestrator()
        ooda.ab_test_manager = ab_manager

        decision = {
            "action": "promote_strategy",
            "test_id": "abc-123",
            "shadow_name": "new_strat",
        }
        result = run_async(ooda._act(decision))
        self.assertTrue(result["executed"])
        ab_manager.promote.assert_called_once_with("abc-123")

    def test_act_reject_strategy(self):
        ab_manager = MagicMock()
        ab_manager.check_test = AsyncMock(return_value={
            "recommendation": "reject",
            "meets_criteria": False,
        })
        ab_manager.reject = AsyncMock()

        ooda = self._make_orchestrator()
        ooda.ab_test_manager = ab_manager

        decision = {
            "action": "promote_strategy",
            "test_id": "abc-123",
        }
        result = run_async(ooda._act(decision))
        self.assertTrue(result["executed"])
        self.assertEqual(result["action"], "reject_strategy")
        ab_manager.reject.assert_called_once()

    def test_act_promote_continue(self):
        ab_manager = MagicMock()
        ab_manager.check_test = AsyncMock(return_value={
            "recommendation": "continue",
        })

        ooda = self._make_orchestrator()
        ooda.ab_test_manager = ab_manager

        decision = {
            "action": "promote_strategy",
            "test_id": "abc-123",
        }
        result = run_async(ooda._act(decision))
        self.assertTrue(result["executed"])
        self.assertIn("not yet ready", result["note"])


class TestOODAABTestCheck(unittest.TestCase):
    """Test A/B test check rule."""

    def test_ab_test_triggers_promotion_check(self):
        ab_manager = MagicMock()
        ab_manager.get_active_tests.return_value = [
            {"test_id": "t1", "shadow_name": "shadow_a", "live_name": "live_a"},
        ]

        sm = MagicMock()
        sm.get_active_strategies.return_value = ["live_a"]
        sm._strategies = {"live_a": {"strategy": MagicMock(), "status": "active"}}

        ooda = OODAOrchestrator(
            strategy_manager=sm,
            health_scorer=MagicMock(),
            regime_classifier=MagicMock(),
            timescale=MagicMock(),
            redis_store=MagicMock(),
            config={"strategy_name": "live_a"},
            ab_test_manager=ab_manager,
        )

        metrics = {"health": {"health_score": 80, "raw_metrics": {"max_drawdown": 2}}}
        decision = ooda._rule_based_decide(
            metrics, {"regime": "ranging"}, {"suitable": True}, "daily")
        self.assertEqual(decision["action"], "promote_strategy")

    def test_no_ab_tests_no_action(self):
        ab_manager = MagicMock()
        ab_manager.get_active_tests.return_value = []

        sm = MagicMock()
        sm.get_active_strategies.return_value = ["live_a"]
        sm._strategies = {"live_a": {"strategy": MagicMock(), "status": "active"}}
        sm.get_allocations.return_value = {}

        ooda = OODAOrchestrator(
            strategy_manager=sm,
            health_scorer=MagicMock(),
            regime_classifier=MagicMock(),
            timescale=MagicMock(),
            redis_store=MagicMock(),
            config={"strategy_name": "live_a"},
            ab_test_manager=ab_manager,
        )

        metrics = {"health": {"health_score": 80, "raw_metrics": {"max_drawdown": 2}}}
        decision = ooda._rule_based_decide(
            metrics, {"regime": "ranging"}, {"suitable": True}, "daily")
        self.assertIsNone(decision)


class TestOODARebalanceCheck(unittest.TestCase):
    """Test allocation rebalance rule."""

    def test_rebalance_triggers_action(self):
        optimizer = MagicMock()
        optimizer.optimize.return_value = {"a": 0.35, "b": 0.35}
        optimizer.get_rebalance_actions.return_value = [
            {"strategy_name": "a", "current_weight": 0.50, "target_weight": 0.35, "change": -0.15},
        ]

        sm = MagicMock()
        sm.get_active_strategies.return_value = ["a", "b"]
        sm._strategies = {
            "a": {"strategy": MagicMock(), "status": "active"},
            "b": {"strategy": MagicMock(), "status": "active"},
        }
        sm.get_allocations.return_value = {"a": 0.50, "b": 0.30}

        ooda = OODAOrchestrator(
            strategy_manager=sm,
            health_scorer=MagicMock(),
            regime_classifier=MagicMock(),
            timescale=MagicMock(),
            redis_store=MagicMock(),
            config={"strategy_name": "a"},
            allocation_optimizer=optimizer,
        )

        metrics = {
            "health": {"health_score": 80, "raw_metrics": {"max_drawdown": 2}},
            "per_strategy_health": {
                "a": {"health_score": 70, "raw_metrics": {"max_drawdown": 1}},
                "b": {"health_score": 60, "raw_metrics": {"max_drawdown": 2}},
            },
        }
        decision = ooda._rule_based_decide(
            metrics, {"regime": "ranging"}, {"suitable": True}, "daily")
        self.assertEqual(decision["action"], "rebalance")


if __name__ == "__main__":
    unittest.main()
