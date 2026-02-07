#!/usr/bin/env python3
"""Tests for the A/B test manager."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, AsyncMock, patch

from core.models import Signal
from core.types import SignalType
from strategy.live_strategy import LiveStrategy
from orchestration.ab_test_manager import ABTestManager
from orchestration.promotion_criteria import PromotionCriteria


class MockABStrategy(LiveStrategy):
    """Mock strategy for A/B test manager tests."""
    def __init__(self, name="mock_shadow"):
        super().__init__({})
        self._name = name

    def on_tick(self, market_state):
        return Signal(
            signal_type=SignalType.HOLD,
            price=market_state.mid_price,
            timestamp=datetime.now(timezone.utc),
        )

    def get_metadata(self):
        return {"name": self._name, "version": "1.0"}

    def get_state(self):
        return {}

    def set_state(self, state):
        pass


class TestABTestManager(unittest.TestCase):

    def _make_config(self):
        config = MagicMock()
        config.get.side_effect = lambda key, default=None: {
            "strategy_runner.symbol": "BTC/USDC",
            "strategy_runner.tick_interval_s": 0,
            "strategy_runner.funding_lookback_hours": 72,
            "strategy_runner.candle_lookback": 100,
            "strategy_runner.candle_timeframe": "1h",
        }.get(key, default)
        return config

    def _make_redis(self):
        redis = MagicMock()
        redis.get_price.return_value = {
            "last": 50000.0, "bid": 49999.0, "ask": 50001.0,
        }
        redis.get_funding.return_value = {
            "rate": -0.0005, "premium": 0.001,
            "mark_price": 50000.0,
        }
        redis.get_position.return_value = None
        redis.get_account_state.return_value = {
            "total_equity": 10000.0, "cash": 5000.0,
            "position_value": 5000.0, "unrealized_pnl": 0.0,
            "drawdown_pct": 0.0,
        }
        redis.get_orderbook.return_value = {
            "bids": [[49999, 1]], "asks": [[50001, 1]],
            "mid_price": 50000.0, "spread": 2.0,
        }
        redis.get_market_regime.return_value = None
        return redis

    def _make_timescale(self):
        ts = MagicMock()
        ts.query_funding_rates.return_value = []
        ts.query_candles.return_value = []
        ts.query_fills_by_strategy.return_value = []
        ts.query_equity_snapshots.return_value = []
        return ts

    def _make_manager(self, criteria_config=None):
        strategy_mgr = MagicMock()
        strategy_mgr.swap_strategy = AsyncMock()
        criteria = PromotionCriteria(criteria_config or {})
        return ABTestManager(
            strategy_manager=strategy_mgr,
            promotion_criteria=criteria,
            timescale=self._make_timescale(),
            redis=self._make_redis(),
            config=self._make_config(),
        )

    def _run_async(self, coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(
                    *pending, return_exceptions=True))
            loop.close()

    def test_start_test_returns_id(self):
        """start_test should return a valid UUID string."""
        mgr = self._make_manager()
        shadow = MockABStrategy("shadow_v2")

        test_id = self._run_async(
            mgr.start_test("live_strat", shadow, "shadow_v2"))

        self.assertIsInstance(test_id, str)
        self.assertEqual(len(test_id), 36)  # UUID length

    def test_start_test_creates_shadow(self):
        """start_test should create a shadow runner and track the test."""
        mgr = self._make_manager()
        shadow = MockABStrategy("shadow_v2")

        test_id = self._run_async(
            mgr.start_test("live_strat", shadow, "shadow_v2"))

        self.assertIn(test_id, mgr._tests)
        self.assertEqual(mgr._tests[test_id]["status"], "active")
        self.assertEqual(mgr._tests[test_id]["live_name"], "live_strat")
        self.assertEqual(mgr._tests[test_id]["shadow_name"], "shadow_v2")

    @patch("orchestration.ab_test_manager.ABTestManager._get_live_metrics")
    def test_check_test_returns_comparison(self, mock_live_metrics):
        """check_test should return shadow and live metrics comparison."""
        mock_live_metrics.return_value = {
            "trade_count": 10, "total_pnl": 100.0,
            "win_rate": 50.0, "sharpe_ratio": 0.5, "max_drawdown": 2.0,
        }

        mgr = self._make_manager()
        shadow = MockABStrategy("shadow_v2")

        async def _test():
            test_id = await mgr.start_test("live_strat", shadow, "shadow_v2")
            return await mgr.check_test(test_id)

        result = self._run_async(_test())

        self.assertEqual(result["status"], "active")
        self.assertIn("shadow_metrics", result)
        self.assertIn("live_metrics", result)
        self.assertIn("recommendation", result)
        self.assertIn("meets_criteria", result)

    @patch("orchestration.ab_test_manager.ABTestManager._get_live_metrics")
    def test_check_test_not_ready_before_duration(self, mock_live_metrics):
        """Test should recommend 'continue' before min duration."""
        mock_live_metrics.return_value = {
            "trade_count": 10, "total_pnl": 100.0,
            "win_rate": 50.0, "sharpe_ratio": 0.5, "max_drawdown": 2.0,
        }

        mgr = self._make_manager()
        shadow = MockABStrategy("shadow_v2")

        async def _test():
            test_id = await mgr.start_test("live_strat", shadow, "shadow_v2")
            return await mgr.check_test(test_id)

        result = self._run_async(_test())

        self.assertEqual(result["recommendation"], "continue")
        self.assertFalse(result["meets_criteria"])

    def test_promote_calls_swap(self):
        """promote should call strategy_manager.swap_strategy."""
        mgr = self._make_manager()
        shadow = MockABStrategy("shadow_v2")

        async def _test():
            test_id = await mgr.start_test("live_strat", shadow, "shadow_v2")
            await mgr.promote(test_id)
            return test_id

        test_id = self._run_async(_test())

        mgr.strategy_manager.swap_strategy.assert_awaited_once_with(
            "live_strat", shadow, "shadow_v2")
        self.assertEqual(mgr._tests[test_id]["status"], "promoted")

    def test_reject_stops_shadow(self):
        """reject should stop the shadow runner and mark as rejected."""
        mgr = self._make_manager()
        shadow = MockABStrategy("shadow_v2")

        async def _test():
            test_id = await mgr.start_test("live_strat", shadow, "shadow_v2")
            await mgr.reject(test_id)
            return test_id

        test_id = self._run_async(_test())

        self.assertEqual(mgr._tests[test_id]["status"], "rejected")

    def test_get_active_tests(self):
        """get_active_tests should return only active tests."""
        mgr = self._make_manager()

        async def _test():
            test_id1 = await mgr.start_test(
                "live_a", MockABStrategy("shadow_a"), "shadow_a")
            test_id2 = await mgr.start_test(
                "live_b", MockABStrategy("shadow_b"), "shadow_b")

            active = mgr.get_active_tests()
            self.assertEqual(len(active), 2)

            await mgr.reject(test_id1)

            active = mgr.get_active_tests()
            self.assertEqual(len(active), 1)
            self.assertEqual(active[0]["shadow_name"], "shadow_b")

        self._run_async(_test())

    def test_multiple_concurrent_tests(self):
        """Should support multiple concurrent A/B tests."""
        mgr = self._make_manager()

        async def _test():
            ids = []
            for i in range(3):
                tid = await mgr.start_test(
                    f"live_{i}",
                    MockABStrategy(f"shadow_{i}"),
                    f"shadow_{i}")
                ids.append(tid)
            return ids

        ids = self._run_async(_test())

        self.assertEqual(len(ids), 3)
        self.assertEqual(len(mgr.get_active_tests()), 3)
        self.assertEqual(len(set(ids)), 3)

    @patch("orchestration.ab_test_manager.ABTestManager._get_live_metrics")
    def test_check_all_tests(self, mock_live_metrics):
        """check_all_tests should return results for all active tests."""
        mock_live_metrics.return_value = {
            "trade_count": 10, "total_pnl": 100.0,
            "win_rate": 50.0, "sharpe_ratio": 0.5, "max_drawdown": 2.0,
        }

        mgr = self._make_manager()

        async def _test():
            await mgr.start_test(
                "live_a", MockABStrategy("shadow_a"), "shadow_a")
            await mgr.start_test(
                "live_b", MockABStrategy("shadow_b"), "shadow_b")
            return await mgr.check_all_tests()

        results = self._run_async(_test())

        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIn("test_id", r)
            self.assertEqual(r["status"], "active")

    def test_promote_marks_concluded(self):
        """After promotion, test should be marked with concluded_at."""
        mgr = self._make_manager()
        shadow = MockABStrategy("shadow_v2")

        async def _test():
            test_id = await mgr.start_test("live_strat", shadow, "shadow_v2")
            await mgr.promote(test_id)
            return test_id

        test_id = self._run_async(_test())

        self.assertEqual(mgr._tests[test_id]["status"], "promoted")
        self.assertIn("concluded_at", mgr._tests[test_id])

    def test_reject_marks_rejected(self):
        """After rejection, test should have rejected status and concluded_at."""
        mgr = self._make_manager()
        shadow = MockABStrategy("shadow_v2")

        async def _test():
            test_id = await mgr.start_test("live_strat", shadow, "shadow_v2")
            await mgr.reject(test_id)
            return test_id

        test_id = self._run_async(_test())

        self.assertEqual(mgr._tests[test_id]["status"], "rejected")
        self.assertIn("concluded_at", mgr._tests[test_id])


if __name__ == "__main__":
    unittest.main()
