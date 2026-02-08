"""Tests for the Dashboard API."""

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.server import DashboardAPI


class TestDashboardAPI(unittest.TestCase):
    """Tests for the DashboardAPI data fetchers."""

    def _make_api(self, redis_data=None, strategy_manager=None):
        redis = MagicMock()
        redis.get_account_state.return_value = redis_data or {
            "total_equity": 10000,
            "cash": 5000,
            "drawdown_pct": 2.5,
            "unrealized_pnl": 150,
        }
        redis.get_market_regime.return_value = {
            "regime": "trending_up", "confidence": 0.85,
        }
        redis.get_all_positions.return_value = [
            {"symbol": "BTC/USDC", "side": "buy", "quantity": 0.1},
        ]

        timescale = MagicMock()
        timescale.query_decisions_by_type.return_value = [
            {"decision_type": "ooda_hourly", "action": "none"},
        ]
        timescale.query_wallet_scores.return_value = [
            {"address": "0xabc", "score": 85},
        ]

        return DashboardAPI(
            redis_store=redis,
            timescale_store=timescale,
            strategy_manager=strategy_manager,
            config={"symbol": "BTC/USDC"},
        )

    def _make_strategy_manager(self, strategies=None):
        sm = MagicMock()
        strategies = strategies or ["funding_rate_arb", "momentum"]
        sm.get_active_strategies.return_value = strategies
        sm._strategies = {
            name: {"status": "active", "started_at": "2025-01-01T00:00:00Z"}
            for name in strategies
        }
        return sm

    # ------------------------------------------------------------------
    # handle_status
    # ------------------------------------------------------------------

    def test_handle_status_returns_dict(self):
        api = self._make_api()
        result = api.handle_status()
        self.assertIsInstance(result, dict)
        self.assertIn("status", result)
        self.assertEqual(result["status"], "running")

    def test_handle_status_includes_equity(self):
        api = self._make_api()
        result = api.handle_status()
        self.assertEqual(result["equity"], 10000)

    def test_handle_status_with_strategy_manager(self):
        sm = self._make_strategy_manager()
        api = self._make_api(strategy_manager=sm)
        result = api.handle_status()
        self.assertEqual(len(result["active_strategies"]), 2)

    def test_handle_status_no_strategy_manager(self):
        api = self._make_api()
        result = api.handle_status()
        self.assertEqual(result["active_strategies"], [])

    # ------------------------------------------------------------------
    # handle_strategies
    # ------------------------------------------------------------------

    def test_handle_strategies_returns_list(self):
        sm = self._make_strategy_manager()
        api = self._make_api(strategy_manager=sm)
        result = api.handle_strategies()
        self.assertIn("strategies", result)
        self.assertEqual(len(result["strategies"]), 2)

    def test_handle_strategies_no_manager(self):
        api = self._make_api()
        result = api.handle_strategies()
        self.assertEqual(result["strategies"], [])

    def test_handle_strategies_includes_status(self):
        sm = self._make_strategy_manager()
        api = self._make_api(strategy_manager=sm)
        result = api.handle_strategies()
        for s in result["strategies"]:
            self.assertIn("status", s)
            self.assertEqual(s["status"], "active")

    # ------------------------------------------------------------------
    # handle_portfolio
    # ------------------------------------------------------------------

    def test_handle_portfolio_returns_metrics(self):
        api = self._make_api()
        result = api.handle_portfolio()
        self.assertEqual(result["equity"], 10000)
        self.assertEqual(result["cash"], 5000)
        self.assertAlmostEqual(result["drawdown_pct"], 2.5)

    def test_handle_portfolio_empty_redis(self):
        api = self._make_api()
        api.redis.get_account_state.return_value = {}
        result = api.handle_portfolio()
        self.assertEqual(result["equity"], 0)

    # ------------------------------------------------------------------
    # handle_positions
    # ------------------------------------------------------------------

    def test_handle_positions_returns_list(self):
        api = self._make_api()
        result = api.handle_positions()
        self.assertIn("positions", result)
        self.assertEqual(len(result["positions"]), 1)

    def test_handle_positions_includes_data(self):
        api = self._make_api()
        result = api.handle_positions()
        self.assertEqual(result["positions"][0]["symbol"], "BTC/USDC")

    # ------------------------------------------------------------------
    # handle_decisions
    # ------------------------------------------------------------------

    def test_handle_decisions_returns_list(self):
        api = self._make_api()
        result = api.handle_decisions()
        self.assertIn("decisions", result)
        self.assertEqual(len(result["decisions"]), 1)

    def test_handle_decisions_includes_data(self):
        api = self._make_api()
        result = api.handle_decisions()
        self.assertEqual(result["decisions"][0]["decision_type"], "ooda_hourly")

    # ------------------------------------------------------------------
    # handle_regime
    # ------------------------------------------------------------------

    def test_handle_regime_returns_dict(self):
        api = self._make_api()
        result = api.handle_regime()
        self.assertIn("regime", result)
        self.assertEqual(result["regime"], "trending_up")

    def test_handle_regime_includes_confidence(self):
        api = self._make_api()
        result = api.handle_regime()
        self.assertAlmostEqual(result["confidence"], 0.85)

    # ------------------------------------------------------------------
    # handle_health
    # ------------------------------------------------------------------

    def test_handle_health_ok(self):
        api = self._make_api()
        result = api.handle_health()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["redis"], "ok")
        self.assertEqual(result["timescale"], "ok")

    def test_handle_health_redis_fail(self):
        api = self._make_api()
        api.redis.get_account_state.side_effect = Exception("fail")
        result = api.handle_health()
        self.assertEqual(result["redis"], "error")
        self.assertEqual(result["status"], "degraded")

    # ------------------------------------------------------------------
    # handle_wallets
    # ------------------------------------------------------------------

    def test_handle_wallets_returns_list(self):
        api = self._make_api()
        result = api.handle_wallets()
        self.assertIn("wallets", result)
        self.assertEqual(len(result["wallets"]), 1)

    def test_handle_wallets_includes_data(self):
        api = self._make_api()
        result = api.handle_wallets()
        self.assertEqual(result["wallets"][0]["address"], "0xabc")

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_data_handled(self):
        api = self._make_api(redis_data={})
        api.redis.get_all_positions.return_value = []
        api.redis.get_market_regime.return_value = None
        api.timescale.query_decisions_by_type.return_value = []
        api.timescale.query_wallet_scores.return_value = []

        # All endpoints should return without error
        self.assertIsInstance(api.handle_status(), dict)
        self.assertIsInstance(api.handle_portfolio(), dict)
        self.assertIsInstance(api.handle_positions(), dict)
        self.assertIsInstance(api.handle_decisions(), dict)
        self.assertIsInstance(api.handle_regime(), dict)
        self.assertIsInstance(api.handle_wallets(), dict)

    def test_redis_exception_handled_in_status(self):
        api = self._make_api()
        api.redis.get_account_state.side_effect = Exception("connection error")
        result = api.handle_status()
        self.assertEqual(result["status"], "running")
        self.assertEqual(result["equity"], 0)

    def test_timescale_exception_handled_in_decisions(self):
        api = self._make_api()
        api.timescale.query_decisions_by_type.side_effect = Exception("db error")
        result = api.handle_decisions()
        self.assertEqual(result["decisions"], [])

    def test_handle_regime_redis_exception(self):
        api = self._make_api()
        api.redis.get_market_regime.side_effect = Exception("fail")
        result = api.handle_regime()
        self.assertEqual(result["regime"], "unknown")

    def test_handle_wallets_no_method(self):
        api = self._make_api()
        del api.timescale.query_wallet_scores
        result = api.handle_wallets()
        self.assertEqual(result["wallets"], [])


if __name__ == "__main__":
    unittest.main()
