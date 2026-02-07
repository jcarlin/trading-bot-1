#!/usr/bin/env python3
"""Tests for portfolio performance tracker."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from metrics.portfolio_performance import PortfolioPerformanceTracker


class TestPortfolioPerformanceTracker(unittest.TestCase):

    def setUp(self):
        self.timescale = MagicMock()
        self.strategy_names = ["strategy_a", "strategy_b"]
        self.tracker = PortfolioPerformanceTracker(
            self.timescale, self.strategy_names)
        self.base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)

    def test_portfolio_metrics_no_data(self):
        self.timescale.query_fills_by_strategy.return_value = []
        self.timescale.query_equity_snapshots.return_value = []
        result = self.tracker.compute_portfolio_metrics(24)
        self.assertEqual(result["total_pnl"], 0.0)
        self.assertEqual(result["portfolio_sharpe"], 0.0)
        self.assertIn("strategy_contributions", result)

    def test_portfolio_metrics_with_fills(self):
        def fills_side_effect(name, start, end):
            if name == "strategy_a":
                return [{"closed_pnl": 10.0}, {"closed_pnl": -3.0}]
            return [{"closed_pnl": 5.0}]

        self.timescale.query_fills_by_strategy.side_effect = fills_side_effect
        self.timescale.query_equity_snapshots.return_value = []
        result = self.tracker.compute_portfolio_metrics(24)
        self.assertAlmostEqual(result["total_pnl"], 12.0, places=2)
        self.assertAlmostEqual(result["strategy_contributions"]["strategy_a"], 7.0, places=2)
        self.assertAlmostEqual(result["strategy_contributions"]["strategy_b"], 5.0, places=2)

    def test_portfolio_metrics_with_equity(self):
        self.timescale.query_fills_by_strategy.return_value = [{"closed_pnl": 5.0}]
        self.timescale.query_equity_snapshots.return_value = [
            {"total_equity": 10000.0},
            {"total_equity": 10100.0},
            {"total_equity": 10050.0},
            {"total_equity": 10200.0},
        ]
        result = self.tracker.compute_portfolio_metrics(24)
        self.assertGreater(result["portfolio_max_dd"], 0)

    def test_portfolio_metrics_sharpe_positive(self):
        """Growing equity should produce positive Sharpe."""
        self.timescale.query_fills_by_strategy.return_value = []
        self.timescale.query_equity_snapshots.return_value = [
            {"total_equity": 10000.0 + i * 10.0} for i in range(50)
        ]
        result = self.tracker.compute_portfolio_metrics(168)
        self.assertGreater(result["portfolio_sharpe"], 0)

    def test_benchmark_comparison_no_data(self):
        self.timescale.query_equity_snapshots.return_value = []
        self.timescale.query_candles.return_value = []
        result = self.tracker.compute_benchmark_comparison(24)
        self.assertEqual(result["portfolio_return"], 0.0)
        self.assertEqual(result["btc_return"], 0.0)
        self.assertEqual(result["eth_return"], 0.0)

    def test_benchmark_comparison_with_data(self):
        self.timescale.query_equity_snapshots.return_value = [
            {"total_equity": 10000.0},
            {"total_equity": 10500.0},
        ]
        self.timescale.query_candles.return_value = [
            {"close": 50000.0},
            {"close": 51000.0},
        ]
        result = self.tracker.compute_benchmark_comparison(24)
        self.assertAlmostEqual(result["portfolio_return"], 0.05, places=4)
        self.assertAlmostEqual(result["btc_return"], 0.02, places=4)
        self.assertGreater(result["alpha"], 0)

    def test_benchmark_comparison_structure(self):
        self.timescale.query_equity_snapshots.return_value = []
        self.timescale.query_candles.return_value = []
        result = self.tracker.compute_benchmark_comparison(24)
        expected_keys = {"portfolio_return", "btc_return", "eth_return",
                        "alpha", "beta", "tracking_error"}
        self.assertEqual(set(result.keys()), expected_keys)

    def test_capital_efficiency(self):
        """PnL divided by avg equity."""
        self.timescale.query_fills_by_strategy.return_value = [{"closed_pnl": 100.0}]
        self.timescale.query_equity_snapshots.return_value = [
            {"total_equity": 10000.0},
            {"total_equity": 10100.0},
        ]
        result = self.tracker.compute_portfolio_metrics(24)
        # Both strategies return same fills, so total pnl = 200
        self.assertGreater(result["capital_efficiency"], 0)

    def test_query_failure_handled(self):
        self.timescale.query_fills_by_strategy.side_effect = Exception("DB error")
        self.timescale.query_equity_snapshots.return_value = []
        result = self.tracker.compute_portfolio_metrics(24)
        self.assertEqual(result["total_pnl"], 0.0)


if __name__ == "__main__":
    unittest.main()
