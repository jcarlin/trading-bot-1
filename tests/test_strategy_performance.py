#!/usr/bin/env python3
"""Tests for the strategy performance tracker."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch


class TestStrategyPerformanceTracker(unittest.TestCase):

    def setUp(self):
        self.mock_timescale = MagicMock()
        from metrics.strategy_performance import StrategyPerformanceTracker
        self.tracker = StrategyPerformanceTracker(
            self.mock_timescale, "funding_rate_arb")

    def test_empty_fills_returns_zeros(self):
        self.mock_timescale.query_fills_by_strategy.return_value = []
        self.mock_timescale.query_equity_snapshots.return_value = []
        metrics = self.tracker.compute_metrics(24)
        self.assertEqual(metrics["trade_count"], 0)
        self.assertEqual(metrics["total_pnl"], 0.0)
        self.assertEqual(metrics["win_rate"], 0.0)
        self.assertEqual(metrics["sharpe_ratio"], 0.0)

    def test_compute_with_mixed_fills(self):
        fills = [
            {"closed_pnl": 100.0, "time": datetime(2025, 1, 1, tzinfo=timezone.utc)},
            {"closed_pnl": 50.0, "time": datetime(2025, 1, 1, 1, tzinfo=timezone.utc)},
            {"closed_pnl": -30.0, "time": datetime(2025, 1, 1, 2, tzinfo=timezone.utc)},
            {"closed_pnl": -20.0, "time": datetime(2025, 1, 1, 3, tzinfo=timezone.utc)},
            {"closed_pnl": 80.0, "time": datetime(2025, 1, 1, 4, tzinfo=timezone.utc)},
        ]
        self.mock_timescale.query_fills_by_strategy.return_value = fills
        self.mock_timescale.query_equity_snapshots.return_value = []

        metrics = self.tracker.compute_metrics(24)
        self.assertEqual(metrics["trade_count"], 5)
        self.assertEqual(metrics["total_pnl"], 180.0)
        self.assertEqual(metrics["win_rate"], 60.0)  # 3/5
        self.assertAlmostEqual(metrics["avg_win"], (100 + 50 + 80) / 3, places=2)
        self.assertAlmostEqual(metrics["avg_loss"], (-30 + -20) / 2, places=2)

    def test_win_rate_calculation(self):
        fills = [
            {"closed_pnl": 10.0},
            {"closed_pnl": 20.0},
            {"closed_pnl": 30.0},
            {"closed_pnl": -5.0},
            {"closed_pnl": -10.0},
        ]
        self.mock_timescale.query_fills_by_strategy.return_value = fills
        self.mock_timescale.query_equity_snapshots.return_value = []
        metrics = self.tracker.compute_metrics(24)
        self.assertEqual(metrics["win_rate"], 60.0)

    def test_profit_factor_no_losses(self):
        fills = [
            {"closed_pnl": 100.0},
            {"closed_pnl": 50.0},
        ]
        self.mock_timescale.query_fills_by_strategy.return_value = fills
        self.mock_timescale.query_equity_snapshots.return_value = []
        metrics = self.tracker.compute_metrics(24)
        self.assertEqual(metrics["profit_factor"], float("inf"))

    def test_profit_factor_with_losses(self):
        fills = [
            {"closed_pnl": 100.0},
            {"closed_pnl": -50.0},
        ]
        self.mock_timescale.query_fills_by_strategy.return_value = fills
        self.mock_timescale.query_equity_snapshots.return_value = []
        metrics = self.tracker.compute_metrics(24)
        self.assertAlmostEqual(metrics["profit_factor"], 2.0, places=2)

    def test_max_drawdown_from_equity(self):
        fills = [{"closed_pnl": 10.0}]
        equity_snapshots = [
            {"total_equity": 10000.0},
            {"total_equity": 10500.0},
            {"total_equity": 10200.0},  # DD from 10500 = 2.857%
            {"total_equity": 10800.0},
        ]
        self.mock_timescale.query_fills_by_strategy.return_value = fills
        self.mock_timescale.query_equity_snapshots.return_value = equity_snapshots
        metrics = self.tracker.compute_metrics(24)
        expected_dd = (10500 - 10200) / 10500 * 100
        self.assertAlmostEqual(metrics["max_drawdown"], expected_dd, places=1)

    def test_compute_all_windows(self):
        self.mock_timescale.query_fills_by_strategy.return_value = []
        self.mock_timescale.query_equity_snapshots.return_value = []
        results = self.tracker.compute_all_windows([1, 4, 24])
        self.assertIn(1, results)
        self.assertIn(4, results)
        self.assertIn(24, results)
        for metrics in results.values():
            self.assertEqual(metrics["trade_count"], 0)

    def test_sharpe_with_equity_data(self):
        fills = [{"closed_pnl": 10.0}]
        # Create equity snapshots with positive trend
        equity_snapshots = [
            {"total_equity": 10000.0},
            {"total_equity": 10100.0},
            {"total_equity": 10200.0},
            {"total_equity": 10300.0},
            {"total_equity": 10400.0},
        ]
        self.mock_timescale.query_fills_by_strategy.return_value = fills
        self.mock_timescale.query_equity_snapshots.return_value = equity_snapshots
        metrics = self.tracker.compute_metrics(24)
        self.assertGreater(metrics["sharpe_ratio"], 0)

    def test_handles_query_exception(self):
        self.mock_timescale.query_fills_by_strategy.side_effect = Exception("DB error")
        self.mock_timescale.query_equity_snapshots.return_value = []
        metrics = self.tracker.compute_metrics(24)
        self.assertEqual(metrics["trade_count"], 0)


if __name__ == "__main__":
    unittest.main()
