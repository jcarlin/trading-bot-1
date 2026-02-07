#!/usr/bin/env python3
"""Tests for correlation analyzer and portfolio performance tracker."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np

from metrics.correlation import CorrelationAnalyzer


class TestCorrelationAnalyzer(unittest.TestCase):

    def setUp(self):
        self.timescale = MagicMock()
        self.strategy_names = ["strategy_a", "strategy_b", "strategy_c"]
        self.analyzer = CorrelationAnalyzer(self.timescale, self.strategy_names)
        self.base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
        # For tests that call compute_* methods, we need fills to be within
        # the window. The methods compute start = now - window_hours, so we
        # generate fills relative to a fixed "now" and patch datetime.
        self.fixed_now = self.base_time + timedelta(hours=24)

    def _make_fills(self, pnl_values, start_time):
        """Create mock fills with given PnL values, each 1 hour apart."""
        fills = []
        for i, pnl in enumerate(pnl_values):
            fills.append({
                "time": start_time + timedelta(hours=i),
                "closed_pnl": pnl,
            })
        return fills

    def _patch_now(self):
        """Return a patcher that fixes datetime.now to self.fixed_now."""
        import metrics.correlation as mod
        original_datetime = datetime

        class MockDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return self.fixed_now
        return patch.object(mod, 'datetime', MockDatetime)

    def test_correlation_matrix_empty_strategies(self):
        analyzer = CorrelationAnalyzer(self.timescale, [])
        result = analyzer.compute_correlation_matrix(24)
        self.assertEqual(result["matrix"], [])
        self.assertEqual(result["strategy_names"], [])

    def test_correlation_matrix_single_strategy(self):
        self.timescale.query_fills_by_strategy.return_value = []
        analyzer = CorrelationAnalyzer(self.timescale, ["single"])
        result = analyzer.compute_correlation_matrix(24)
        self.assertEqual(len(result["matrix"]), 1)
        self.assertEqual(result["matrix"][0], [1.0])

    def test_correlation_matrix_perfectly_correlated(self):
        """Two strategies with identical PnL should have correlation 1.0."""
        fills = self._make_fills([1.0, -0.5, 2.0, -1.0, 0.5], self.base_time)
        self.timescale.query_fills_by_strategy.return_value = fills

        analyzer = CorrelationAnalyzer(self.timescale, ["a", "b"])
        with self._patch_now():
            result = analyzer.compute_correlation_matrix(24)

        self.assertEqual(len(result["matrix"]), 2)
        self.assertAlmostEqual(result["matrix"][0][1], 1.0, places=3)
        self.assertAlmostEqual(result["matrix"][1][0], 1.0, places=3)

    def test_correlation_matrix_uncorrelated(self):
        """Two strategies with orthogonal PnL patterns."""
        fills_a = self._make_fills([1.0, 0.0, 1.0, 0.0], self.base_time)
        fills_b = self._make_fills([0.0, 1.0, 0.0, 1.0], self.base_time)

        def side_effect(name, start, end):
            if name == "a":
                return fills_a
            return fills_b

        self.timescale.query_fills_by_strategy.side_effect = side_effect

        analyzer = CorrelationAnalyzer(self.timescale, ["a", "b"])
        with self._patch_now():
            result = analyzer.compute_correlation_matrix(24)

        # Diagonal should be 1.0
        self.assertAlmostEqual(result["matrix"][0][0], 1.0, places=3)
        self.assertAlmostEqual(result["matrix"][1][1], 1.0, places=3)
        # Off-diagonal should be negative or near -1.0
        self.assertLess(result["matrix"][0][1], 0.5)

    def test_correlation_matrix_no_fills(self):
        """All strategies have zero PnL -> identity matrix (constant series)."""
        self.timescale.query_fills_by_strategy.return_value = []
        result = self.analyzer.compute_correlation_matrix(24)
        # With zero variance, should return identity
        for i in range(3):
            self.assertAlmostEqual(result["matrix"][i][i], 1.0, places=3)

    def test_correlation_period_hours(self):
        self.timescale.query_fills_by_strategy.return_value = []
        result = self.analyzer.compute_correlation_matrix(48)
        self.assertEqual(result["period_hours"], 48)

    def test_marginal_contribution_empty(self):
        analyzer = CorrelationAnalyzer(self.timescale, [])
        result = analyzer.compute_marginal_contribution(24)
        self.assertEqual(result, {})

    def test_marginal_contribution_returns_all_strategies(self):
        self.timescale.query_fills_by_strategy.return_value = []
        result = self.analyzer.compute_marginal_contribution(24)
        self.assertIn("strategy_a", result)
        self.assertIn("strategy_b", result)
        self.assertIn("strategy_c", result)

    def test_marginal_contribution_structure(self):
        fills = self._make_fills([1.0, 2.0, -1.0], self.base_time)
        self.timescale.query_fills_by_strategy.return_value = fills
        with self._patch_now():
            result = self.analyzer.compute_marginal_contribution(24)
        for name in self.strategy_names:
            self.assertIn("marginal_risk", result[name])
            self.assertIn("marginal_return", result[name])
            self.assertIn("info_ratio", result[name])

    def test_bin_fills_hourly(self):
        start = self.base_time
        end = start + timedelta(hours=5)
        fills = [
            {"time": start + timedelta(hours=0, minutes=30), "closed_pnl": 10.0},
            {"time": start + timedelta(hours=0, minutes=45), "closed_pnl": 5.0},
            {"time": start + timedelta(hours=2, minutes=15), "closed_pnl": -3.0},
        ]
        buckets = self.analyzer._bin_fills_hourly(fills, start, end)
        self.assertEqual(len(buckets), 5)
        self.assertAlmostEqual(buckets[0], 15.0)  # Two fills in hour 0
        self.assertAlmostEqual(buckets[1], 0.0)
        self.assertAlmostEqual(buckets[2], -3.0)

    def test_bin_fills_hourly_string_times(self):
        """Handles string timestamps correctly."""
        start = self.base_time
        end = start + timedelta(hours=3)
        fills = [
            {"time": (start + timedelta(hours=1, minutes=10)).isoformat(), "closed_pnl": 7.0},
        ]
        buckets = self.analyzer._bin_fills_hourly(fills, start, end)
        self.assertAlmostEqual(buckets[1], 7.0)


class TestCorrelationAnalyzerEdgeCases(unittest.TestCase):

    def test_query_failure_handled(self):
        timescale = MagicMock()
        timescale.query_fills_by_strategy.side_effect = Exception("DB error")
        analyzer = CorrelationAnalyzer(timescale, ["a", "b"])
        # Should not raise
        result = analyzer.compute_correlation_matrix(24)
        self.assertIsNotNone(result["matrix"])


if __name__ == "__main__":
    unittest.main()
