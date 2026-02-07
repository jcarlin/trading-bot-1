#!/usr/bin/env python3
"""Tests for promotion criteria."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest

from orchestration.promotion_criteria import PromotionCriteria


class TestPromotionCriteria(unittest.TestCase):

    def _make_shadow_metrics(self, **overrides):
        defaults = {
            "total_pnl": 1000.0,
            "trade_count": 30,
            "win_rate": 55.0,
            "max_drawdown": 3.0,
            "sharpe": 1.2,
            "wins": 17,
            "losses": 13,
        }
        defaults.update(overrides)
        return defaults

    def _make_live_metrics(self, **overrides):
        defaults = {
            "total_pnl": 500.0,
            "trade_count": 25,
            "win_rate": 48.0,
            "sharpe_ratio": 0.8,
            "max_drawdown": 4.0,
        }
        defaults.update(overrides)
        return defaults

    def test_all_criteria_met(self):
        """All criteria met should return meets_criteria=True."""
        criteria = PromotionCriteria()
        shadow = self._make_shadow_metrics()
        live = self._make_live_metrics()

        result = criteria.evaluate(shadow, live, duration_hours=200)

        self.assertTrue(result["meets_criteria"])
        for check in result["checks"].values():
            self.assertTrue(check["passed"])

    def test_fails_duration(self):
        """Should fail when duration is below minimum."""
        criteria = PromotionCriteria()
        shadow = self._make_shadow_metrics()
        live = self._make_live_metrics()

        result = criteria.evaluate(shadow, live, duration_hours=24)

        self.assertFalse(result["meets_criteria"])
        self.assertFalse(result["checks"]["duration"]["passed"])
        self.assertEqual(result["checks"]["duration"]["actual"], 24)

    def test_fails_min_trades(self):
        """Should fail when trade count is below minimum."""
        criteria = PromotionCriteria()
        shadow = self._make_shadow_metrics(trade_count=5)
        live = self._make_live_metrics()

        result = criteria.evaluate(shadow, live, duration_hours=200)

        self.assertFalse(result["meets_criteria"])
        self.assertFalse(result["checks"]["trades"]["passed"])

    def test_fails_min_sharpe(self):
        """Should fail when Sharpe is below minimum."""
        criteria = PromotionCriteria()
        shadow = self._make_shadow_metrics(sharpe=0.2)
        live = self._make_live_metrics()

        result = criteria.evaluate(shadow, live, duration_hours=200)

        self.assertFalse(result["meets_criteria"])
        self.assertFalse(result["checks"]["sharpe"]["passed"])

    def test_fails_min_win_rate(self):
        """Should fail when win rate is below minimum."""
        criteria = PromotionCriteria()
        shadow = self._make_shadow_metrics(win_rate=30.0)
        live = self._make_live_metrics()

        result = criteria.evaluate(shadow, live, duration_hours=200)

        self.assertFalse(result["meets_criteria"])
        self.assertFalse(result["checks"]["win_rate"]["passed"])

    def test_fails_max_drawdown(self):
        """Should fail when drawdown exceeds maximum."""
        criteria = PromotionCriteria()
        shadow = self._make_shadow_metrics(max_drawdown=8.0)
        live = self._make_live_metrics()

        result = criteria.evaluate(shadow, live, duration_hours=200)

        self.assertFalse(result["meets_criteria"])
        self.assertFalse(result["checks"]["drawdown"]["passed"])

    def test_fails_improvement_pct(self):
        """Should fail when improvement is below threshold."""
        criteria = PromotionCriteria()
        # Shadow PnL barely above live — less than 10% improvement
        shadow = self._make_shadow_metrics(total_pnl=520.0)
        live = self._make_live_metrics(total_pnl=500.0)

        result = criteria.evaluate(shadow, live, duration_hours=200)

        self.assertFalse(result["meets_criteria"])
        self.assertFalse(result["checks"]["improvement"]["passed"])

    def test_config_defaults(self):
        """Default config values should be set correctly."""
        criteria = PromotionCriteria()
        self.assertEqual(criteria.min_duration_hours, 168)
        self.assertEqual(criteria.min_trades, 20)
        self.assertEqual(criteria.min_sharpe, 0.5)
        self.assertEqual(criteria.min_win_rate, 45.0)
        self.assertEqual(criteria.max_dd, 5.0)
        self.assertEqual(criteria.min_improvement_pct, 10.0)

    def test_custom_config(self):
        """Custom config should override defaults."""
        config = {
            "min_duration_hours": 48,
            "min_trades": 10,
            "min_sharpe": 1.0,
            "min_win_rate": 60.0,
            "max_dd": 3.0,
            "min_improvement_pct": 20.0,
        }
        criteria = PromotionCriteria(config)
        self.assertEqual(criteria.min_duration_hours, 48)
        self.assertEqual(criteria.min_trades, 10)
        self.assertEqual(criteria.min_sharpe, 1.0)
        self.assertEqual(criteria.min_win_rate, 60.0)
        self.assertEqual(criteria.max_dd, 3.0)
        self.assertEqual(criteria.min_improvement_pct, 20.0)

    def test_statistical_significance_significant(self):
        """Should detect significant difference between distributions."""
        criteria = PromotionCriteria()
        # Shadow clearly better than live
        shadow_returns = [100.0, 120.0, 110.0, 130.0, 105.0,
                          115.0, 125.0, 140.0, 108.0, 112.0]
        live_returns = [10.0, 15.0, 12.0, 8.0, 11.0,
                        9.0, 14.0, 13.0, 7.0, 10.0]

        is_sig, p_val = criteria.is_statistically_significant(
            shadow_returns, live_returns)

        self.assertTrue(is_sig)
        self.assertLess(p_val, 0.05)

    def test_statistical_significance_not_significant(self):
        """Should not flag significance for similar distributions."""
        criteria = PromotionCriteria()
        # Very similar distributions
        shadow_returns = [10.0, 11.0, 9.0, 10.5, 10.2]
        live_returns = [10.1, 10.9, 9.1, 10.4, 10.3]

        is_sig, p_val = criteria.is_statistically_significant(
            shadow_returns, live_returns)

        self.assertFalse(is_sig)
        self.assertGreater(p_val, 0.05)

    def test_equal_distributions(self):
        """Identical distributions should not be significant."""
        criteria = PromotionCriteria()
        returns = [10.0, 20.0, 15.0, 12.0, 18.0]

        is_sig, p_val = criteria.is_statistically_significant(returns, returns)

        self.assertFalse(is_sig)
        # p-value should be 1.0 (or very close) for identical data
        self.assertGreaterEqual(p_val, 0.9)


if __name__ == "__main__":
    unittest.main()
