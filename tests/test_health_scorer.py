"""Tests for metrics.health_scorer.StrategyHealthScorer."""

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from metrics.health_scorer import StrategyHealthScorer


class TestStrategyHealthScorer(unittest.TestCase):
    """Tests for StrategyHealthScorer."""

    def setUp(self):
        self.mock_tracker = MagicMock()
        self.scorer = StrategyHealthScorer(self.mock_tracker)

    def _mock_metrics(self, sharpe=1.5, sortino=2.0, max_dd=3.0,
                      win_rate=55.0, profit_factor=1.5, calmar=2.0,
                      trade_count=50, total_pnl=100.0):
        """Create mock metrics dict."""
        return {
            "trade_count": trade_count,
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "avg_win": 5.0,
            "avg_loss": -3.0,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
        }

    def test_healthy_strategy_gets_high_score(self):
        """A strategy with good metrics should score high."""
        self.mock_tracker.compute_metrics.return_value = self._mock_metrics(
            sharpe=2.5, sortino=3.0, max_dd=1.0, win_rate=65.0,
            profit_factor=2.5, calmar=4.0)

        result = self.scorer.compute_health_score(168)
        self.assertGreaterEqual(result["health_score"], 70)
        self.assertIn(result["grade"], ["A", "B"])

    def test_poor_strategy_gets_low_score(self):
        """A strategy with poor metrics should score low."""
        self.mock_tracker.compute_metrics.return_value = self._mock_metrics(
            sharpe=-0.5, sortino=-0.3, max_dd=8.0, win_rate=30.0,
            profit_factor=0.5, calmar=-1.0)

        result = self.scorer.compute_health_score(168)
        self.assertLessEqual(result["health_score"], 30)
        self.assertIn(result["grade"], ["D", "F"])

    def test_grade_a(self):
        """Score >= 80 should get grade A."""
        self.mock_tracker.compute_metrics.return_value = self._mock_metrics(
            sharpe=3.0, sortino=4.0, max_dd=0.5, win_rate=80.0,
            profit_factor=3.0, calmar=5.0)

        result = self.scorer.compute_health_score(168)
        self.assertEqual(result["grade"], "A")

    def test_grade_f(self):
        """Score < 20 should get grade F."""
        self.mock_tracker.compute_metrics.return_value = self._mock_metrics(
            sharpe=0.0, sortino=0.0, max_dd=10.0, win_rate=10.0,
            profit_factor=0.5, calmar=0.0)

        result = self.scorer.compute_health_score(168)
        self.assertEqual(result["grade"], "F")

    def test_should_pause_low_score(self):
        """Should pause when health score < 20."""
        self.mock_tracker.compute_metrics.return_value = self._mock_metrics(
            sharpe=0.0, sortino=0.0, max_dd=9.0, win_rate=10.0,
            profit_factor=0.5, calmar=0.0)

        should_pause, reason = self.scorer.should_pause(168)
        self.assertTrue(should_pause)
        self.assertIn("critically low", reason.lower())

    def test_should_pause_high_drawdown(self):
        """Should pause when max drawdown > 5%."""
        self.mock_tracker.compute_metrics.return_value = self._mock_metrics(
            sharpe=2.0, sortino=2.5, max_dd=6.0, win_rate=60.0,
            profit_factor=2.0, calmar=3.0)

        should_pause, reason = self.scorer.should_pause(168)
        self.assertTrue(should_pause)
        self.assertIn("drawdown", reason.lower())

    def test_should_not_pause_healthy(self):
        """Should not pause when strategy is healthy."""
        self.mock_tracker.compute_metrics.return_value = self._mock_metrics(
            sharpe=2.0, sortino=2.5, max_dd=2.0, win_rate=60.0,
            profit_factor=2.0, calmar=3.0)

        should_pause, reason = self.scorer.should_pause(168)
        self.assertFalse(should_pause)
        self.assertEqual(reason, "")

    def test_custom_weights(self):
        """Custom weights should be used."""
        custom_weights = {"sharpe": 1.0, "sortino": 0, "max_drawdown": 0,
                          "win_rate": 0, "profit_factor": 0, "calmar": 0}
        scorer = StrategyHealthScorer(self.mock_tracker, weights=custom_weights)

        self.mock_tracker.compute_metrics.return_value = self._mock_metrics(sharpe=1.5)
        result = scorer.compute_health_score(168)
        # Sharpe 1.5 normalized to 0-3 range = 50%
        self.assertAlmostEqual(result["health_score"], 50.0, places=0)

    def test_result_structure(self):
        """Result should have all expected keys."""
        self.mock_tracker.compute_metrics.return_value = self._mock_metrics()
        result = self.scorer.compute_health_score(168)

        self.assertIn("health_score", result)
        self.assertIn("grade", result)
        self.assertIn("components", result)
        self.assertIn("raw_metrics", result)

    def test_score_bounded(self):
        """Score should be between 0 and 100."""
        # Very high metrics
        self.mock_tracker.compute_metrics.return_value = self._mock_metrics(
            sharpe=10.0, sortino=15.0, max_dd=0.0, win_rate=100.0,
            profit_factor=10.0, calmar=20.0)

        result = self.scorer.compute_health_score(168)
        self.assertLessEqual(result["health_score"], 100.0)
        self.assertGreaterEqual(result["health_score"], 0.0)

    def test_normalize_inverted(self):
        """Inverted normalization (drawdown) should work correctly."""
        # 0% drawdown = 100 score
        self.assertEqual(StrategyHealthScorer._normalize(0.0, 10.0, 0.0), 100.0)
        # 10% drawdown = 0 score
        self.assertEqual(StrategyHealthScorer._normalize(10.0, 10.0, 0.0), 0.0)
        # 5% drawdown = 50 score
        self.assertEqual(StrategyHealthScorer._normalize(5.0, 10.0, 0.0), 50.0)

    def test_normalize_normal(self):
        """Normal normalization should work correctly."""
        # Sharpe 0 = 0 score
        self.assertEqual(StrategyHealthScorer._normalize(0.0, 0.0, 3.0), 0.0)
        # Sharpe 3 = 100 score
        self.assertEqual(StrategyHealthScorer._normalize(3.0, 0.0, 3.0), 100.0)
        # Sharpe 1.5 = 50 score
        self.assertEqual(StrategyHealthScorer._normalize(1.5, 0.0, 3.0), 50.0)


if __name__ == "__main__":
    unittest.main()
