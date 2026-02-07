"""Tests for reporting.report_generator.ReportGenerator."""

import os
import sys
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from reporting.report_generator import ReportGenerator


class TestReportGenerator(unittest.TestCase):
    """Tests for ReportGenerator."""

    def setUp(self):
        self.mock_tracker = MagicMock()
        self.mock_health_scorer = MagicMock()
        self.mock_timescale = MagicMock()

        self.generator = ReportGenerator(
            performance_tracker=self.mock_tracker,
            health_scorer=self.mock_health_scorer,
            timescale=self.mock_timescale,
            strategy_name="funding_rate_arb",
        )

    def _mock_metrics(self):
        return {
            "trade_count": 50,
            "total_pnl": 250.50,
            "win_rate": 62.0,
            "avg_win": 10.0,
            "avg_loss": -5.0,
            "profit_factor": 2.1,
            "sharpe_ratio": 1.8,
            "sortino_ratio": 2.5,
            "max_drawdown": 3.2,
            "calmar_ratio": 2.8,
        }

    def _mock_health(self, score=72.5, grade="B"):
        return {
            "health_score": score,
            "grade": grade,
            "components": {
                "sharpe": 60.0,
                "sortino": 62.5,
                "max_drawdown": 68.0,
                "win_rate": 62.0,
                "profit_factor": 55.0,
                "calmar": 56.0,
            },
            "raw_metrics": self._mock_metrics(),
        }

    def test_hourly_report(self):
        """Hourly report should contain key metrics."""
        self.mock_tracker.compute_metrics.return_value = self._mock_metrics()
        self.mock_health_scorer.compute_health_score.return_value = self._mock_health()

        report = self.generator.generate_hourly_report()

        self.assertIn("HOURLY STATUS REPORT", report)
        self.assertIn("funding_rate_arb", report)
        self.assertIn("Health Score", report)

    def test_weekly_report(self):
        """Weekly report should have all required sections."""
        self.mock_tracker.compute_metrics.return_value = self._mock_metrics()
        self.mock_health_scorer.compute_health_score.return_value = self._mock_health()
        self.mock_timescale.query_decisions.return_value = [
            {"decision_type": "signal_generated", "time": datetime.now(timezone.utc)},
            {"decision_type": "signal_generated", "time": datetime.now(timezone.utc)},
        ]

        report = self.generator.generate_weekly_report()

        self.assertIn("WEEKLY PERFORMANCE REPORT", report)
        self.assertIn("PERFORMANCE SUMMARY", report)
        self.assertIn("STRATEGY HEALTH", report)
        self.assertIn("RISK EVENTS", report)
        self.assertIn("NOTABLE DECISIONS", report)
        self.assertIn("RECOMMENDATIONS", report)
        self.assertIn("Total Trades: 50", report)
        self.assertIn("Total Signals: 2", report)

    def test_monthly_report(self):
        """Monthly report should have all required sections."""
        self.mock_tracker.compute_metrics.return_value = self._mock_metrics()
        self.mock_health_scorer.compute_health_score.return_value = self._mock_health()
        self.mock_timescale.query_decisions.return_value = []

        report = self.generator.generate_monthly_report()

        self.assertIn("MONTHLY PERFORMANCE REPORT", report)
        self.assertIn("MONTHLY OVERVIEW", report)
        self.assertIn("STRATEGY RANKING", report)
        self.assertIn("ATTRIBUTION ANALYSIS", report)
        self.assertIn("DECISION AUDIT", report)
        self.assertIn("RECOMMENDATIONS", report)

    def test_recommendations_critical(self):
        """Low health score should generate critical recommendation."""
        self.mock_tracker.compute_metrics.return_value = self._mock_metrics()
        self.mock_health_scorer.compute_health_score.return_value = self._mock_health(
            score=15.0, grade="F")
        self.mock_timescale.query_decisions.return_value = []

        report = self.generator.generate_weekly_report()
        self.assertIn("CRITICAL", report)

    def test_recommendations_high_drawdown(self):
        """High drawdown should trigger recommendation."""
        metrics = self._mock_metrics()
        metrics["max_drawdown"] = 7.5
        self.mock_tracker.compute_metrics.return_value = metrics
        self.mock_health_scorer.compute_health_score.return_value = self._mock_health()
        self.mock_timescale.query_decisions.return_value = []

        report = self.generator.generate_weekly_report()
        self.assertIn("Drawdown", report)

    def test_recommendations_healthy(self):
        """Healthy strategy should get positive recommendation."""
        self.mock_tracker.compute_metrics.return_value = self._mock_metrics()
        self.mock_health_scorer.compute_health_score.return_value = self._mock_health(
            score=85.0, grade="A")
        self.mock_timescale.query_decisions.return_value = []

        report = self.generator.generate_weekly_report()
        self.assertIn("performing well", report)

    def test_report_handles_exceptions(self):
        """Reports should handle exceptions gracefully."""
        self.mock_tracker.compute_metrics.side_effect = Exception("DB error")
        self.mock_health_scorer.compute_health_score.side_effect = Exception("Error")
        self.mock_timescale.query_decisions.side_effect = Exception("Error")

        # Should not raise
        report = self.generator.generate_weekly_report()
        self.assertIn("WEEKLY PERFORMANCE REPORT", report)

    def test_report_returns_string(self):
        """All report methods should return strings."""
        self.mock_tracker.compute_metrics.return_value = self._mock_metrics()
        self.mock_health_scorer.compute_health_score.return_value = self._mock_health()
        self.mock_timescale.query_decisions.return_value = []

        self.assertIsInstance(self.generator.generate_hourly_report(), str)
        self.assertIsInstance(self.generator.generate_weekly_report(), str)
        self.assertIsInstance(self.generator.generate_monthly_report(), str)


if __name__ == "__main__":
    unittest.main()
