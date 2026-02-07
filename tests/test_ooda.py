"""Tests for orchestration.ooda.OODAOrchestrator."""

import asyncio
import os
import sys
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from orchestration.ooda import OODAOrchestrator


class TestOODAOrchestrator(unittest.TestCase):
    """Tests for OODAOrchestrator."""

    def setUp(self):
        self.mock_strategy_manager = MagicMock()
        self.mock_strategy_manager.pause_strategy = AsyncMock()
        self.mock_strategy_manager.resume_strategy = AsyncMock()
        self.mock_strategy_manager.get_active_strategies.return_value = ["funding_rate_arb"]

        self.mock_health_scorer = MagicMock()
        self.mock_regime_classifier = MagicMock()
        self.mock_timescale = MagicMock()
        self.mock_redis = MagicMock()

        self.config = {
            "symbol": "BTC/USDC",
            "strategy_name": "funding_rate_arb",
            "decay_pause_threshold": 50.0,
            "health_pause_threshold": 20,
        }

        self.orchestrator = OODAOrchestrator(
            strategy_manager=self.mock_strategy_manager,
            health_scorer=self.mock_health_scorer,
            regime_classifier=self.mock_regime_classifier,
            timescale=self.mock_timescale,
            redis_store=self.mock_redis,
            config=self.config,
        )

    def _set_healthy_state(self):
        """Configure mocks for a healthy strategy state."""
        self.mock_health_scorer.compute_health_score.return_value = {
            "health_score": 75.0,
            "grade": "B",
            "components": {},
            "raw_metrics": {"max_drawdown": 2.0},
        }
        self.mock_timescale.query_candles.return_value = []
        self.mock_regime_classifier.classify.return_value = {
            "regime": "ranging",
            "confidence": 0.7,
            "indicators": {},
        }

    def _set_unhealthy_state(self):
        """Configure mocks for an unhealthy strategy state."""
        self.mock_health_scorer.compute_health_score.return_value = {
            "health_score": 15.0,
            "grade": "F",
            "components": {},
            "raw_metrics": {"max_drawdown": 8.0},
        }
        self.mock_timescale.query_candles.return_value = []
        self.mock_regime_classifier.classify.return_value = {
            "regime": "volatile",
            "confidence": 0.9,
            "indicators": {},
        }

    def test_evaluate_healthy_no_action(self):
        """Healthy strategy should result in no action."""
        self._set_healthy_state()

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                self.orchestrator.evaluate("hourly"))

            self.assertEqual(result["checkpoint_type"], "hourly")
            self.assertIsNone(result["decision"])
        finally:
            loop.close()

    def test_evaluate_low_health_pauses(self):
        """Low health score should trigger pause decision."""
        self._set_unhealthy_state()

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                self.orchestrator.evaluate("hourly"))

            self.assertIsNotNone(result["decision"])
            self.assertEqual(result["decision"]["action"], "pause_strategy")
            self.mock_strategy_manager.pause_strategy.assert_called_once()
        finally:
            loop.close()

    def test_evaluate_high_decay_pauses(self):
        """High backtest decay should trigger pause."""
        self._set_healthy_state()

        # Add backtest comparator with high decay
        mock_comparator = MagicMock()
        mock_comparator.compare.return_value = {
            "decay_pct": 60.0,
            "should_pause": True,
            "backtest_pnl": 100.0,
            "live_pnl": 40.0,
        }
        self.orchestrator.backtest_comparator = mock_comparator

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                self.orchestrator.evaluate("daily"))

            self.assertIsNotNone(result["decision"])
            self.assertEqual(result["decision"]["action"], "pause_strategy")
        finally:
            loop.close()

    def test_evaluate_high_drawdown_adjusts_risk(self):
        """High drawdown should trigger risk adjustment."""
        self.mock_health_scorer.compute_health_score.return_value = {
            "health_score": 45.0,
            "grade": "C",
            "components": {},
            "raw_metrics": {"max_drawdown": 6.5},
        }
        self.mock_timescale.query_candles.return_value = []
        self.mock_regime_classifier.classify.return_value = {
            "regime": "ranging",
            "confidence": 0.7,
            "indicators": {},
        }

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                self.orchestrator.evaluate("hourly"))

            self.assertIsNotNone(result["decision"])
            self.assertEqual(result["decision"]["action"], "adjust_risk")
        finally:
            loop.close()

    def test_evaluate_volatile_regime_adjusts_risk(self):
        """Volatile regime should recommend risk adjustment."""
        self.mock_health_scorer.compute_health_score.return_value = {
            "health_score": 65.0,
            "grade": "B",
            "components": {},
            "raw_metrics": {"max_drawdown": 2.0},
        }
        # Must return non-empty candles so regime classifier gets called
        self.mock_timescale.query_candles.return_value = [
            {"time": "2025-01-01T00:00:00Z", "open": 100, "high": 105,
             "low": 95, "close": 102, "volume": 1000},
        ]
        self.mock_regime_classifier.classify.return_value = {
            "regime": "volatile",
            "confidence": 0.9,
            "indicators": {},
        }

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                self.orchestrator.evaluate("daily"))

            self.assertIsNotNone(result["decision"])
            self.assertEqual(result["decision"]["action"], "adjust_risk")
        finally:
            loop.close()

    def test_decision_logged(self):
        """Every evaluation should log a decision."""
        self._set_healthy_state()

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                self.orchestrator.evaluate("hourly"))

            self.mock_timescale.insert_decision.assert_called()
        finally:
            loop.close()

    def test_evaluate_handles_exceptions(self):
        """Evaluation should handle exceptions gracefully."""
        self.mock_health_scorer.compute_health_score.side_effect = Exception("Error")
        self.mock_timescale.query_candles.side_effect = Exception("Error")

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                self.orchestrator.evaluate("hourly"))

            # Should not raise, should return error flag
            self.assertIn("checkpoint_type", result)
        finally:
            loop.close()

    def test_weekly_generates_report(self):
        """Weekly evaluation should generate a report."""
        self._set_healthy_state()

        mock_report_gen = MagicMock()
        mock_report_gen.generate_weekly_report.return_value = "Weekly report text"
        self.orchestrator.report_generator = mock_report_gen

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                self.orchestrator.evaluate("weekly"))

            mock_report_gen.generate_weekly_report.assert_called_once()
        finally:
            loop.close()

    def test_result_structure(self):
        """Result should have all expected keys."""
        self._set_healthy_state()

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                self.orchestrator.evaluate("hourly"))

            self.assertIn("checkpoint_type", result)
            self.assertIn("decision", result)
            self.assertIn("metrics", result)
            self.assertIn("duration_seconds", result)
        finally:
            loop.close()


if __name__ == "__main__":
    unittest.main()
