"""Tests for multi-strategy integration (WS5).

Tests OODA multi-strategy decision rules, portfolio report generation,
multi-strategy runner initialization, and backward compatibility.
"""

import asyncio
import os
import sys
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from orchestration.ooda import OODAOrchestrator
from reporting.report_generator import ReportGenerator


def _run_async(coro):
    """Helper to run async code in tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestOODAMultiStrategy(unittest.TestCase):
    """Tests for multi-strategy OODA decision rules 5-7."""

    def setUp(self):
        self.mock_strategy_manager = MagicMock()
        self.mock_strategy_manager.pause_strategy = AsyncMock()
        self.mock_strategy_manager.resume_strategy = AsyncMock()

        self.mock_health_scorer = MagicMock()
        self.mock_regime_classifier = MagicMock()
        self.mock_timescale = MagicMock()
        self.mock_redis = MagicMock()

        self.config = {
            "symbol": "BTC/USDC",
            "strategy_name": "funding_rate_arb",
            "decay_pause_threshold": 50.0,
            "health_pause_threshold": 20,
            "correlation_limit": 0.8,
            "portfolio_dd_limit": 15.0,
        }

        # Correlation analyzer mock
        self.mock_correlation = MagicMock()
        self.mock_correlation.compute_correlation_matrix.return_value = {
            "matrix": [[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]],
            "strategy_names": ["funding_rate_arb", "momentum_trend", "mean_reversion"],
            "period_hours": 168,
        }

        # Portfolio tracker mock
        self.mock_portfolio = MagicMock()
        self.mock_portfolio.compute_portfolio_metrics.return_value = {
            "total_pnl": 0.5,
            "portfolio_sharpe": 1.2,
            "portfolio_sortino": 1.5,
            "portfolio_max_dd": 3.0,
            "capital_efficiency": 0.05,
            "strategy_contributions": {
                "funding_rate_arb": 0.2,
                "momentum_trend": 0.15,
                "mean_reversion": 0.15,
            },
        }

        self.orchestrator = OODAOrchestrator(
            strategy_manager=self.mock_strategy_manager,
            health_scorer=self.mock_health_scorer,
            regime_classifier=self.mock_regime_classifier,
            timescale=self.mock_timescale,
            redis_store=self.mock_redis,
            config=self.config,
            correlation_analyzer=self.mock_correlation,
            portfolio_tracker=self.mock_portfolio,
        )

    def _set_multi_healthy_state(self):
        """Configure mocks for multiple healthy strategies."""
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
        self.mock_strategy_manager.get_active_strategies.return_value = [
            "funding_rate_arb", "momentum_trend", "mean_reversion",
        ]

    # ----- Rule 5: Correlation too high -----

    def test_rule5_high_correlation_triggers_allocation(self):
        """High correlation between strategies should recommend allocation reduction."""
        self._set_multi_healthy_state()

        # Set high correlation between two strategies
        self.mock_correlation.compute_correlation_matrix.return_value = {
            "matrix": [[1.0, 0.9, 0.1], [0.9, 1.0, 0.2], [0.1, 0.2, 1.0]],
            "strategy_names": ["funding_rate_arb", "momentum_trend", "mean_reversion"],
            "period_hours": 168,
        }

        result = _run_async(self.orchestrator.evaluate("hourly"))

        self.assertIsNotNone(result["decision"])
        self.assertEqual(result["decision"]["action"], "adjust_allocation")
        self.assertIn("momentum_trend", result["decision"]["strategy_name"])
        self.assertIn("0.9", result["decision"]["reason"])

    def test_rule5_low_correlation_no_action(self):
        """Low correlation should not trigger allocation adjustment."""
        self._set_multi_healthy_state()

        # All correlations below threshold
        self.mock_correlation.compute_correlation_matrix.return_value = {
            "matrix": [[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]],
            "strategy_names": ["funding_rate_arb", "momentum_trend", "mean_reversion"],
            "period_hours": 168,
        }

        result = _run_async(self.orchestrator.evaluate("hourly"))

        self.assertIsNone(result["decision"])

    def test_rule5_exactly_at_threshold_no_action(self):
        """Correlation at exactly the threshold should not trigger."""
        self._set_multi_healthy_state()

        self.mock_correlation.compute_correlation_matrix.return_value = {
            "matrix": [[1.0, 0.8, 0.1], [0.8, 1.0, 0.2], [0.1, 0.2, 1.0]],
            "strategy_names": ["funding_rate_arb", "momentum_trend", "mean_reversion"],
            "period_hours": 168,
        }

        result = _run_async(self.orchestrator.evaluate("hourly"))

        # 0.8 == threshold, not > threshold
        self.assertIsNone(result["decision"])

    # ----- Rule 6: Portfolio drawdown -----

    def test_rule6_portfolio_dd_pauses_worst_strategy(self):
        """Portfolio DD > limit should pause the worst-performing strategy."""
        self._set_multi_healthy_state()

        self.mock_portfolio.compute_portfolio_metrics.return_value = {
            "total_pnl": -0.5,
            "portfolio_sharpe": -0.3,
            "portfolio_sortino": -0.4,
            "portfolio_max_dd": 18.0,
            "capital_efficiency": -0.02,
            "strategy_contributions": {},
        }

        # Different health scores per strategy
        def health_side_effect(window_hours=168, strategy_name=None):
            scores = {
                "funding_rate_arb": {"health_score": 60, "grade": "C"},
                "momentum_trend": {"health_score": 25, "grade": "D"},
                "mean_reversion": {"health_score": 50, "grade": "C"},
            }
            if strategy_name and strategy_name in scores:
                return scores[strategy_name]
            return {"health_score": 75, "grade": "B", "components": {},
                    "raw_metrics": {"max_drawdown": 2.0}}

        self.mock_health_scorer.compute_health_score.side_effect = health_side_effect

        result = _run_async(self.orchestrator.evaluate("hourly"))

        self.assertIsNotNone(result["decision"])
        self.assertEqual(result["decision"]["action"], "pause_strategy")
        self.assertIn("momentum_trend", result["decision"]["strategy_name"])

    def test_rule6_portfolio_dd_below_limit_no_action(self):
        """Portfolio DD below limit should not trigger pause."""
        self._set_multi_healthy_state()

        self.mock_portfolio.compute_portfolio_metrics.return_value = {
            "total_pnl": 0.5,
            "portfolio_sharpe": 1.2,
            "portfolio_max_dd": 10.0,
            "capital_efficiency": 0.05,
            "strategy_contributions": {},
        }

        result = _run_async(self.orchestrator.evaluate("hourly"))

        self.assertIsNone(result["decision"])

    # ----- Rule 7: Strategy-regime suitability -----

    def test_rule7_momentum_in_ranging_adjusts_risk(self):
        """Momentum strategy in ranging regime should trigger risk adjustment."""
        self.mock_health_scorer.compute_health_score.return_value = {
            "health_score": 75.0,
            "grade": "B",
            "components": {},
            "raw_metrics": {"max_drawdown": 2.0},
        }
        self.mock_timescale.query_candles.return_value = [
            {"time": "2025-01-01T00:00:00Z", "open": 100, "high": 105,
             "low": 95, "close": 102, "volume": 1000},
        ]
        self.mock_regime_classifier.classify.return_value = {
            "regime": "ranging",
            "confidence": 0.8,
            "indicators": {},
        }

        # Create mock strategy objects with metadata
        mock_funding = MagicMock()
        mock_funding.get_metadata.return_value = {"name": "funding_rate_arb", "category": "funding_rate_arb"}
        mock_momentum = MagicMock()
        mock_momentum.get_metadata.return_value = {"name": "momentum_trend", "category": "momentum"}
        mock_mean_rev = MagicMock()
        mock_mean_rev.get_metadata.return_value = {"name": "mean_reversion", "category": "mean_reversion"}

        self.mock_strategy_manager.get_active_strategies.return_value = [
            "funding_rate_arb", "momentum_trend", "mean_reversion",
        ]
        self.mock_strategy_manager._strategies = {
            "funding_rate_arb": {"strategy": mock_funding, "status": "active"},
            "momentum_trend": {"strategy": mock_momentum, "status": "active"},
            "mean_reversion": {"strategy": mock_mean_rev, "status": "active"},
        }

        result = _run_async(self.orchestrator.evaluate("hourly"))

        self.assertIsNotNone(result["decision"])
        self.assertEqual(result["decision"]["action"], "adjust_risk")
        self.assertIn("momentum_trend", result["decision"]["strategy_name"])
        self.assertIn("ranging", result["decision"]["reason"])

    def test_rule7_all_strategies_suited_no_action(self):
        """When all strategies suit the regime, no action needed."""
        self._set_multi_healthy_state()

        # ranging regime - mean_reversion and funding_rate_arb are suited
        self.mock_strategy_manager.get_active_strategies.return_value = [
            "funding_rate_arb", "mean_reversion",
        ]

        mock_funding = MagicMock()
        mock_funding.get_metadata.return_value = {"category": "funding_rate_arb"}
        mock_mean_rev = MagicMock()
        mock_mean_rev.get_metadata.return_value = {"category": "mean_reversion"}

        self.mock_strategy_manager._strategies = {
            "funding_rate_arb": {"strategy": mock_funding, "status": "active"},
            "mean_reversion": {"strategy": mock_mean_rev, "status": "active"},
        }

        result = _run_async(self.orchestrator.evaluate("hourly"))
        self.assertIsNone(result["decision"])

    # ----- Allocation action handling -----

    def test_adjust_allocation_action_executes(self):
        """adjust_allocation action should be handled in _act."""
        self._set_multi_healthy_state()

        # High correlation
        self.mock_correlation.compute_correlation_matrix.return_value = {
            "matrix": [[1.0, 0.95], [0.95, 1.0]],
            "strategy_names": ["funding_rate_arb", "momentum_trend"],
            "period_hours": 168,
        }

        result = _run_async(self.orchestrator.evaluate("hourly"))

        self.assertIsNotNone(result["decision"])
        self.assertEqual(result["decision"]["action"], "adjust_allocation")
        self.assertIsNotNone(result["outcome"])
        self.assertTrue(result["outcome"]["executed"])

    # ----- Observe gathers portfolio data -----

    def test_observe_gathers_correlation(self):
        """_observe should collect correlation data."""
        self._set_multi_healthy_state()

        metrics = self.orchestrator._observe("hourly")

        self.assertIn("correlation", metrics)
        self.mock_correlation.compute_correlation_matrix.assert_called()

    def test_observe_gathers_portfolio_metrics(self):
        """_observe should collect portfolio-level metrics."""
        self._set_multi_healthy_state()

        metrics = self.orchestrator._observe("hourly")

        self.assertIn("portfolio", metrics)
        self.mock_portfolio.compute_portfolio_metrics.assert_called()


class TestPortfolioReport(unittest.TestCase):
    """Tests for portfolio report generation."""

    def setUp(self):
        self.mock_perf_tracker = MagicMock()
        self.mock_health_scorer = MagicMock()
        self.mock_timescale = MagicMock()

        self.mock_correlation = MagicMock()
        self.mock_correlation.compute_correlation_matrix.return_value = {
            "matrix": [[1.0, 0.3], [0.3, 1.0]],
            "strategy_names": ["funding_rate_arb", "momentum_trend"],
        }

        self.mock_portfolio = MagicMock()
        self.mock_portfolio.compute_portfolio_metrics.return_value = {
            "total_pnl": 1.5,
            "portfolio_sharpe": 1.8,
            "portfolio_max_dd": 4.5,
            "capital_efficiency": 0.03,
            "strategy_contributions": {
                "funding_rate_arb": 0.8,
                "momentum_trend": 0.7,
            },
        }

        self.report_gen = ReportGenerator(
            self.mock_perf_tracker,
            self.mock_health_scorer,
            self.mock_timescale,
            "funding_rate_arb",
            correlation_analyzer=self.mock_correlation,
            portfolio_tracker=self.mock_portfolio,
        )

    def test_portfolio_report_generated(self):
        """Portfolio report should be generated successfully."""
        health_map = {
            "funding_rate_arb": {"health_score": 80, "grade": "A"},
            "momentum_trend": {"health_score": 65, "grade": "B"},
        }

        report = self.report_gen.generate_portfolio_report(
            strategy_health_map=health_map)

        self.assertIn("PORTFOLIO REPORT", report)
        self.assertIn("1.5", report)  # total PnL
        self.assertIn("1.8", report)  # Sharpe
        self.assertIn("funding_rate_arb", report)
        self.assertIn("momentum_trend", report)

    def test_portfolio_report_with_wallet_highlights(self):
        """Portfolio report should include wallet intelligence highlights."""
        health_map = {
            "funding_rate_arb": {"health_score": 80, "grade": "A"},
        }

        wallets = [
            {"address": "0xABCDEF1234567890ABCDEF", "score": 85.5},
            {"address": "0x1234567890ABCDEF123456", "score": 72.0},
        ]

        report = self.report_gen.generate_portfolio_report(
            strategy_health_map=health_map,
            wallet_highlights=wallets)

        self.assertIn("WALLET INTELLIGENCE", report)
        self.assertIn("85.5", report)

    def test_portfolio_report_ranking_sorted_by_score(self):
        """Strategy ranking should be sorted by health score descending."""
        health_map = {
            "funding_rate_arb": {"health_score": 50, "grade": "C"},
            "momentum_trend": {"health_score": 90, "grade": "A"},
        }

        report = self.report_gen.generate_portfolio_report(
            strategy_health_map=health_map)

        # momentum_trend (90) should appear before funding_rate_arb (50)
        mt_pos = report.index("momentum_trend")
        fra_pos = report.index("funding_rate_arb")
        # In the ranking section, momentum_trend should come first
        ranking_section = report.split("STRATEGY RANKING")[1].split("CORRELATION")[0]
        self.assertIn("1. momentum_trend", ranking_section)
        self.assertIn("2. funding_rate_arb", ranking_section)

    def test_portfolio_report_no_data(self):
        """Portfolio report should handle missing data gracefully."""
        report_gen = ReportGenerator(
            self.mock_perf_tracker,
            self.mock_health_scorer,
            self.mock_timescale,
            "funding_rate_arb",
        )

        report = report_gen.generate_portfolio_report()

        self.assertIn("PORTFOLIO REPORT", report)
        self.assertIn("No strategy health data", report)


class TestMultiStrategyConfig(unittest.TestCase):
    """Tests for multi-strategy config and runner initialization patterns."""

    def test_ooda_constructor_accepts_new_params(self):
        """OODAOrchestrator should accept correlation_analyzer and portfolio_tracker."""
        mock_corr = MagicMock()
        mock_portfolio = MagicMock()

        orch = OODAOrchestrator(
            strategy_manager=MagicMock(),
            health_scorer=MagicMock(),
            regime_classifier=MagicMock(),
            timescale=MagicMock(),
            redis_store=MagicMock(),
            correlation_analyzer=mock_corr,
            portfolio_tracker=mock_portfolio,
        )

        self.assertIs(orch.correlation_analyzer, mock_corr)
        self.assertIs(orch.portfolio_tracker, mock_portfolio)

    def test_ooda_backward_compat_without_new_params(self):
        """OODAOrchestrator should work without new Phase 3 params."""
        orch = OODAOrchestrator(
            strategy_manager=MagicMock(),
            health_scorer=MagicMock(),
            regime_classifier=MagicMock(),
            timescale=MagicMock(),
            redis_store=MagicMock(),
        )

        self.assertIsNone(orch.correlation_analyzer)
        self.assertIsNone(orch.portfolio_tracker)
        self.assertEqual(orch.correlation_limit, 0.8)
        self.assertEqual(orch.portfolio_dd_limit, 15.0)

    def test_report_generator_accepts_new_params(self):
        """ReportGenerator should accept correlation_analyzer and portfolio_tracker."""
        mock_corr = MagicMock()
        mock_portfolio = MagicMock()

        rg = ReportGenerator(
            MagicMock(), MagicMock(), MagicMock(), "test",
            correlation_analyzer=mock_corr,
            portfolio_tracker=mock_portfolio,
        )

        self.assertIs(rg.correlation_analyzer, mock_corr)
        self.assertIs(rg.portfolio_tracker, mock_portfolio)

    def test_report_generator_backward_compat(self):
        """ReportGenerator should work without new Phase 3 params."""
        rg = ReportGenerator(
            MagicMock(), MagicMock(), MagicMock(), "test")

        self.assertIsNone(rg.correlation_analyzer)
        self.assertIsNone(rg.portfolio_tracker)

    def test_ooda_single_strategy_no_multi_rules_fire(self):
        """With single strategy, multi-strategy rules should not trigger."""
        mock_sm = MagicMock()
        mock_sm.get_active_strategies.return_value = ["funding_rate_arb"]
        mock_sm.pause_strategy = AsyncMock()

        mock_hs = MagicMock()
        mock_hs.compute_health_score.return_value = {
            "health_score": 75.0,
            "grade": "B",
            "components": {},
            "raw_metrics": {"max_drawdown": 2.0},
        }

        mock_ts = MagicMock()
        mock_ts.query_candles.return_value = []

        mock_rc = MagicMock()
        mock_rc.classify.return_value = {
            "regime": "ranging",
            "confidence": 0.7,
        }

        orch = OODAOrchestrator(
            strategy_manager=mock_sm,
            health_scorer=mock_hs,
            regime_classifier=mock_rc,
            timescale=mock_ts,
            redis_store=MagicMock(),
            config={
                "symbol": "BTC/USDC",
                "strategy_name": "funding_rate_arb",
                "correlation_limit": 0.8,
                "portfolio_dd_limit": 15.0,
            },
        )

        result = _run_async(orch.evaluate("hourly"))
        self.assertIsNone(result["decision"])


if __name__ == "__main__":
    unittest.main()
