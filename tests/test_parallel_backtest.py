#!/usr/bin/env python3
"""Tests for parallel backtest runner and MonteCarloResult."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from backtest.monte_carlo_result import MonteCarloResult
from backtest.parallel_runner import ParallelBacktestRunner, _compute_metrics_standalone
from backtest.engine import BacktestResult
from core.models import Signal, Trade
from core.types import SignalType, Side


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv_df(n=500, start_price=50000.0, volatility=0.01):
    """Generate synthetic OHLCV data."""
    dates = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    np.random.seed(42)
    returns = np.random.normal(0, volatility, n)
    prices = start_price * np.cumprod(1 + returns)
    data = {
        "open": prices * (1 - 0.001),
        "high": prices * (1 + 0.005),
        "low": prices * (1 - 0.005),
        "close": prices,
        "volume": np.random.uniform(100, 1000, n),
    }
    df = pd.DataFrame(data, index=dates)
    df.attrs["symbol"] = "BTC/USDC"
    return df


def _make_backtest_result(initial=10000, final=11000, n_trades=5):
    """Make a mock BacktestResult."""
    dates = pd.date_range("2025-01-01", periods=100, freq="h", tz="UTC")
    equity = np.linspace(initial, final, 100)
    equity_curve = pd.Series(equity, index=dates, name="equity")
    trades = []
    for i in range(n_trades):
        pnl = (final - initial) / n_trades
        trades.append(Trade(
            symbol="BTC/USDC", side=Side.BUY,
            entry_price=50000.0, exit_price=50000.0 + pnl,
            quantity=0.01,
            entry_time=dates[i * 10], exit_time=dates[i * 10 + 5],
            pnl=pnl, commission=1.0, exit_reason="signal",
        ))
    return BacktestResult(
        trades=trades, equity_curve=equity_curve,
        initial_capital=initial, final_equity=final,
    )


class MockStrategy:
    """A simple mock strategy for testing."""
    def __init__(self, params):
        self.params = params
        self.indicators = {}

    def setup(self, df):
        pass

    def generate_signal(self, index, df):
        return Signal(
            signal_type=SignalType.HOLD,
            price=df["close"].iloc[index],
            timestamp=df.index[index],
        )


# ===================================================================
# TestMonteCarloResult
# ===================================================================

class TestMonteCarloResult(unittest.TestCase):

    def test_default_fields(self):
        result = MonteCarloResult()
        self.assertEqual(result.n_simulations, 0)
        self.assertEqual(result.metric_distributions, {})
        self.assertEqual(result.percentiles, {})
        self.assertEqual(result.confidence_intervals, {})
        self.assertEqual(result.original_metrics, {})
        self.assertEqual(result.p_value_vs_random, 1.0)
        self.assertEqual(result.duration_seconds, 0.0)

    def test_fields_populated(self):
        result = MonteCarloResult(
            n_simulations=100,
            metric_distributions={"sharpe_ratio": [1.0, 1.5, 2.0]},
            percentiles={"sharpe_ratio": {5: 0.5, 50: 1.5, 95: 2.5}},
            confidence_intervals={"sharpe_ratio": {"lower": 0.3, "upper": 2.7, "level": 0.95}},
            original_metrics={"sharpe_ratio": 1.8},
            p_value_vs_random=0.05,
            duration_seconds=15.0,
        )
        self.assertEqual(result.n_simulations, 100)
        self.assertEqual(len(result.metric_distributions["sharpe_ratio"]), 3)
        self.assertAlmostEqual(result.p_value_vs_random, 0.05)

    def test_summary_contains_key_info(self):
        result = MonteCarloResult(
            n_simulations=100,
            percentiles={"sharpe_ratio": {5: 0.5, 50: 1.5, 95: 2.5}},
            original_metrics={"sharpe_ratio": 1.8},
            p_value_vs_random=0.03,
            duration_seconds=10.0,
        )
        summary = result.summary()
        self.assertIn("100", summary)
        self.assertIn("0.03", summary)
        self.assertIn("sharpe_ratio", summary)

    def test_summary_no_percentiles(self):
        result = MonteCarloResult(n_simulations=50)
        summary = result.summary()
        self.assertIn("50", summary)
        self.assertNotIn("Percentiles", summary)

    def test_to_dict(self):
        result = MonteCarloResult(
            n_simulations=100,
            p_value_vs_random=0.05,
            duration_seconds=10.0,
        )
        d = result.to_dict()
        self.assertEqual(d["n_simulations"], 100)
        self.assertAlmostEqual(d["p_value_vs_random"], 0.05)
        self.assertAlmostEqual(d["duration_seconds"], 10.0)

    def test_to_dict_complete(self):
        result = MonteCarloResult(
            n_simulations=50,
            metric_distributions={"total_return": [0.1, 0.2]},
            percentiles={"total_return": {50: 0.15}},
            confidence_intervals={"total_return": {"lower": 0.08, "upper": 0.22, "level": 0.95}},
            original_metrics={"total_return": 0.18},
        )
        d = result.to_dict()
        self.assertIn("metric_distributions", d)
        self.assertIn("percentiles", d)
        self.assertIn("confidence_intervals", d)
        self.assertIn("original_metrics", d)


# ===================================================================
# TestComputeMetricsStandalone
# ===================================================================

class TestComputeMetricsStandalone(unittest.TestCase):

    def test_with_trades(self):
        result = _make_backtest_result(initial=10000, final=12000, n_trades=5)
        metrics = _compute_metrics_standalone(result)
        self.assertEqual(metrics["trade_count"], 5)
        self.assertAlmostEqual(metrics["total_return"], 0.2)
        self.assertGreater(metrics["win_rate"], 0)

    def test_no_trades(self):
        dates = pd.date_range("2025-01-01", periods=10, freq="h", tz="UTC")
        equity = pd.Series([10000.0] * 10, index=dates, name="equity")
        result = BacktestResult(
            trades=[], equity_curve=equity,
            initial_capital=10000, final_equity=10000,
        )
        metrics = _compute_metrics_standalone(result)
        self.assertEqual(metrics["trade_count"], 0)
        self.assertEqual(metrics["sharpe_ratio"], 0.0)

    def test_metrics_keys(self):
        result = _make_backtest_result()
        metrics = _compute_metrics_standalone(result)
        expected_keys = {"sharpe_ratio", "total_return", "profit_factor",
                         "win_rate", "trade_count", "max_drawdown"}
        self.assertEqual(set(metrics.keys()), expected_keys)


# ===================================================================
# TestParallelBacktestRunner
# ===================================================================

class TestParallelBacktestRunner(unittest.TestCase):

    def test_default_config(self):
        runner = ParallelBacktestRunner()
        self.assertGreater(runner.max_workers, 0)
        self.assertEqual(runner.chunk_size, 10)
        self.assertEqual(runner.timeout_per_backtest_s, 60)

    def test_custom_config(self):
        runner = ParallelBacktestRunner({
            "max_workers": 2,
            "chunk_size": 5,
            "timeout_per_backtest_s": 30,
        })
        self.assertEqual(runner.max_workers, 2)
        self.assertEqual(runner.chunk_size, 5)
        self.assertEqual(runner.timeout_per_backtest_s, 30)

    def test_generate_shuffled_returns_mode(self):
        runner = ParallelBacktestRunner()
        df = _make_ohlcv_df(n=100)
        shuffled = runner._generate_shuffled_data(df, 3, "returns")
        self.assertEqual(len(shuffled), 3)
        for s in shuffled:
            self.assertEqual(len(s), len(df))
            self.assertListEqual(list(s.columns), list(df.columns))
            # First close should be same (reconstruction starts from same point)
            self.assertAlmostEqual(s["close"].iloc[0], df["close"].iloc[0], places=2)

    def test_generate_shuffled_block_mode(self):
        runner = ParallelBacktestRunner()
        df = _make_ohlcv_df(n=100)
        shuffled = runner._generate_shuffled_data(df, 3, "block")
        self.assertEqual(len(shuffled), 3)
        for s in shuffled:
            self.assertEqual(len(s), len(df))

    def test_generate_shuffled_different_results(self):
        """Shuffled data should differ from original."""
        runner = ParallelBacktestRunner()
        np.random.seed(None)  # Reset seed for randomness
        df = _make_ohlcv_df(n=200)
        shuffled = runner._generate_shuffled_data(df, 5, "returns")
        # At least one shuffled should differ from original
        diffs = 0
        for s in shuffled:
            if not np.allclose(s["close"].values, df["close"].values, atol=1.0):
                diffs += 1
        self.assertGreater(diffs, 0, "Expected at least one shuffled df to differ from original")

    def test_generate_shuffled_preserves_attrs(self):
        runner = ParallelBacktestRunner()
        df = _make_ohlcv_df(n=100)
        df.attrs["symbol"] = "ETH/USDC"
        shuffled = runner._generate_shuffled_data(df, 1, "returns")
        self.assertEqual(shuffled[0].attrs.get("symbol"), "ETH/USDC")

    def test_monte_carlo_basic(self):
        """Monte Carlo with mock engine should produce distributions."""
        runner = ParallelBacktestRunner({"max_workers": 1})
        df = _make_ohlcv_df(n=200)

        mock_result = _make_backtest_result()

        with patch("backtest.engine.BacktestEngine") as MockEngine, \
             patch("risk.manager.RiskManager") as MockRM:
            mock_engine_inst = MagicMock()
            mock_engine_inst.run.return_value = mock_result
            MockEngine.return_value = mock_engine_inst
            MockRM.return_value = MagicMock()

            mc_result = runner.run_monte_carlo(
                df, MockStrategy, {"p": 14},
                risk_config={"position_sizing": "percent_equity"},
                n_simulations=5,
            )

            self.assertIsInstance(mc_result, MonteCarloResult)
            self.assertEqual(mc_result.n_simulations, 5)
            self.assertIn("sharpe_ratio", mc_result.metric_distributions)
            self.assertIn("sharpe_ratio", mc_result.percentiles)
            self.assertIn("sharpe_ratio", mc_result.confidence_intervals)

    def test_monte_carlo_percentiles_computed(self):
        """Percentiles should contain standard quantiles."""
        runner = ParallelBacktestRunner({"max_workers": 1})
        df = _make_ohlcv_df(n=200)

        mock_result = _make_backtest_result()

        with patch("backtest.engine.BacktestEngine") as MockEngine, \
             patch("risk.manager.RiskManager") as MockRM:
            mock_engine_inst = MagicMock()
            mock_engine_inst.run.return_value = mock_result
            MockEngine.return_value = mock_engine_inst
            MockRM.return_value = MagicMock()

            mc_result = runner.run_monte_carlo(
                df, MockStrategy, {"p": 14},
                risk_config={},
                n_simulations=10,
            )

            for metric in ["sharpe_ratio", "total_return"]:
                self.assertIn(5, mc_result.percentiles[metric])
                self.assertIn(25, mc_result.percentiles[metric])
                self.assertIn(50, mc_result.percentiles[metric])
                self.assertIn(75, mc_result.percentiles[metric])
                self.assertIn(95, mc_result.percentiles[metric])

    def test_monte_carlo_p_value_computed(self):
        """p_value should be between 0 and 1."""
        runner = ParallelBacktestRunner({"max_workers": 1})
        df = _make_ohlcv_df(n=200)

        mock_result = _make_backtest_result()

        with patch("backtest.engine.BacktestEngine") as MockEngine, \
             patch("risk.manager.RiskManager") as MockRM:
            mock_engine_inst = MagicMock()
            mock_engine_inst.run.return_value = mock_result
            MockEngine.return_value = mock_engine_inst
            MockRM.return_value = MagicMock()

            mc_result = runner.run_monte_carlo(
                df, MockStrategy, {"p": 14},
                risk_config={},
                n_simulations=10,
            )

            self.assertGreaterEqual(mc_result.p_value_vs_random, 0.0)
            self.assertLessEqual(mc_result.p_value_vs_random, 1.0)

    def test_monte_carlo_original_metrics_stored(self):
        """Original strategy metrics should be stored in result."""
        runner = ParallelBacktestRunner({"max_workers": 1})
        df = _make_ohlcv_df(n=200)

        mock_result = _make_backtest_result(initial=10000, final=12000, n_trades=5)

        with patch("backtest.engine.BacktestEngine") as MockEngine, \
             patch("risk.manager.RiskManager") as MockRM:
            mock_engine_inst = MagicMock()
            mock_engine_inst.run.return_value = mock_result
            MockEngine.return_value = mock_engine_inst
            MockRM.return_value = MagicMock()

            mc_result = runner.run_monte_carlo(
                df, MockStrategy, {"p": 14},
                risk_config={},
                n_simulations=3,
            )

            self.assertIn("total_return", mc_result.original_metrics)
            self.assertAlmostEqual(mc_result.original_metrics["total_return"], 0.2)

    def test_strategy_comparison_basic(self):
        """Strategy comparison should return results for all strategies."""
        runner = ParallelBacktestRunner({"max_workers": 1})
        df = _make_ohlcv_df(n=200)

        mock_result = _make_backtest_result()

        with patch("backtest.engine.BacktestEngine") as MockEngine, \
             patch("risk.manager.RiskManager") as MockRM:
            mock_engine_inst = MagicMock()
            mock_engine_inst.run.return_value = mock_result
            MockEngine.return_value = mock_engine_inst
            MockRM.return_value = MagicMock()

            strategies = [
                (MockStrategy, {"p": 10}),
                (MockStrategy, {"p": 20}),
            ]
            results = runner.run_strategy_comparison(df, strategies, risk_config={})

            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["strategy_name"], "MockStrategy")
            self.assertIn("metrics", results[0])

    def test_strategy_comparison_handles_errors(self):
        """Strategy comparison should handle individual strategy failures."""
        runner = ParallelBacktestRunner({"max_workers": 1})
        df = _make_ohlcv_df(n=200)

        with patch("backtest.engine.BacktestEngine") as MockEngine, \
             patch("risk.manager.RiskManager") as MockRM:
            mock_engine_inst = MagicMock()
            mock_engine_inst.run.side_effect = RuntimeError("strategy failed")
            MockEngine.return_value = mock_engine_inst
            MockRM.return_value = MagicMock()

            strategies = [(MockStrategy, {"p": 10})]
            results = runner.run_strategy_comparison(df, strategies, risk_config={})

            self.assertEqual(len(results), 1)
            self.assertIn("error", results[0])

    def test_monte_carlo_confidence_intervals(self):
        """Confidence intervals should have lower, upper, level keys."""
        runner = ParallelBacktestRunner({"max_workers": 1})
        df = _make_ohlcv_df(n=200)

        # Use a result with mixed wins/losses to avoid inf profit_factor
        mock_result = _make_backtest_result(initial=10000, final=10500, n_trades=5)

        with patch("backtest.engine.BacktestEngine") as MockEngine, \
             patch("risk.manager.RiskManager") as MockRM:
            mock_engine_inst = MagicMock()
            mock_engine_inst.run.return_value = mock_result
            MockEngine.return_value = mock_engine_inst
            MockRM.return_value = MagicMock()

            mc_result = runner.run_monte_carlo(
                df, MockStrategy, {"p": 14},
                risk_config={},
                n_simulations=10,
            )

            # Check metrics that won't produce inf/nan values
            for metric in ["sharpe_ratio", "total_return", "win_rate", "max_drawdown"]:
                if metric in mc_result.confidence_intervals:
                    ci = mc_result.confidence_intervals[metric]
                    self.assertIn("lower", ci)
                    self.assertIn("upper", ci)
                    self.assertIn("level", ci)
                    self.assertEqual(ci["level"], 0.95)
                    self.assertLessEqual(ci["lower"], ci["upper"])


if __name__ == "__main__":
    unittest.main()
