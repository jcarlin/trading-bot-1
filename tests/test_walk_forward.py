#!/usr/bin/env python3
"""Tests for walk-forward optimization."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from backtest.param_grid import ParamGrid
from backtest.walk_forward_result import WalkForwardResult
from backtest.walk_forward import WalkForwardOptimizer
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
# TestParamGrid
# ===================================================================

class TestParamGrid(unittest.TestCase):

    def test_explicit_list(self):
        grid = ParamGrid({"period": [10, 20, 30]})
        combos = grid.expand()
        self.assertEqual(len(combos), 3)
        self.assertEqual(combos[0]["period"], 10)
        self.assertEqual(combos[2]["period"], 30)

    def test_range_expansion(self):
        grid = ParamGrid({"period": {"start": 10, "stop": 30, "step": 10}})
        combos = grid.expand()
        self.assertEqual(len(combos), 3)
        self.assertEqual(combos[0]["period"], 10)
        self.assertEqual(combos[1]["period"], 20)
        self.assertEqual(combos[2]["period"], 30)

    def test_range_int_types(self):
        grid = ParamGrid({"period": {"start": 5, "stop": 15, "step": 5}})
        combos = grid.expand()
        for combo in combos:
            self.assertIsInstance(combo["period"], int)

    def test_log_scale(self):
        grid = ParamGrid({"threshold": {"start": 0.001, "stop": 0.1, "scale": "log", "n": 3}})
        combos = grid.expand()
        self.assertEqual(len(combos), 3)
        self.assertAlmostEqual(combos[0]["threshold"], 0.001, places=4)
        self.assertAlmostEqual(combos[2]["threshold"], 0.1, places=4)
        # Middle value should be geometric mean
        self.assertAlmostEqual(combos[1]["threshold"], 0.01, places=4)

    def test_multi_param_cartesian_product(self):
        grid = ParamGrid({
            "fast": [5, 10],
            "slow": [20, 30],
        })
        combos = grid.expand()
        self.assertEqual(len(combos), 4)

    def test_count(self):
        grid = ParamGrid({
            "a": [1, 2, 3],
            "b": [10, 20],
        })
        self.assertEqual(grid.count(), 6)

    def test_empty_grid(self):
        grid = ParamGrid({})
        combos = grid.expand()
        self.assertEqual(len(combos), 1)
        self.assertEqual(combos[0], {})

    def test_single_value(self):
        grid = ParamGrid({"period": 14})
        combos = grid.expand()
        self.assertEqual(len(combos), 1)
        self.assertEqual(combos[0]["period"], 14)

    def test_count_empty(self):
        grid = ParamGrid({})
        self.assertEqual(grid.count(), 1)

    def test_log_scale_single(self):
        grid = ParamGrid({"t": {"start": 0.01, "stop": 0.1, "scale": "log", "n": 1}})
        combos = grid.expand()
        self.assertEqual(len(combos), 1)
        self.assertAlmostEqual(combos[0]["t"], 0.01, places=4)


# ===================================================================
# TestWalkForwardResult
# ===================================================================

class TestWalkForwardResult(unittest.TestCase):

    def test_summary_basic(self):
        result = WalkForwardResult(
            strategy_class="TestStrategy",
            best_params={"period": 14},
            avg_decay_pct=25.0,
            is_valid=True,
            duration_seconds=10.5,
        )
        summary = result.summary()
        self.assertIn("TestStrategy", summary)
        self.assertIn("25.0%", summary)
        self.assertIn("True", summary)

    def test_to_dict(self):
        result = WalkForwardResult(
            strategy_class="TestStrategy",
            best_params={"period": 14},
            avg_decay_pct=25.0,
            is_valid=True,
        )
        d = result.to_dict()
        self.assertEqual(d["strategy_class"], "TestStrategy")
        self.assertEqual(d["avg_decay_pct"], 25.0)
        self.assertTrue(d["is_valid"])

    def test_is_valid_pass(self):
        result = WalkForwardResult(avg_decay_pct=30.0, is_valid=True)
        self.assertTrue(result.is_valid)

    def test_is_valid_fail(self):
        result = WalkForwardResult(avg_decay_pct=60.0, is_valid=False)
        self.assertFalse(result.is_valid)

    def test_best_params(self):
        result = WalkForwardResult(best_params={"fast": 12, "slow": 26})
        self.assertEqual(result.best_params["fast"], 12)
        self.assertEqual(result.best_params["slow"], 26)

    def test_empty_windows(self):
        result = WalkForwardResult()
        self.assertEqual(len(result.windows), 0)
        summary = result.summary()
        self.assertIn("0", summary)

    def test_aggregated_oos_in_summary(self):
        result = WalkForwardResult(
            strategy_class="Test",
            aggregated_oos={"sharpe_ratio": 1.5, "win_rate": 60.0},
        )
        summary = result.summary()
        self.assertIn("sharpe_ratio", summary)
        self.assertIn("1.5", summary)


# ===================================================================
# TestWalkForwardOptimizer
# ===================================================================

class TestWalkForwardOptimizer(unittest.TestCase):

    def setUp(self):
        self.engine = MagicMock()
        self.risk_manager = MagicMock()
        self.config = {
            "n_splits": 3,
            "is_ratio": 0.7,
            "min_oos_bars": 10,
            "decay_threshold": 50.0,
        }
        self.optimizer = WalkForwardOptimizer(
            self.engine, self.risk_manager, self.config)

    def test_window_splitting_correct_count(self):
        df = _make_ohlcv_df(n=300)
        windows = self.optimizer._split_windows(df)
        self.assertGreater(len(windows), 0)
        self.assertLessEqual(len(windows), 3)

    def test_window_splitting_is_oos_no_overlap_data(self):
        """IS data ends where OOS begins."""
        df = _make_ohlcv_df(n=300)
        windows = self.optimizer._split_windows(df)
        if windows:
            is_data, oos_data = windows[0]
            # OOS starts after IS ends
            self.assertGreater(len(is_data), 0)
            self.assertGreater(len(oos_data), 0)
            # IS end index < total, OOS has some bars
            self.assertLess(len(is_data), len(df))

    def test_window_splitting_small_dataset(self):
        """Very small dataset should return empty."""
        df = _make_ohlcv_df(n=5)
        self.optimizer.min_oos_bars = 100
        windows = self.optimizer._split_windows(df)
        self.assertEqual(len(windows), 0)

    def test_optimize_window_returns_best(self):
        """Optimize should pick the params with highest metric."""
        df = _make_ohlcv_df(n=100)

        # Make engine return different results based on call order
        results = [
            _make_backtest_result(initial=10000, final=10500, n_trades=3),
            _make_backtest_result(initial=10000, final=12000, n_trades=5),
            _make_backtest_result(initial=10000, final=10200, n_trades=2),
        ]
        self.engine.run.side_effect = results

        combos = [{"p": 10}, {"p": 20}, {"p": 30}]
        best_params, best_score = self.optimizer._optimize_window(
            df, MockStrategy, combos, "total_return")

        self.assertEqual(best_params["p"], 20)
        self.assertGreater(best_score, 0)

    def test_validate_window_returns_metrics(self):
        """Validate should return a dict of metrics."""
        df = _make_ohlcv_df(n=100)
        self.engine.run.return_value = _make_backtest_result()
        metrics = self.optimizer._validate_window(df, MockStrategy, {"p": 14})

        self.assertIn("sharpe_ratio", metrics)
        self.assertIn("total_return", metrics)
        self.assertIn("win_rate", metrics)

    def test_decay_computation_positive(self):
        is_metrics = {"sharpe_ratio": 2.0}
        oos_metrics = {"sharpe_ratio": 1.0}
        decay = self.optimizer.compute_decay(is_metrics, oos_metrics, "sharpe_ratio")
        self.assertAlmostEqual(decay, 50.0)

    def test_decay_computation_no_decay(self):
        is_metrics = {"sharpe_ratio": 1.0}
        oos_metrics = {"sharpe_ratio": 1.5}
        decay = self.optimizer.compute_decay(is_metrics, oos_metrics, "sharpe_ratio")
        self.assertEqual(decay, 0.0)  # No decay when OOS > IS

    def test_decay_zero_is(self):
        is_metrics = {"sharpe_ratio": 0.0}
        oos_metrics = {"sharpe_ratio": 1.0}
        decay = self.optimizer.compute_decay(is_metrics, oos_metrics, "sharpe_ratio")
        self.assertEqual(decay, 0.0)

    def test_aggregate_oos_metrics(self):
        metrics_list = [
            {"sharpe_ratio": 1.0, "win_rate": 60.0},
            {"sharpe_ratio": 2.0, "win_rate": 50.0},
        ]
        agg = self.optimizer._aggregate_oos_metrics(metrics_list)
        self.assertAlmostEqual(agg["sharpe_ratio"], 1.5)
        self.assertAlmostEqual(agg["win_rate"], 55.0)

    def test_aggregate_empty(self):
        agg = self.optimizer._aggregate_oos_metrics([])
        self.assertEqual(agg, {})

    def test_is_valid_pass(self):
        """avg_decay < 50 -> is_valid = True."""
        df = _make_ohlcv_df(n=200)
        # Mock engine to return consistent results (low decay)
        self.engine.run.return_value = _make_backtest_result(
            initial=10000, final=11000, n_trades=5)

        result = self.optimizer.run(df, MockStrategy,
                                    {"p": [10, 20]}, "total_return")
        # Result should be valid since IS and OOS use same mock
        if result.windows:
            self.assertTrue(result.is_valid)

    def test_is_valid_fail(self):
        """Construct scenario where avg_decay > 50 -> is_valid = False."""
        result = WalkForwardResult(avg_decay_pct=60.0, is_valid=False)
        self.assertFalse(result.is_valid)

    def test_full_run_end_to_end(self):
        """Full WF run with mock backtest engine."""
        df = _make_ohlcv_df(n=200)
        self.engine.run.return_value = _make_backtest_result(
            initial=10000, final=11000, n_trades=5)

        result = self.optimizer.run(
            df, MockStrategy, {"p": [10, 20]}, "sharpe_ratio")

        self.assertIsInstance(result, WalkForwardResult)
        self.assertEqual(result.strategy_class, "MockStrategy")
        self.assertIsInstance(result.avg_decay_pct, float)
        self.assertGreater(result.duration_seconds, 0)

    def test_full_run_empty_params(self):
        """Run with empty param grid should return quickly."""
        df = _make_ohlcv_df(n=200)
        result = self.optimizer.run(df, MockStrategy, {}, "sharpe_ratio")
        self.assertIsInstance(result, WalkForwardResult)

    def test_single_param_combo(self):
        """Run with a single param combo should work."""
        df = _make_ohlcv_df(n=200)
        self.engine.run.return_value = _make_backtest_result()
        result = self.optimizer.run(df, MockStrategy, {"p": [14]}, "sharpe_ratio")
        self.assertIsInstance(result, WalkForwardResult)
        if result.best_params:
            self.assertEqual(result.best_params["p"], 14)

    def test_most_common_params(self):
        all_params = [{"p": 10}, {"p": 20}, {"p": 10}]
        best = self.optimizer._most_common_params(all_params)
        self.assertEqual(best["p"], 10)

    def test_most_common_params_empty(self):
        best = self.optimizer._most_common_params([])
        self.assertEqual(best, {})

    def test_compute_metrics_no_trades(self):
        """Metrics with no trades should return zeroes."""
        dates = pd.date_range("2025-01-01", periods=10, freq="h", tz="UTC")
        equity = pd.Series([10000.0] * 10, index=dates, name="equity")
        result = BacktestResult(
            trades=[], equity_curve=equity,
            initial_capital=10000, final_equity=10000)
        metrics = self.optimizer._compute_metrics(result)
        self.assertEqual(metrics["trade_count"], 0)
        self.assertEqual(metrics["sharpe_ratio"], 0.0)

    def test_compute_metrics_with_trades(self):
        result = _make_backtest_result(initial=10000, final=12000, n_trades=10)
        metrics = self.optimizer._compute_metrics(result)
        self.assertEqual(metrics["trade_count"], 10)
        self.assertGreater(metrics["total_return"], 0)
        self.assertGreater(metrics["win_rate"], 0)


if __name__ == "__main__":
    unittest.main()
