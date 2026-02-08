#!/usr/bin/env python3
"""Tests for Optuna optimizer and OptunaResult."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd

from backtest.optuna_result import OptunaResult
from backtest.optuna_optimizer import OptunaOptimizer
from backtest.engine import BacktestResult
from backtest.walk_forward import WalkForwardOptimizer
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
# TestOptunaResult
# ===================================================================

class TestOptunaResult(unittest.TestCase):

    def test_default_fields(self):
        result = OptunaResult()
        self.assertEqual(result.best_params, {})
        self.assertEqual(result.best_score, 0.0)
        self.assertEqual(result.n_trials, 0)
        self.assertEqual(result.n_completed, 0)
        self.assertEqual(result.n_pruned, 0)
        self.assertEqual(result.param_importance, {})
        self.assertEqual(result.trial_history, [])
        self.assertEqual(result.duration_seconds, 0.0)
        self.assertEqual(result.strategy_class, "")
        self.assertEqual(result.metric, "")

    def test_fields_populated(self):
        result = OptunaResult(
            best_params={"period": 14, "threshold": 0.05},
            best_score=1.85,
            n_trials=50,
            n_completed=45,
            n_pruned=5,
            param_importance={"period": 0.7, "threshold": 0.3},
            trial_history=[{"params": {"period": 14}, "score": 1.85, "state": "COMPLETE", "duration_s": 0.5}],
            duration_seconds=30.0,
            strategy_class="MomentumStrategy",
            metric="sharpe_ratio",
        )
        self.assertEqual(result.best_params["period"], 14)
        self.assertEqual(result.best_score, 1.85)
        self.assertEqual(result.n_trials, 50)
        self.assertEqual(result.n_completed, 45)
        self.assertEqual(result.n_pruned, 5)
        self.assertAlmostEqual(result.param_importance["period"], 0.7)

    def test_summary_contains_key_info(self):
        result = OptunaResult(
            best_params={"period": 14},
            best_score=1.5,
            n_trials=100,
            n_completed=90,
            n_pruned=10,
            strategy_class="TestStrat",
            metric="sharpe_ratio",
            param_importance={"period": 0.8},
        )
        summary = result.summary()
        self.assertIn("TestStrat", summary)
        self.assertIn("sharpe_ratio", summary)
        self.assertIn("1.5", summary)
        self.assertIn("90 completed", summary)
        self.assertIn("10 pruned", summary)
        self.assertIn("period", summary)

    def test_summary_no_importance(self):
        result = OptunaResult(strategy_class="Test", metric="sharpe_ratio")
        summary = result.summary()
        self.assertNotIn("Param importance", summary)

    def test_to_dict(self):
        result = OptunaResult(
            best_params={"p": 10},
            best_score=2.0,
            n_trials=50,
            strategy_class="Test",
            metric="sharpe_ratio",
        )
        d = result.to_dict()
        self.assertEqual(d["best_params"], {"p": 10})
        self.assertEqual(d["best_score"], 2.0)
        self.assertEqual(d["n_trials"], 50)
        self.assertEqual(d["strategy_class"], "Test")
        self.assertEqual(d["metric"], "sharpe_ratio")

    def test_to_dict_roundtrip(self):
        result = OptunaResult(
            best_params={"a": 1, "b": 0.5},
            best_score=1.23,
            n_trials=10,
            n_completed=8,
            n_pruned=2,
            duration_seconds=5.0,
        )
        d = result.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["n_completed"], 8)
        self.assertEqual(d["n_pruned"], 2)


# ===================================================================
# TestOptunaOptimizer
# ===================================================================

class TestOptunaOptimizer(unittest.TestCase):

    def setUp(self):
        self.engine = MagicMock()
        self.risk_manager = MagicMock()
        self.config = {
            "n_trials": 10,
            "n_jobs": 1,
            "sampler": "tpe",
            "pruner": "median",
            "metric": "sharpe_ratio",
            "direction": "maximize",
            "timeout_s": 30,
        }
        self.optimizer = OptunaOptimizer(self.engine, self.risk_manager, self.config)

    def test_is_available_with_optuna(self):
        """is_available returns True when optuna importable."""
        with patch.dict("sys.modules", {"optuna": MagicMock()}):
            self.assertTrue(self.optimizer.is_available())

    def test_is_available_without_optuna(self):
        """is_available returns False when optuna not installed."""
        with patch.dict("sys.modules", {"optuna": None}):
            import importlib
            # Force ImportError
            original = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__
            def mock_import(name, *args, **kwargs):
                if name == "optuna":
                    raise ImportError("No module named 'optuna'")
                return original(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                self.assertFalse(self.optimizer.is_available())

    def test_optimize_returns_none_without_optuna(self):
        """optimize() returns None when optuna not installed."""
        import sys
        # Temporarily remove optuna from sys.modules if present
        optuna_mod = sys.modules.pop("optuna", None)
        sys.modules["optuna"] = None  # Force ImportError on import

        try:
            # Re-import to trigger the lazy import path
            optimizer = OptunaOptimizer(self.engine, self.risk_manager, self.config)
            result = optimizer.optimize(
                _make_ohlcv_df(100), MockStrategy, {"p": [1, 2]})
            self.assertIsNone(result)
        finally:
            # Restore
            if optuna_mod is not None:
                sys.modules["optuna"] = optuna_mod
            else:
                sys.modules.pop("optuna", None)

    def test_suggest_params_categorical(self):
        """List values should use suggest_categorical."""
        trial = MagicMock()
        trial.suggest_categorical.return_value = 20
        param_space = {"period": [10, 20, 30]}

        params = self.optimizer._suggest_params(trial, param_space)
        trial.suggest_categorical.assert_called_once_with("period", [10, 20, 30])
        self.assertEqual(params["period"], 20)

    def test_suggest_params_int(self):
        """Int low/high should use suggest_int."""
        trial = MagicMock()
        trial.suggest_int.return_value = 15
        param_space = {"period": {"low": 5, "high": 30}}

        params = self.optimizer._suggest_params(trial, param_space)
        trial.suggest_int.assert_called_once_with("period", 5, 30, step=1)
        self.assertEqual(params["period"], 15)

    def test_suggest_params_int_with_step(self):
        """Int with step parameter."""
        trial = MagicMock()
        trial.suggest_int.return_value = 10
        param_space = {"period": {"low": 5, "high": 30, "step": 5}}

        params = self.optimizer._suggest_params(trial, param_space)
        trial.suggest_int.assert_called_once_with("period", 5, 30, step=5)

    def test_suggest_params_float(self):
        """Float low/high should use suggest_float."""
        trial = MagicMock()
        trial.suggest_float.return_value = 0.05
        param_space = {"threshold": {"low": 0.01, "high": 0.1}}

        params = self.optimizer._suggest_params(trial, param_space)
        trial.suggest_float.assert_called_once_with("threshold", 0.01, 0.1, log=False)
        self.assertAlmostEqual(params["threshold"], 0.05)

    def test_suggest_params_log_float(self):
        """Float with log=True should use suggest_float(log=True)."""
        trial = MagicMock()
        trial.suggest_float.return_value = 0.03
        param_space = {"lr": {"low": 0.001, "high": 0.1, "log": True}}

        params = self.optimizer._suggest_params(trial, param_space)
        trial.suggest_float.assert_called_once_with("lr", 0.001, 0.1, log=True)

    def test_suggest_params_fixed_value(self):
        """Single fixed value should be passed through."""
        trial = MagicMock()
        param_space = {"mode": "aggressive"}

        params = self.optimizer._suggest_params(trial, param_space)
        self.assertEqual(params["mode"], "aggressive")

    def test_suggest_params_mixed(self):
        """Mixed param space with different types."""
        trial = MagicMock()
        trial.suggest_categorical.return_value = "ema"
        trial.suggest_int.return_value = 14
        trial.suggest_float.return_value = 0.05

        param_space = {
            "method": ["sma", "ema"],
            "period": {"low": 5, "high": 30},
            "threshold": {"low": 0.01, "high": 0.1},
        }
        params = self.optimizer._suggest_params(trial, param_space)
        self.assertEqual(params["method"], "ema")
        self.assertEqual(params["period"], 14)
        self.assertAlmostEqual(params["threshold"], 0.05)

    def test_extract_metric_sharpe(self):
        result = _make_backtest_result(initial=10000, final=12000, n_trades=5)
        score = self.optimizer._extract_metric(result, "sharpe_ratio")
        self.assertIsInstance(score, float)

    def test_extract_metric_total_return(self):
        result = _make_backtest_result(initial=10000, final=12000, n_trades=5)
        score = self.optimizer._extract_metric(result, "total_return")
        self.assertAlmostEqual(score, 0.2)

    def test_extract_metric_calmar(self):
        result = _make_backtest_result(initial=10000, final=12000, n_trades=5)
        score = self.optimizer._extract_metric(result, "calmar_ratio")
        self.assertIsInstance(score, float)

    def test_extract_metric_no_trades(self):
        dates = pd.date_range("2025-01-01", periods=10, freq="h", tz="UTC")
        equity = pd.Series([10000.0] * 10, index=dates, name="equity")
        result = BacktestResult(
            trades=[], equity_curve=equity,
            initial_capital=10000, final_equity=10000,
        )
        score = self.optimizer._extract_metric(result, "sharpe_ratio")
        self.assertEqual(score, 0.0)

    def test_create_objective_returns_callable(self):
        df = _make_ohlcv_df(100)
        objective = self.optimizer._create_objective(
            df, MockStrategy, {"p": [10, 20]}, "sharpe_ratio")
        self.assertTrue(callable(objective))

    def test_objective_calls_engine(self):
        """Objective function should call backtest engine."""
        df = _make_ohlcv_df(100)
        self.engine.run.return_value = _make_backtest_result()

        objective = self.optimizer._create_objective(
            df, MockStrategy, {"p": [10, 20]}, "total_return")

        trial = MagicMock()
        trial.suggest_categorical.return_value = 10
        score = objective(trial)

        self.engine.run.assert_called_once()
        self.assertIsInstance(score, float)

    def test_objective_handles_error(self):
        """Objective should return -inf on error (maximize direction)."""
        df = _make_ohlcv_df(100)
        self.engine.run.side_effect = RuntimeError("boom")

        objective = self.optimizer._create_objective(
            df, MockStrategy, {"p": [10, 20]}, "sharpe_ratio")

        trial = MagicMock()
        trial.suggest_categorical.return_value = 10
        score = objective(trial)

        self.assertEqual(score, float("-inf"))

    def test_config_defaults(self):
        optimizer = OptunaOptimizer(self.engine, self.risk_manager, {})
        self.assertEqual(optimizer.n_trials, 100)
        self.assertEqual(optimizer.n_jobs, 1)
        self.assertEqual(optimizer.sampler, "tpe")
        self.assertEqual(optimizer.pruner, "median")
        self.assertEqual(optimizer.default_metric, "sharpe_ratio")
        self.assertEqual(optimizer.direction, "maximize")
        self.assertEqual(optimizer.timeout_s, 300)

    def test_config_custom(self):
        config = {
            "n_trials": 50,
            "n_jobs": 4,
            "sampler": "random",
            "pruner": "none",
            "metric": "total_return",
            "direction": "minimize",
            "timeout_s": 60,
        }
        optimizer = OptunaOptimizer(self.engine, self.risk_manager, config)
        self.assertEqual(optimizer.n_trials, 50)
        self.assertEqual(optimizer.n_jobs, 4)
        self.assertEqual(optimizer.sampler, "random")
        self.assertEqual(optimizer.pruner, "none")
        self.assertEqual(optimizer.default_metric, "total_return")
        self.assertEqual(optimizer.direction, "minimize")
        self.assertEqual(optimizer.timeout_s, 60)


# ===================================================================
# TestWalkForwardOptunaIntegration
# ===================================================================

class TestWalkForwardOptunaIntegration(unittest.TestCase):
    """Test walk-forward optimizer with use_optuna=True."""

    def setUp(self):
        self.engine = MagicMock()
        self.risk_manager = MagicMock()

    def test_use_optuna_flag_stored(self):
        config = {"use_optuna": True, "optuna_config": {"n_trials": 5}}
        wf = WalkForwardOptimizer(self.engine, self.risk_manager, config)
        self.assertTrue(wf.use_optuna)
        self.assertEqual(wf.optuna_config["n_trials"], 5)

    def test_use_optuna_false_by_default(self):
        config = {}
        wf = WalkForwardOptimizer(self.engine, self.risk_manager, config)
        self.assertFalse(wf.use_optuna)
        self.assertIsNone(wf.optuna_config)

    def test_optimize_window_delegates_to_optuna(self):
        """When use_optuna=True, _optimize_window calls OptunaOptimizer."""
        config = {
            "use_optuna": True,
            "optuna_config": {
                "n_trials": 5,
                "param_space": {"p": [10, 20, 30]},
            },
            "min_oos_bars": 10,
        }
        wf = WalkForwardOptimizer(self.engine, self.risk_manager, config)

        # Mock the optuna optimizer via the module it's imported from
        with patch("backtest.optuna_optimizer.OptunaOptimizer") as MockOptuna:
            mock_opt = MagicMock()
            mock_opt.is_available.return_value = True
            mock_result = MagicMock()
            mock_result.best_params = {"p": 20}
            mock_result.best_score = 1.5
            mock_opt.optimize.return_value = mock_result
            MockOptuna.return_value = mock_opt

            df = _make_ohlcv_df(100)
            best_params, best_score = wf._optimize_window(
                df, MockStrategy, [{"p": 10}], "sharpe_ratio")

            self.assertEqual(best_params, {"p": 20})
            self.assertEqual(best_score, 1.5)

    def test_optimize_window_falls_back_without_param_space(self):
        """When use_optuna=True but no param_space, returns empty."""
        config = {
            "use_optuna": True,
            "optuna_config": {"n_trials": 5},
            "min_oos_bars": 10,
        }
        wf = WalkForwardOptimizer(self.engine, self.risk_manager, config)

        with patch("backtest.optuna_optimizer.OptunaOptimizer") as MockOptuna:
            mock_opt = MagicMock()
            mock_opt.is_available.return_value = True
            MockOptuna.return_value = mock_opt

            df = _make_ohlcv_df(100)
            best_params, best_score = wf._optimize_window(
                df, MockStrategy, [{"p": 10}], "sharpe_ratio")

            self.assertEqual(best_params, {})
            self.assertEqual(best_score, float("-inf"))

    def test_grid_search_still_works(self):
        """When use_optuna=False, grid search is used."""
        config = {"use_optuna": False, "min_oos_bars": 10}
        wf = WalkForwardOptimizer(self.engine, self.risk_manager, config)

        self.engine.run.return_value = _make_backtest_result(
            initial=10000, final=11000, n_trades=5)

        df = _make_ohlcv_df(100)
        best_params, best_score = wf._optimize_window_grid(
            df, MockStrategy, [{"p": 10}, {"p": 20}], "total_return")

        self.assertIn("p", best_params)
        self.assertGreater(best_score, float("-inf"))


if __name__ == "__main__":
    unittest.main()
