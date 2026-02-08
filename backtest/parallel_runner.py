"""Parallel backtest runner using process pools.

Runs multiple backtests concurrently for parameter sweeps, Monte Carlo
simulations, and strategy comparisons.
"""

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from backtest.monte_carlo_result import MonteCarloResult

logger = logging.getLogger(__name__)


def _run_single_backtest(df_bytes, strategy_module, strategy_class_name,
                         params, risk_config, initial_capital, commission_pct):
    """Run a single backtest in a worker process.

    Args are kept serializable for process pool transfer.
    Returns a dict with params, metrics, duration_s, and error.
    """
    import io
    import importlib
    import time as _time

    start = _time.time()
    try:
        # Reconstruct DataFrame
        df = pd.read_parquet(io.BytesIO(df_bytes))

        # Reconstruct engine and risk manager
        from backtest.engine import BacktestEngine
        from risk.manager import RiskManager

        engine = BacktestEngine(initial_capital=initial_capital,
                                commission_pct=commission_pct)
        rm = RiskManager(risk_config)

        # Reconstruct strategy
        mod = importlib.import_module(strategy_module)
        cls = getattr(mod, strategy_class_name)
        strategy = cls(params)

        result = engine.run(df, strategy, rm)

        # Compute metrics
        metrics = _compute_metrics_standalone(result)
        return {
            "params": params,
            "metrics": metrics,
            "duration_s": _time.time() - start,
            "error": None,
        }
    except Exception as e:
        return {
            "params": params,
            "metrics": {},
            "duration_s": _time.time() - start,
            "error": str(e),
        }


def _compute_metrics_standalone(result) -> dict:
    """Compute standard metrics from a BacktestResult (standalone function for workers)."""
    trades = result.trades
    equity_curve = result.equity_curve

    trade_count = len(trades)
    if trade_count == 0:
        return {
            "sharpe_ratio": 0.0, "total_return": 0.0, "profit_factor": 0.0,
            "win_rate": 0.0, "trade_count": 0, "max_drawdown": 0.0,
        }

    total_return = (result.final_equity - result.initial_capital) / result.initial_capital

    winners = [t for t in trades if t.pnl > 0]
    win_rate = len(winners) / trade_count * 100.0

    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    if len(equity_curve) > 0:
        peak = equity_curve.cummax()
        dd = (equity_curve - peak) / peak * 100
        max_drawdown = abs(dd.min()) if len(dd) > 0 else 0.0
    else:
        max_drawdown = 0.0

    if len(equity_curve) > 1:
        returns = equity_curve.pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = float(returns.mean() / returns.std() * np.sqrt(8760))
        else:
            sharpe_ratio = 0.0
    else:
        sharpe_ratio = 0.0

    return {
        "sharpe_ratio": sharpe_ratio,
        "total_return": total_return,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "trade_count": trade_count,
        "max_drawdown": max_drawdown,
    }


class ParallelBacktestRunner:
    """Runs multiple backtests in parallel using process pools.

    Config keys:
        max_workers: Number of parallel workers (default: cpu_count - 1).
        chunk_size: Not used directly by ProcessPoolExecutor.submit but kept for API.
        timeout_per_backtest_s: Timeout per individual backtest (default 60).
    """

    def __init__(self, config: dict = None):
        config = config or {}
        self.max_workers = config.get("max_workers", max(1, os.cpu_count() - 1))
        self.chunk_size = config.get("chunk_size", 10)
        self.timeout_per_backtest_s = config.get("timeout_per_backtest_s", 60)

    def run_param_sweep(self, df: pd.DataFrame, strategy_class,
                        param_combos: list[dict], risk_config: dict,
                        initial_capital: float = 10000,
                        commission_pct: float = 0.001) -> list[dict]:
        """Run backtests for all param combos in parallel.

        Args:
            df: OHLCV DataFrame.
            strategy_class: BaseStrategy subclass.
            param_combos: List of parameter dicts.
            risk_config: Risk manager config dict.
            initial_capital: Starting capital.
            commission_pct: Commission percentage.

        Returns:
            List of dicts: [{params, metrics, duration_s, error}]
        """
        import io

        # Serialize DataFrame once
        buf = io.BytesIO()
        df.to_parquet(buf)
        df_bytes = buf.getvalue()

        strategy_module = strategy_class.__module__
        strategy_class_name = strategy_class.__name__

        results = []

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_params = {}
            for params in param_combos:
                future = executor.submit(
                    _run_single_backtest,
                    df_bytes, strategy_module, strategy_class_name,
                    params, risk_config, initial_capital, commission_pct,
                )
                future_to_params[future] = params

            for future in as_completed(future_to_params):
                try:
                    result = future.result(timeout=self.timeout_per_backtest_s)
                    results.append(result)
                except Exception as e:
                    params = future_to_params[future]
                    results.append({
                        "params": params,
                        "metrics": {},
                        "duration_s": 0.0,
                        "error": str(e),
                    })

        return results

    def run_monte_carlo(self, df: pd.DataFrame, strategy_class, params: dict,
                        risk_config: dict, n_simulations: int = 100,
                        shuffle_mode: str = "returns",
                        initial_capital: float = 10000,
                        commission_pct: float = 0.001) -> MonteCarloResult:
        """Monte Carlo simulation by shuffling returns or blocks.

        Runs the strategy on the original data once, then on n_simulations
        shuffled versions to build a distribution of metrics.

        Args:
            df: OHLCV DataFrame.
            strategy_class: BaseStrategy subclass.
            params: Strategy parameters.
            risk_config: Risk manager config dict.
            n_simulations: Number of shuffled simulations.
            shuffle_mode: "returns" (shuffle daily returns) or "block" (shuffle blocks).
            initial_capital: Starting capital.
            commission_pct: Commission percentage.

        Returns:
            MonteCarloResult with distributions and statistics.
        """
        start_time = time.time()

        # Run original backtest
        from backtest.engine import BacktestEngine
        from risk.manager import RiskManager

        engine = BacktestEngine(initial_capital=initial_capital,
                                commission_pct=commission_pct)
        rm = RiskManager(risk_config)

        strategy = strategy_class(params)
        original_result = engine.run(df, strategy, rm)
        original_metrics = _compute_metrics_standalone(original_result)

        # Generate shuffled DataFrames and run backtests
        shuffled_dfs = self._generate_shuffled_data(df, n_simulations, shuffle_mode)

        sim_results = []
        for shuffled_df in shuffled_dfs:
            try:
                strat = strategy_class(params)
                result = engine.run(shuffled_df, strat, rm)
                metrics = _compute_metrics_standalone(result)
                sim_results.append(metrics)
            except Exception:
                logger.debug("Monte Carlo simulation failed")
                continue

        # Build distributions
        metric_names = ["sharpe_ratio", "total_return", "profit_factor",
                        "win_rate", "max_drawdown"]
        metric_distributions = {}
        percentiles = {}
        confidence_intervals = {}

        for metric_name in metric_names:
            values = [m.get(metric_name, 0.0) for m in sim_results]
            if values:
                metric_distributions[metric_name] = values
                percentiles[metric_name] = {
                    5: float(np.percentile(values, 5)),
                    25: float(np.percentile(values, 25)),
                    50: float(np.percentile(values, 50)),
                    75: float(np.percentile(values, 75)),
                    95: float(np.percentile(values, 95)),
                }
                confidence_intervals[metric_name] = {
                    "lower": float(np.percentile(values, 2.5)),
                    "upper": float(np.percentile(values, 97.5)),
                    "level": 0.95,
                }

        # Compute p-value: fraction of simulations that beat original
        original_sharpe = original_metrics.get("sharpe_ratio", 0.0)
        sim_sharpes = metric_distributions.get("sharpe_ratio", [])
        if sim_sharpes:
            p_value = sum(1 for s in sim_sharpes if s >= original_sharpe) / len(sim_sharpes)
        else:
            p_value = 1.0

        return MonteCarloResult(
            n_simulations=len(sim_results),
            metric_distributions=metric_distributions,
            percentiles=percentiles,
            confidence_intervals=confidence_intervals,
            original_metrics=original_metrics,
            p_value_vs_random=p_value,
            duration_seconds=time.time() - start_time,
        )

    def run_strategy_comparison(self, df: pd.DataFrame,
                                strategies: list[tuple],
                                risk_config: dict,
                                initial_capital: float = 10000,
                                commission_pct: float = 0.001) -> list[dict]:
        """Run different strategies on same data in parallel.

        Args:
            df: OHLCV DataFrame.
            strategies: List of (strategy_class, params) tuples.
            risk_config: Risk manager config dict.
            initial_capital: Starting capital.
            commission_pct: Commission percentage.

        Returns:
            List of dicts: [{strategy_name, params, metrics}]
        """
        from backtest.engine import BacktestEngine
        from risk.manager import RiskManager

        engine = BacktestEngine(initial_capital=initial_capital,
                                commission_pct=commission_pct)
        rm = RiskManager(risk_config)

        results = []
        for strategy_class, params in strategies:
            try:
                strategy = strategy_class(params)
                result = engine.run(df, strategy, rm)
                metrics = _compute_metrics_standalone(result)
                results.append({
                    "strategy_name": strategy_class.__name__,
                    "params": params,
                    "metrics": metrics,
                })
            except Exception as e:
                results.append({
                    "strategy_name": strategy_class.__name__,
                    "params": params,
                    "metrics": {},
                    "error": str(e),
                })

        return results

    def _generate_shuffled_data(self, df: pd.DataFrame, n: int,
                                 mode: str) -> list[pd.DataFrame]:
        """Generate n shuffled versions of the OHLCV data.

        Args:
            df: Original OHLCV DataFrame.
            n: Number of shuffled copies.
            mode: "returns" or "block".

        Returns:
            List of shuffled DataFrames.
        """
        shuffled = []

        if mode == "block":
            block_size = max(5, len(df) // 20)
            n_blocks = len(df) // block_size

            for _ in range(n):
                indices = list(range(n_blocks))
                np.random.shuffle(indices)
                new_blocks = []
                for idx in indices:
                    start = idx * block_size
                    end = min(start + block_size, len(df))
                    new_blocks.append(df.iloc[start:end].copy())
                # Append remainder
                remainder_start = n_blocks * block_size
                if remainder_start < len(df):
                    new_blocks.append(df.iloc[remainder_start:].copy())
                new_df = pd.concat(new_blocks, ignore_index=False)
                new_df.index = df.index[:len(new_df)]
                if hasattr(df, 'attrs'):
                    new_df.attrs = df.attrs.copy()
                shuffled.append(new_df)
        else:
            # Shuffle returns mode
            close = df["close"].values.copy()
            returns = np.diff(close) / close[:-1]

            for _ in range(n):
                shuffled_returns = returns.copy()
                np.random.shuffle(shuffled_returns)

                # Reconstruct prices
                new_close = np.zeros(len(close))
                new_close[0] = close[0]
                for j in range(len(shuffled_returns)):
                    new_close[j + 1] = new_close[j] * (1 + shuffled_returns[j])

                # Build OHLCV from shuffled close
                ratio_open = (df["open"].values / df["close"].values)
                ratio_high = (df["high"].values / df["close"].values)
                ratio_low = (df["low"].values / df["close"].values)

                new_df = pd.DataFrame({
                    "open": new_close * ratio_open,
                    "high": new_close * ratio_high,
                    "low": new_close * ratio_low,
                    "close": new_close,
                    "volume": df["volume"].values.copy(),
                }, index=df.index)
                if hasattr(df, 'attrs'):
                    new_df.attrs = df.attrs.copy()
                shuffled.append(new_df)

        return shuffled
