"""Walk-forward optimization to prevent overfitting.

Splits data into N IS/OOS windows. Optimizes params on IS, validates on OOS.
Reports aggregated OOS performance as realistic expected performance.
"""

import copy
import logging
import time
from collections import Counter
from typing import Any, Optional

import numpy as np
import pandas as pd

from backtest.param_grid import ParamGrid
from backtest.walk_forward_result import WalkForwardResult

logger = logging.getLogger(__name__)


class WalkForwardOptimizer:
    """Walk-forward optimization to prevent overfitting.

    Splits data into N IS/OOS windows. Optimizes params on IS, validates on OOS.
    Reports aggregated OOS performance as realistic expected performance.

    Config: n_splits=5, is_ratio=0.7, min_oos_bars=100, metric="sharpe_ratio"
    """

    def __init__(self, backtest_engine, risk_manager, config: dict):
        self.backtest_engine = backtest_engine
        self.risk_manager = risk_manager
        self.n_splits = config.get("n_splits", 5)
        self.is_ratio = config.get("is_ratio", 0.7)
        self.min_oos_bars = config.get("min_oos_bars", 100)
        self.decay_threshold = config.get("decay_threshold", 50.0)

    def run(self, df: pd.DataFrame, strategy_class, param_grid: dict,
            metric: str = "sharpe_ratio") -> WalkForwardResult:
        """Run full walk-forward optimization.

        Args:
            df: OHLCV DataFrame with DatetimeIndex.
            strategy_class: Strategy class (BaseStrategy subclass) to instantiate.
            param_grid: Parameter grid config dict.
            metric: Metric to optimize on (sharpe_ratio, total_return, profit_factor).

        Returns:
            WalkForwardResult with per-window and aggregated results.
        """
        start_time = time.time()
        grid = ParamGrid(param_grid)
        param_combos = grid.expand()

        if not param_combos:
            return WalkForwardResult(
                strategy_class=strategy_class.__name__,
                param_grid=param_grid,
                duration_seconds=time.time() - start_time,
            )

        windows = self._split_windows(df)
        if not windows:
            return WalkForwardResult(
                strategy_class=strategy_class.__name__,
                param_grid=param_grid,
                duration_seconds=time.time() - start_time,
            )

        result_windows = []
        all_best_params = []
        decay_values = []

        for i, (is_data, oos_data) in enumerate(windows):
            logger.info("Walk-forward window %d/%d: IS=%d bars, OOS=%d bars",
                        i + 1, len(windows), len(is_data), len(oos_data))

            # Optimize on in-sample
            best_params, is_score = self._optimize_window(
                is_data, strategy_class, param_combos, metric)

            # Validate on out-of-sample
            oos_metrics = self._validate_window(
                oos_data, strategy_class, best_params)

            # Run IS with best params for comparison
            is_metrics = self._validate_window(
                is_data, strategy_class, best_params)

            # Compute decay
            decay = self.compute_decay(is_metrics, oos_metrics, metric)

            result_windows.append({
                "window": i,
                "is_bars": len(is_data),
                "oos_bars": len(oos_data),
                "is_params": best_params,
                "is_metrics": is_metrics,
                "oos_metrics": oos_metrics,
                "decay_pct": decay,
            })

            all_best_params.append(best_params)
            decay_values.append(decay)

        # Aggregate OOS metrics
        aggregated_oos = self._aggregate_oos_metrics(
            [w["oos_metrics"] for w in result_windows])

        # Find most common best params
        best_params = self._most_common_params(all_best_params)

        avg_decay = float(np.mean(decay_values)) if decay_values else 0.0
        is_valid = avg_decay < self.decay_threshold

        return WalkForwardResult(
            windows=result_windows,
            aggregated_oos=aggregated_oos,
            best_params=best_params,
            avg_decay_pct=avg_decay,
            is_valid=is_valid,
            strategy_class=strategy_class.__name__,
            param_grid=param_grid,
            duration_seconds=time.time() - start_time,
        )

    def _split_windows(self, df: pd.DataFrame) -> list[tuple]:
        """Split data into anchored walk-forward windows.

        Window i: IS = data[0 : start + i*step], OOS = data[start + i*step : start + (i+1)*step]
        """
        n = len(df)
        if n < self.min_oos_bars * 2:
            logger.warning("Insufficient data for walk-forward: %d bars", n)
            return []

        # Calculate the initial IS size and OOS step
        total_oos_bars = int(n * (1 - self.is_ratio))
        oos_step = max(self.min_oos_bars, total_oos_bars // self.n_splits)

        # Ensure we have enough bars
        initial_is_end = int(n * self.is_ratio)
        if initial_is_end < self.min_oos_bars:
            initial_is_end = n - oos_step * self.n_splits
            if initial_is_end < self.min_oos_bars:
                logger.warning("Cannot create valid walk-forward windows")
                return []

        windows = []
        for i in range(self.n_splits):
            is_end = initial_is_end + i * oos_step
            oos_start = is_end
            oos_end = min(oos_start + oos_step, n)

            if oos_start >= n or (oos_end - oos_start) < max(1, self.min_oos_bars // 2):
                break

            is_data = df.iloc[:is_end].copy()
            oos_data = df.iloc[oos_start:oos_end].copy()

            if len(oos_data) > 0:
                windows.append((is_data, oos_data))

        return windows

    def _optimize_window(self, is_data: pd.DataFrame, strategy_class,
                         param_combos: list[dict], metric: str) -> tuple[dict, float]:
        """Grid search on in-sample data, return best params and score."""
        best_params = param_combos[0] if param_combos else {}
        best_score = float("-inf")

        for params in param_combos:
            try:
                strategy = strategy_class(params)
                result = self.backtest_engine.run(is_data, strategy, self.risk_manager)
                score = self._extract_metric(result, metric)
                if score > best_score:
                    best_score = score
                    best_params = params
            except Exception:
                logger.debug("Param combo failed: %s", params)
                continue

        return best_params, best_score

    def _validate_window(self, data: pd.DataFrame, strategy_class,
                         best_params: dict) -> dict:
        """Run backtest on data with given params, return metrics dict."""
        try:
            strategy = strategy_class(best_params)
            result = self.backtest_engine.run(data, strategy, self.risk_manager)
            return self._compute_metrics(result)
        except Exception:
            logger.exception("Validation backtest failed")
            return {"sharpe_ratio": 0.0, "total_return": 0.0, "profit_factor": 0.0,
                    "win_rate": 0.0, "trade_count": 0, "max_drawdown": 0.0}

    def compute_decay(self, is_metrics: dict, oos_metrics: dict,
                      metric: str = "sharpe_ratio") -> float:
        """Compute IS->OOS performance decay percentage.

        decay = max(0, (is_metric - oos_metric) / abs(is_metric) * 100)
        """
        is_val = is_metrics.get(metric, 0.0)
        oos_val = oos_metrics.get(metric, 0.0)

        if abs(is_val) < 1e-10:
            return 0.0

        decay = (is_val - oos_val) / abs(is_val) * 100.0
        return max(0.0, decay)

    def _extract_metric(self, result, metric: str) -> float:
        """Extract the optimization metric from a backtest result."""
        metrics = self._compute_metrics(result)
        return metrics.get(metric, 0.0)

    def _compute_metrics(self, result) -> dict:
        """Compute standard metrics from a BacktestResult."""
        trades = result.trades
        equity_curve = result.equity_curve

        trade_count = len(trades)
        if trade_count == 0:
            return {
                "sharpe_ratio": 0.0,
                "total_return": 0.0,
                "profit_factor": 0.0,
                "win_rate": 0.0,
                "trade_count": 0,
                "max_drawdown": 0.0,
            }

        # Total return
        total_return = (result.final_equity - result.initial_capital) / result.initial_capital

        # Win rate
        winners = [t for t in trades if t.pnl > 0]
        win_rate = len(winners) / trade_count * 100.0

        # Profit factor
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Max drawdown
        if len(equity_curve) > 0:
            peak = equity_curve.cummax()
            dd = (equity_curve - peak) / peak * 100
            max_drawdown = abs(dd.min()) if len(dd) > 0 else 0.0
        else:
            max_drawdown = 0.0

        # Sharpe ratio (annualized, assuming hourly bars)
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

    def _aggregate_oos_metrics(self, oos_metrics_list: list[dict]) -> dict:
        """Average OOS metrics across all windows."""
        if not oos_metrics_list:
            return {}

        aggregated = {}
        keys = oos_metrics_list[0].keys()
        for key in keys:
            values = [m.get(key, 0.0) for m in oos_metrics_list]
            aggregated[key] = float(np.mean(values))
        return aggregated

    def _most_common_params(self, all_params: list[dict]) -> dict:
        """Find the most frequently selected params across windows."""
        if not all_params:
            return {}

        # Count each param combination
        param_strs = [str(sorted(p.items())) for p in all_params]
        counter = Counter(param_strs)
        most_common_str = counter.most_common(1)[0][0]

        # Find the corresponding dict
        for p in all_params:
            if str(sorted(p.items())) == most_common_str:
                return p

        return all_params[0]
