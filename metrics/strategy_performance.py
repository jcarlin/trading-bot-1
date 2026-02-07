"""Per-strategy performance metrics computed from live fill data."""

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 365


class StrategyPerformanceTracker:
    """Computes rolling-window performance metrics from live fills."""

    def __init__(self, timescale, strategy_name: str):
        self.timescale = timescale
        self.strategy_name = strategy_name

    def compute_metrics(self, window_hours: int) -> dict:
        """Compute performance metrics for a given time window.

        Args:
            window_hours: Number of hours to look back.

        Returns:
            Dict with keys: trade_count, total_pnl, win_rate, avg_win,
            avg_loss, profit_factor, sharpe_ratio, sortino_ratio,
            max_drawdown, calmar_ratio.
        """
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=window_hours)

        # Query fills and equity snapshots
        try:
            fills = self.timescale.query_fills_by_strategy(
                self.strategy_name, start, end)
        except Exception:
            logger.exception("Failed to query fills")
            fills = []

        try:
            equity_snapshots = self.timescale.query_equity_snapshots(start, end)
        except Exception:
            logger.exception("Failed to query equity snapshots")
            equity_snapshots = []

        # Compute fill-based metrics
        trade_count = len(fills)
        if trade_count == 0:
            return self._empty_metrics()

        pnl_values = [float(f.get("closed_pnl", 0.0)) for f in fills]
        total_pnl = sum(pnl_values)

        winners = [p for p in pnl_values if p > 0]
        losers = [p for p in pnl_values if p <= 0]

        win_rate = (len(winners) / trade_count * 100) if trade_count > 0 else 0.0
        avg_win = (sum(winners) / len(winners)) if winners else 0.0
        avg_loss = (sum(losers) / len(losers)) if losers else 0.0

        gross_profit = sum(winners) if winners else 0.0
        gross_loss = abs(sum(losers)) if losers else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        # Compute equity-based metrics
        sharpe = 0.0
        sortino = 0.0
        max_dd = 0.0
        calmar = 0.0

        if equity_snapshots:
            equity_values = [float(s.get("total_equity", 0.0)) for s in equity_snapshots]
            equity_series = pd.Series(equity_values)

            if len(equity_series) >= 2:
                returns = equity_series.pct_change().dropna()

                if len(returns) > 0 and returns.std() > 0:
                    # Annualize based on the window
                    periods_per_year = TRADING_DAYS_PER_YEAR * 24 / max(window_hours / len(returns), 1)
                    sharpe = float(returns.mean() / returns.std() * math.sqrt(periods_per_year))

                    # Sortino: downside deviation only
                    downside_returns = returns[returns < 0]
                    if len(downside_returns) > 0:
                        downside_std = downside_returns.std()
                        if downside_std > 0:
                            sortino = float(returns.mean() / downside_std * math.sqrt(periods_per_year))

                # Max drawdown
                if len(equity_series) > 0:
                    cummax = equity_series.cummax()
                    drawdown = (equity_series - cummax) / cummax
                    max_dd = abs(drawdown.min()) * 100

                    # Calmar ratio
                    if max_dd > 0:
                        total_return = (equity_series.iloc[-1] - equity_series.iloc[0]) / equity_series.iloc[0]
                        annualized_return = total_return * (TRADING_DAYS_PER_YEAR * 24 / window_hours)
                        calmar = float(annualized_return / (max_dd / 100))

        return {
            "trade_count": trade_count,
            "total_pnl": round(total_pnl, 4),
            "win_rate": round(win_rate, 2),
            "avg_win": round(avg_win, 4),
            "avg_loss": round(avg_loss, 4),
            "profit_factor": round(profit_factor, 4),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "max_drawdown": round(max_dd, 2),
            "calmar_ratio": round(calmar, 4),
        }

    def compute_all_windows(self, windows: list[int]) -> dict[int, dict]:
        """Compute metrics across multiple time windows.

        Args:
            windows: List of window sizes in hours.

        Returns:
            Dict mapping window_hours to metrics dict.
        """
        results = {}
        for window in windows:
            results[window] = self.compute_metrics(window)
        return results

    @staticmethod
    def _empty_metrics() -> dict:
        return {
            "trade_count": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
        }
