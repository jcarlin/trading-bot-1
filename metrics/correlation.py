"""Correlation analysis across strategy returns."""

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """Computes pairwise correlation between strategy returns and marginal risk contribution."""

    def __init__(self, timescale, strategy_names: list[str]):
        self.timescale = timescale
        self.strategy_names = strategy_names

    def compute_correlation_matrix(self, window_hours: int = 168) -> dict:
        """Compute pairwise Pearson correlation matrix of hourly PnL.

        Returns:
            {matrix: list[list[float]], strategy_names: list, period_hours: int}
        """
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=window_hours)

        # Get hourly PnL series for each strategy
        pnl_series = {}
        for name in self.strategy_names:
            try:
                fills = self.timescale.query_fills_by_strategy(name, start, end)
                hourly_pnl = self._bin_fills_hourly(fills, start, end)
                pnl_series[name] = hourly_pnl
            except Exception:
                logger.debug("Failed to get fills for %s", name)
                pnl_series[name] = np.zeros(window_hours)

        n = len(self.strategy_names)
        if n == 0:
            return {"matrix": [], "strategy_names": [], "period_hours": window_hours}

        # Build matrix from numpy arrays
        data = np.array([pnl_series[name] for name in self.strategy_names])

        # Compute correlation
        if data.shape[0] < 2:
            matrix = [[1.0]]
        else:
            # Handle constant series (std=0)
            stds = np.std(data, axis=1)
            if np.any(stds == 0):
                matrix = np.eye(n).tolist()
            else:
                corr = np.corrcoef(data)
                matrix = [[round(float(corr[i][j]), 4) for j in range(n)] for i in range(n)]

        return {
            "matrix": matrix,
            "strategy_names": list(self.strategy_names),
            "period_hours": window_hours,
        }

    def compute_marginal_contribution(self, window_hours: int = 168) -> dict:
        """Compute marginal risk and return contribution per strategy.

        Returns:
            {strategy_name: {marginal_risk: float, marginal_return: float, info_ratio: float}}
        """
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=window_hours)

        result = {}
        pnl_series = {}

        for name in self.strategy_names:
            try:
                fills = self.timescale.query_fills_by_strategy(name, start, end)
                hourly_pnl = self._bin_fills_hourly(fills, start, end)
                pnl_series[name] = hourly_pnl
            except Exception:
                logger.debug("Failed to get fills for %s", name)
                pnl_series[name] = np.zeros(window_hours)

        # Portfolio total
        all_pnl = np.array([pnl_series[name] for name in self.strategy_names])
        portfolio_pnl = np.sum(all_pnl, axis=0) if len(all_pnl) > 0 else np.zeros(window_hours)
        portfolio_std = float(np.std(portfolio_pnl)) if len(portfolio_pnl) > 0 else 0.0

        for name in self.strategy_names:
            strategy_pnl = pnl_series[name]

            # Marginal return: mean of this strategy's PnL
            marginal_return = float(np.mean(strategy_pnl))

            # Marginal risk: std of portfolio without this strategy vs with
            others_pnl = portfolio_pnl - strategy_pnl
            others_std = float(np.std(others_pnl))
            marginal_risk = portfolio_std - others_std

            # Info ratio: marginal_return / marginal_risk
            info_ratio = (marginal_return / marginal_risk) if marginal_risk > 0 else 0.0

            result[name] = {
                "marginal_risk": round(marginal_risk, 6),
                "marginal_return": round(marginal_return, 6),
                "info_ratio": round(info_ratio, 4),
            }

        return result

    def _bin_fills_hourly(self, fills: list[dict], start: datetime,
                          end: datetime) -> np.ndarray:
        """Bin fill PnLs into hourly buckets."""
        hours = int((end - start).total_seconds() / 3600)
        buckets = np.zeros(max(hours, 1))

        for fill in fills:
            fill_time = fill.get("time")
            if fill_time is None:
                continue
            if isinstance(fill_time, str):
                fill_time = datetime.fromisoformat(fill_time)
            if not fill_time.tzinfo:
                fill_time = fill_time.replace(tzinfo=timezone.utc)

            offset = (fill_time - start).total_seconds() / 3600
            idx = int(offset)
            if 0 <= idx < len(buckets):
                buckets[idx] += float(fill.get("closed_pnl", 0.0))

        return buckets
