"""Portfolio-level performance aggregation across strategies."""

import logging
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 365


class PortfolioPerformanceTracker:
    """Aggregates performance metrics across all active strategies."""

    def __init__(self, timescale, strategy_names: list[str]):
        self.timescale = timescale
        self.strategy_names = strategy_names

    def compute_portfolio_metrics(self, window_hours: int = 168) -> dict:
        """Compute portfolio-level aggregate metrics.

        Returns:
            {total_pnl, portfolio_sharpe, portfolio_sortino, portfolio_max_dd,
             capital_efficiency, strategy_contributions: dict[str, float]}
        """
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=window_hours)

        # Aggregate PnL per strategy
        strategy_contributions = {}
        total_pnl = 0.0

        for name in self.strategy_names:
            try:
                fills = self.timescale.query_fills_by_strategy(name, start, end)
                pnl = sum(float(f.get("closed_pnl", 0.0)) for f in fills)
                strategy_contributions[name] = round(pnl, 4)
                total_pnl += pnl
            except Exception:
                logger.debug("Failed to query fills for %s", name)
                strategy_contributions[name] = 0.0

        # Equity-based metrics
        portfolio_sharpe = 0.0
        portfolio_sortino = 0.0
        portfolio_max_dd = 0.0
        capital_efficiency = 0.0

        try:
            equity_snapshots = self.timescale.query_equity_snapshots(start, end)
        except Exception:
            equity_snapshots = []

        if equity_snapshots:
            equity_values = [float(s.get("total_equity", 0.0)) for s in equity_snapshots]
            equity_series = pd.Series(equity_values)

            if len(equity_series) >= 2:
                returns = equity_series.pct_change().dropna()

                if len(returns) > 0 and returns.std() > 0:
                    periods_per_year = TRADING_DAYS_PER_YEAR * 24 / max(window_hours / len(returns), 1)
                    portfolio_sharpe = float(returns.mean() / returns.std() * math.sqrt(periods_per_year))

                    downside_returns = returns[returns < 0]
                    if len(downside_returns) > 0:
                        downside_std = downside_returns.std()
                        if downside_std > 0:
                            portfolio_sortino = float(
                                returns.mean() / downside_std * math.sqrt(periods_per_year))

                # Max drawdown
                cummax = equity_series.cummax()
                drawdown = (equity_series - cummax) / cummax
                portfolio_max_dd = abs(float(drawdown.min())) * 100

                # Capital efficiency: return per unit of avg equity
                avg_equity = equity_series.mean()
                if avg_equity > 0:
                    capital_efficiency = total_pnl / avg_equity

        return {
            "total_pnl": round(total_pnl, 4),
            "portfolio_sharpe": round(portfolio_sharpe, 4),
            "portfolio_sortino": round(portfolio_sortino, 4),
            "portfolio_max_dd": round(portfolio_max_dd, 2),
            "capital_efficiency": round(capital_efficiency, 6),
            "strategy_contributions": strategy_contributions,
        }

    def compute_benchmark_comparison(self, window_hours: int = 168) -> dict:
        """Compare portfolio return against BTC/ETH benchmarks.

        Returns:
            {portfolio_return, btc_return, eth_return, alpha, beta, tracking_error}
        """
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=window_hours)

        # Portfolio return from equity snapshots
        portfolio_return = 0.0
        try:
            equity_snapshots = self.timescale.query_equity_snapshots(start, end)
            if len(equity_snapshots) >= 2:
                first_eq = float(equity_snapshots[0].get("total_equity", 0))
                last_eq = float(equity_snapshots[-1].get("total_equity", 0))
                if first_eq > 0:
                    portfolio_return = (last_eq - first_eq) / first_eq
        except Exception:
            logger.debug("Failed to compute portfolio return")

        # BTC return from candles
        btc_return = self._compute_asset_return("BTC/USDC", start, end)
        eth_return = self._compute_asset_return("ETH/USDC", start, end)

        # Alpha and beta (simplified: alpha = portfolio - benchmark, beta from correlation)
        alpha = portfolio_return - btc_return

        # Beta and tracking error from returns series
        beta = 0.0
        tracking_error = 0.0
        try:
            equity_snapshots = self.timescale.query_equity_snapshots(start, end)
            btc_candles = self.timescale.query_candles("BTC/USDC", "1h", start, end)

            if len(equity_snapshots) >= 2 and len(btc_candles) >= 2:
                eq_values = [float(s.get("total_equity", 0)) for s in equity_snapshots]
                eq_returns = pd.Series(eq_values).pct_change().dropna().values

                btc_prices = [float(c.get("close", 0)) for c in btc_candles]
                btc_returns = pd.Series(btc_prices).pct_change().dropna().values

                # Align to same length
                min_len = min(len(eq_returns), len(btc_returns))
                if min_len > 1:
                    eq_r = eq_returns[:min_len]
                    btc_r = btc_returns[:min_len]

                    btc_var = np.var(btc_r)
                    if btc_var > 0:
                        beta = float(np.cov(eq_r, btc_r)[0][1] / btc_var)

                    diff = eq_r - btc_r
                    tracking_error = float(np.std(diff) * math.sqrt(TRADING_DAYS_PER_YEAR * 24))
        except Exception:
            logger.debug("Failed to compute beta/tracking error")

        return {
            "portfolio_return": round(portfolio_return, 6),
            "btc_return": round(btc_return, 6),
            "eth_return": round(eth_return, 6),
            "alpha": round(alpha, 6),
            "beta": round(beta, 4),
            "tracking_error": round(tracking_error, 6),
        }

    def _compute_asset_return(self, symbol: str, start: datetime,
                               end: datetime) -> float:
        """Compute simple return for an asset from candle data."""
        try:
            candles = self.timescale.query_candles(symbol, "1h", start, end)
            if len(candles) >= 2:
                first_close = float(candles[0].get("close", 0))
                last_close = float(candles[-1].get("close", 0))
                if first_close > 0:
                    return (last_close - first_close) / first_close
        except Exception:
            logger.debug("Failed to get candles for %s", symbol)
        return 0.0
