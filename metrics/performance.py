"""Performance metrics and reporting for backtest results.

Computes standard trading metrics (Sharpe, drawdown, win-rate, etc.)
and optionally generates an HTML tearsheet via quantstats.
"""

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from backtest.engine import BacktestResult

logger = logging.getLogger(__name__)

# Crypto markets run 365 days a year.
TRADING_DAYS_PER_YEAR = 365


class PerformanceReporter:
    """Compute and present performance metrics for a BacktestResult.

    Args:
        result: A BacktestResult produced by the BacktestEngine.
    """

    def __init__(self, result: BacktestResult):
        self.result = result
        self._summary: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return a dictionary of key performance metrics.

        Metrics include win-rate, Sharpe ratio, max drawdown,
        profit factor, and overall return statistics.
        """
        if self._summary is not None:
            return self._summary

        trades = self.result.trades
        equity = self.result.equity_curve

        total_trades = len(trades)
        winners = [t for t in trades if t.is_winner]
        losers = [t for t in trades if not t.is_winner]
        winning_trades = len(winners)
        losing_trades = len(losers)
        win_rate = (winning_trades / total_trades * 100) if total_trades else 0.0

        pnl_list = [t.pnl for t in trades]
        total_pnl = sum(pnl_list)
        avg_trade_pnl = (total_pnl / total_trades) if total_trades else 0.0
        best_trade = max(pnl_list) if pnl_list else 0.0
        worst_trade = min(pnl_list) if pnl_list else 0.0

        gross_profit = sum(t.pnl for t in winners)
        gross_loss = abs(sum(t.pnl for t in losers))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        max_dd = self._max_drawdown(equity)
        sharpe = self._sharpe_ratio(equity)

        total_return_pct = (
            (self.result.final_equity - self.result.initial_capital)
            / self.result.initial_capital
            * 100
        )

        self._summary = {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 2),
            "avg_trade_pnl": round(avg_trade_pnl, 2),
            "best_trade": round(best_trade, 2),
            "worst_trade": round(worst_trade, 2),
            "profit_factor": round(profit_factor, 4),
            "max_drawdown": round(max_dd, 2),
            "sharpe_ratio": round(sharpe, 4),
            "initial_capital": round(self.result.initial_capital, 2),
            "final_equity": round(self.result.final_equity, 2),
            "total_return_pct": round(total_return_pct, 2),
        }
        return self._summary

    def print_summary(self) -> None:
        """Print a formatted performance summary to the console."""
        s = self.summary()

        header = " Backtest Performance Summary "
        print(f"\n{'=' * 50}")
        print(f"{header:=^50}")
        print(f"{'=' * 50}\n")

        print(f"  Initial Capital:    ${s['initial_capital']:>12,.2f}")
        print(f"  Final Equity:       ${s['final_equity']:>12,.2f}")
        print(f"  Total Return:        {s['total_return_pct']:>12.2f} %")
        print()
        print(f"  Total Trades:        {s['total_trades']:>12d}")
        print(f"  Winning Trades:      {s['winning_trades']:>12d}")
        print(f"  Losing Trades:       {s['losing_trades']:>12d}")
        print(f"  Win Rate:            {s['win_rate']:>12.2f} %")
        print()
        print(f"  Total PnL:          ${s['total_pnl']:>12,.2f}")
        print(f"  Avg Trade PnL:      ${s['avg_trade_pnl']:>12,.2f}")
        print(f"  Best Trade:         ${s['best_trade']:>12,.2f}")
        print(f"  Worst Trade:        ${s['worst_trade']:>12,.2f}")
        print()
        print(f"  Profit Factor:       {s['profit_factor']:>12.4f}")
        print(f"  Sharpe Ratio:        {s['sharpe_ratio']:>12.4f}")
        print(f"  Max Drawdown:        {s['max_drawdown']:>12.2f} %")
        print(f"\n{'=' * 50}\n")

    def generate_report(self, output_path: str = "backtest_report.html") -> None:
        """Generate an HTML report.

        Uses quantstats for a full tearsheet if available; otherwise falls
        back to writing a simple HTML table with the summary metrics.
        """
        equity = self.result.equity_curve
        returns = equity.pct_change().dropna()

        try:
            import quantstats as qs  # type: ignore[import-untyped]

            qs.reports.html(returns, output=output_path, title="Backtest Report")
            logger.info("Quantstats HTML report saved to %s", output_path)
        except ImportError:
            logger.warning(
                "quantstats not installed — generating basic HTML report."
            )
            self._write_basic_html(output_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _max_drawdown(equity: pd.Series) -> float:
        """Compute the maximum drawdown as a percentage.

        Returns a non-negative number (e.g. 15.3 means a 15.3 % peak-to-
        trough decline).
        """
        if equity.empty:
            return 0.0
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        return abs(drawdown.min()) * 100

    @staticmethod
    def _sharpe_ratio(equity: pd.Series) -> float:
        """Annualised Sharpe ratio from the equity curve.

        Assumes 365 trading days (crypto) and a risk-free rate of 0.
        """
        if len(equity) < 2:
            return 0.0

        returns = equity.pct_change().dropna()
        if returns.std() == 0:
            return 0.0

        return float(
            returns.mean() / returns.std() * math.sqrt(TRADING_DAYS_PER_YEAR)
        )

    def _write_basic_html(self, output_path: str) -> None:
        """Write a minimal HTML file containing the summary table."""
        s = self.summary()
        rows = "\n".join(
            f"    <tr><td>{k}</td><td>{v}</td></tr>" for k, v in s.items()
        )
        html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Backtest Report</title>
  <style>
    body {{ font-family: monospace; margin: 2em; }}
    table {{ border-collapse: collapse; width: 500px; }}
    td, th {{ border: 1px solid #ccc; padding: 6px 12px; text-align: left; }}
    th {{ background: #f5f5f5; }}
  </style>
</head>
<body>
  <h1>Backtest Report</h1>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
{rows}
  </table>
</body>
</html>"""
        with open(output_path, "w") as f:
            f.write(html)
        logger.info("Basic HTML report saved to %s", output_path)
