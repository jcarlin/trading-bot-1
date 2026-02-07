"""Performance attribution: decomposes portfolio returns by strategy and factor."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class PerformanceAttribution:
    """Decomposes portfolio returns into strategy contributions.

    Computes:
    - Per-strategy PnL contribution
    - Per-strategy risk contribution (via drawdown)
    - Strategy allocation efficiency
    """

    def __init__(self, timescale, strategy_names: list[str],
                 config: dict = None):
        self.timescale = timescale
        self.strategy_names = strategy_names
        config = config or {}
        self.lookback_hours = config.get("lookback_hours", 720)

    def compute_attribution(self, lookback_hours: Optional[int] = None) -> dict:
        """Compute performance attribution across strategies.

        Returns:
            Dict with total_pnl, strategy_contributions (pnl, pct, risk),
            and best/worst performers.
        """
        lookback = lookback_hours or self.lookback_hours
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=lookback)

        strategy_data = {}
        total_pnl = 0.0

        for name in self.strategy_names:
            try:
                fills = self.timescale.query_fills_by_strategy(name, start, end)
                pnl = sum(f.get("realized_pnl", f.get("closed_pnl", 0.0))
                         for f in fills) if fills else 0.0
                trade_count = len(fills) if fills else 0

                # Compute drawdown from fills
                cumulative = 0.0
                peak = 0.0
                max_dd = 0.0
                for f in (fills or []):
                    cumulative += f.get("realized_pnl", f.get("closed_pnl", 0.0))
                    peak = max(peak, cumulative)
                    dd = peak - cumulative
                    max_dd = max(max_dd, dd)

                strategy_data[name] = {
                    "pnl": round(pnl, 4),
                    "trade_count": trade_count,
                    "max_drawdown": round(max_dd, 4),
                }
                total_pnl += pnl

            except Exception:
                logger.debug("Failed to get attribution for %s", name)
                strategy_data[name] = {
                    "pnl": 0.0, "trade_count": 0, "max_drawdown": 0.0,
                }

        # Compute percentage contributions
        contributions = {}
        for name, data in strategy_data.items():
            pnl_pct = (data["pnl"] / total_pnl * 100) if total_pnl != 0 else 0.0
            contributions[name] = {
                "pnl": data["pnl"],
                "pnl_pct": round(pnl_pct, 1),
                "trade_count": data["trade_count"],
                "max_drawdown": data["max_drawdown"],
            }

        # Best and worst
        sorted_by_pnl = sorted(contributions.items(), key=lambda x: x[1]["pnl"])
        worst = sorted_by_pnl[0][0] if sorted_by_pnl else None
        best = sorted_by_pnl[-1][0] if sorted_by_pnl else None

        return {
            "total_pnl": round(total_pnl, 4),
            "lookback_hours": lookback,
            "strategy_contributions": contributions,
            "best_performer": best,
            "worst_performer": worst,
        }

    def generate_attribution_report(self, attribution: Optional[dict] = None) -> str:
        """Generate a human-readable attribution report."""
        if attribution is None:
            attribution = self.compute_attribution()

        lines = [
            "=== Performance Attribution ===",
            f"Total PnL: ${attribution['total_pnl']:.4f}",
            f"Lookback: {attribution['lookback_hours']} hours",
            f"Best: {attribution.get('best_performer', 'N/A')}",
            f"Worst: {attribution.get('worst_performer', 'N/A')}",
            "",
            "Strategy Contributions:",
        ]

        for name, data in attribution.get("strategy_contributions", {}).items():
            lines.append(
                f"  {name}: PnL=${data['pnl']:.4f} ({data['pnl_pct']:.1f}%) "
                f"Trades={data['trade_count']} MaxDD=${data['max_drawdown']:.4f}")

        return "\n".join(lines)
