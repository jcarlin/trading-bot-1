"""Order flow microstructure analysis.

Tracks individual trades in a rolling window, computes buy/sell volume
imbalance, cumulative volume delta (CVD), detects large trades, and
calculates VWAP.
"""

import statistics
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Optional


class OrderFlowAnalyzer:
    """Analyzes order flow microstructure from trade data."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.window_seconds = config.get("window_seconds", 300)
        self.large_trade_std_mult = config.get("large_trade_std_mult", 2.0)
        self.max_trades = config.get("max_trades", 10000)
        self._trades: deque = deque()

    def add_trade(self, trade: dict) -> None:
        """Add a single trade.

        Args:
            trade: {timestamp: datetime, price: float, size: float,
                    side: str}  side is "buy" or "sell"
        """
        self._trades.append(trade)
        self._prune()

    def add_trades_batch(self, trades: list[dict]) -> None:
        """Add multiple trades."""
        for trade in trades:
            self._trades.append(trade)
        self._prune()

    def _prune(self) -> None:
        """Remove old trades and cap at max_trades."""
        cutoff = datetime.now(timezone.utc) - timedelta(
            seconds=self.window_seconds)
        while self._trades and self._trades[0].get("timestamp", cutoff) < cutoff:
            self._trades.popleft()
        while len(self._trades) > self.max_trades:
            self._trades.popleft()

    def _get_window_trades(self, window_seconds: int = None) -> list[dict]:
        """Get trades within the specified window."""
        window = window_seconds or self.window_seconds
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=window)
        return [
            t for t in self._trades
            if t.get("timestamp", cutoff) >= cutoff
        ]

    def get_imbalance(self, window_seconds: int = None) -> dict:
        """Compute buy/sell volume imbalance.

        Returns:
            {buy_vol, sell_vol, imbalance_ratio, imbalance_pct}
        """
        trades = self._get_window_trades(window_seconds)

        buy_vol = 0.0
        sell_vol = 0.0

        for t in trades:
            side = t.get("side", "").lower()
            size = float(t.get("size", 0))
            if side == "buy":
                buy_vol += size
            elif side == "sell":
                sell_vol += size

        total = buy_vol + sell_vol
        if total > 0:
            imbalance_ratio = buy_vol / total
            imbalance_pct = (buy_vol - sell_vol) / total * 100
        else:
            imbalance_ratio = 0.5
            imbalance_pct = 0.0

        return {
            "buy_vol": buy_vol,
            "sell_vol": sell_vol,
            "imbalance_ratio": imbalance_ratio,
            "imbalance_pct": imbalance_pct,
        }

    def get_cumulative_delta(self, window_seconds: int = None) -> dict:
        """Compute cumulative volume delta.

        Returns:
            {cvd, cvd_normalized, trend}
        """
        trades = self._get_window_trades(window_seconds)

        cvd = 0.0
        total_volume = 0.0

        for t in trades:
            side = t.get("side", "").lower()
            size = float(t.get("size", 0))
            total_volume += size
            if side == "buy":
                cvd += size
            elif side == "sell":
                cvd -= size

        cvd_normalized = cvd / total_volume if total_volume > 0 else 0.0

        if cvd > 0:
            trend = "bullish"
        elif cvd < 0:
            trend = "bearish"
        else:
            trend = "neutral"

        return {
            "cvd": cvd,
            "cvd_normalized": cvd_normalized,
            "trend": trend,
        }

    def detect_large_trades(self, window_seconds: int = None) -> list[dict]:
        """Detect trades where size > mean + large_trade_std_mult * std.

        Returns:
            List of trade dicts that qualify as large.
        """
        trades = self._get_window_trades(window_seconds)
        if len(trades) < 2:
            return []

        sizes = [float(t.get("size", 0)) for t in trades]
        mean_size = statistics.mean(sizes)
        std_size = statistics.stdev(sizes)

        threshold = mean_size + self.large_trade_std_mult * std_size

        return [t for t in trades if float(t.get("size", 0)) > threshold]

    def get_vwap(self, window_seconds: int = None) -> Optional[float]:
        """Compute volume-weighted average price.

        Returns:
            VWAP float or None if no trades.
        """
        trades = self._get_window_trades(window_seconds)
        if not trades:
            return None

        total_pv = 0.0
        total_vol = 0.0

        for t in trades:
            price = float(t.get("price", 0))
            size = float(t.get("size", 0))
            total_pv += price * size
            total_vol += size

        if total_vol <= 0:
            return None

        return total_pv / total_vol

    def to_market_state_metadata(self) -> dict:
        """Return metadata dict suitable for MarketState.metadata."""
        imbalance = self.get_imbalance()
        cvd = self.get_cumulative_delta()
        large_trades = self.detect_large_trades()
        vwap = self.get_vwap()

        return {
            "order_flow": {
                "buy_vol": imbalance["buy_vol"],
                "sell_vol": imbalance["sell_vol"],
                "imbalance_ratio": imbalance["imbalance_ratio"],
                "imbalance_pct": imbalance["imbalance_pct"],
                "cvd": cvd["cvd"],
                "cvd_normalized": cvd["cvd_normalized"],
                "cvd_trend": cvd["trend"],
                "large_trade_count": len(large_trades),
                "vwap": vwap,
            }
        }
