"""Signal quality assessment for live trading strategies."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class SignalQualityAssessor:
    """Measures how well strategy signals predict price moves.

    Evaluates signal accuracy by comparing entry signals to subsequent
    price action using candle data.
    """

    def __init__(self, timescale, strategy_name: str, symbol: str = "BTC/USDC"):
        self.timescale = timescale
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.lookahead_hours = 4  # hours to look ahead for price evaluation

    def assess(self, window_hours: int = 24) -> dict:
        """Assess signal quality over a time window.

        Args:
            window_hours: Number of hours to look back for fills.

        Returns:
            dict with signal quality metrics.
        """
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=window_hours)

        try:
            fills = self.timescale.query_fills_by_strategy(
                self.strategy_name, start, end)
        except Exception:
            logger.exception("Failed to query fills for signal quality")
            fills = []

        if not fills:
            return self._empty_assessment()

        # Get candles for price evaluation (extend beyond window for lookahead)
        candle_end = end + timedelta(hours=self.lookahead_hours)
        try:
            candles = self.timescale.query_candles(
                self.symbol, "1h", start, candle_end)
        except Exception:
            logger.exception("Failed to query candles for signal quality")
            candles = []

        if not candles:
            return self._empty_assessment()

        # Build price lookup from candles
        price_series = {c["time"]: float(c["close"]) for c in candles}
        sorted_times = sorted(price_series.keys())

        # Analyze each entry fill
        accurate_count = 0
        favorable_moves = []
        adverse_moves = []
        timing_scores = []
        total_entries = 0

        for fill in fills:
            side = fill.get("side", "")
            fill_price = float(fill.get("price", 0))
            fill_time = fill.get("time")

            if fill_price <= 0 or fill_time is None:
                continue

            # Only evaluate entry fills (buy or sell, not exits tracked by PnL)
            # Simple heuristic: if closed_pnl is 0 or near-zero, it's likely an entry
            closed_pnl = float(fill.get("closed_pnl", 0))
            if abs(closed_pnl) > 0.01:
                continue  # This is an exit fill

            total_entries += 1

            # Find prices after the fill
            future_prices = self._get_future_prices(
                fill_time, sorted_times, price_series, self.lookahead_hours)

            if not future_prices:
                continue

            # Calculate favorable/adverse moves
            is_long = side.lower() in ("buy", "long")
            max_price = max(future_prices)
            min_price = min(future_prices)

            if is_long:
                favorable_move = (max_price - fill_price) / fill_price * 100
                adverse_move = (fill_price - min_price) / fill_price * 100
                # Accurate if price went up
                accurate = max_price > fill_price
            else:
                favorable_move = (fill_price - min_price) / fill_price * 100
                adverse_move = (max_price - fill_price) / fill_price * 100
                accurate = min_price < fill_price

            if accurate:
                accurate_count += 1

            favorable_moves.append(favorable_move)
            adverse_moves.append(adverse_move)

            # Timing score: how close to the best possible entry in the window
            if is_long:
                best_entry = min_price
                timing = max(0, (1 - (fill_price - best_entry) /
                             (max_price - best_entry + 1e-10))) * 100
            else:
                best_entry = max_price
                timing = max(0, (1 - (best_entry - fill_price) /
                             (best_entry - min_price + 1e-10))) * 100
            timing_scores.append(timing)

        sample_size = len(favorable_moves)
        if sample_size == 0:
            return self._empty_assessment()

        signal_accuracy = (accurate_count / sample_size * 100) if sample_size > 0 else 0.0

        return {
            "signal_accuracy_pct": round(signal_accuracy, 2),
            "avg_favorable_move_pct": round(sum(favorable_moves) / sample_size, 4),
            "avg_adverse_move_pct": round(sum(adverse_moves) / sample_size, 4),
            "timing_score": round(sum(timing_scores) / sample_size, 2),
            "false_positive_rate": round(100 - signal_accuracy, 2),
            "sample_size": sample_size,
            "total_entries": total_entries,
        }

    def _get_future_prices(self, fill_time, sorted_times: list,
                           price_series: dict, hours: int) -> list[float]:
        """Get prices in the hours after a fill."""
        cutoff = fill_time + timedelta(hours=hours)
        prices = []
        for t in sorted_times:
            if t > fill_time and t <= cutoff:
                prices.append(price_series[t])
        return prices

    @staticmethod
    def _empty_assessment() -> dict:
        return {
            "signal_accuracy_pct": 0.0,
            "avg_favorable_move_pct": 0.0,
            "avg_adverse_move_pct": 0.0,
            "timing_score": 0.0,
            "false_positive_rate": 0.0,
            "sample_size": 0,
            "total_entries": 0,
        }
