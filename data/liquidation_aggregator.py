"""Liquidation cascade detection and aggregation.

Tracks liquidation events in a rolling window, computes side imbalance,
and detects cascading liquidation events that may signal forced selling
or buying pressure.
"""

from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Optional


class LiquidationAggregator:
    """Aggregates liquidation events and detects cascade patterns."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.window_seconds = config.get("window_seconds", 300)
        self.cascade_threshold = config.get("cascade_threshold", 10)
        self.cascade_window_seconds = config.get("cascade_window_seconds", 60)
        self._events: deque = deque()

    def add_liquidation(self, event: dict) -> None:
        """Add a single liquidation event.

        Args:
            event: {timestamp: datetime, side: str, size: float,
                    price: float, symbol: str, exchange: str}
        """
        self._events.append(event)
        self._prune()

    def add_liquidations_batch(self, events: list[dict]) -> None:
        """Add multiple liquidation events."""
        for event in events:
            self._events.append(event)
        self._prune()

    def _prune(self) -> None:
        """Remove events older than window_seconds from now."""
        cutoff = datetime.now(timezone.utc) - timedelta(
            seconds=self.window_seconds)
        while self._events and self._events[0].get("timestamp", cutoff) < cutoff:
            self._events.popleft()

    def get_imbalance(self, window_seconds: int = None) -> dict:
        """Compute liquidation volume imbalance.

        Returns:
            {long_vol, short_vol, imbalance_ratio, dominant_side}
        """
        window = window_seconds or self.window_seconds
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=window)

        long_vol = 0.0
        short_vol = 0.0

        for event in self._events:
            ts = event.get("timestamp")
            if ts is not None and ts < cutoff:
                continue
            side = event.get("side", "").lower()
            size = float(event.get("size", 0))
            if side == "long":
                long_vol += size
            elif side == "short":
                short_vol += size

        total = long_vol + short_vol
        if total > 0:
            imbalance_ratio = max(long_vol, short_vol) / total
        else:
            imbalance_ratio = 0.5

        dominant_side = "long" if long_vol > short_vol else "short"
        if long_vol == short_vol:
            dominant_side = "neutral"

        return {
            "long_vol": long_vol,
            "short_vol": short_vol,
            "imbalance_ratio": imbalance_ratio,
            "dominant_side": dominant_side,
        }

    def detect_cascade(self) -> Optional[dict]:
        """Detect if a liquidation cascade is occurring.

        A cascade is defined as cascade_threshold or more events within
        cascade_window_seconds.

        Returns:
            {is_cascade, count, total_volume, dominant_side} or None
            if no events.
        """
        if not self._events:
            return None

        cutoff = datetime.now(timezone.utc) - timedelta(
            seconds=self.cascade_window_seconds)

        cascade_events = []
        for event in self._events:
            ts = event.get("timestamp")
            if ts is not None and ts >= cutoff:
                cascade_events.append(event)

        count = len(cascade_events)
        total_volume = sum(float(e.get("size", 0)) for e in cascade_events)

        long_vol = sum(
            float(e.get("size", 0)) for e in cascade_events
            if e.get("side", "").lower() == "long"
        )
        short_vol = sum(
            float(e.get("size", 0)) for e in cascade_events
            if e.get("side", "").lower() == "short"
        )

        dominant_side = "long" if long_vol > short_vol else "short"
        if long_vol == short_vol:
            dominant_side = "neutral"

        return {
            "is_cascade": count >= self.cascade_threshold,
            "count": count,
            "total_volume": total_volume,
            "dominant_side": dominant_side,
        }

    def get_summary(self, window_seconds: int = None) -> dict:
        """Get combined imbalance and cascade summary.

        Returns:
            {imbalance: {...}, cascade: {...}, event_count: int}
        """
        imbalance = self.get_imbalance(window_seconds)
        cascade = self.detect_cascade()
        return {
            "imbalance": imbalance,
            "cascade": cascade,
            "event_count": len(self._events),
        }

    def to_market_state_metadata(self) -> dict:
        """Return metadata dict suitable for MarketState.metadata."""
        summary = self.get_summary()
        imbalance = summary.get("imbalance", {})
        cascade = summary.get("cascade") or {}
        return {
            "liquidation_long_vol": imbalance.get("long_vol", 0.0),
            "liquidation_short_vol": imbalance.get("short_vol", 0.0),
            "liquidation_imbalance_ratio": imbalance.get(
                "imbalance_ratio", 0.5),
            "liquidation_dominant_side": imbalance.get(
                "dominant_side", "neutral"),
            "liquidation_cascade": cascade.get("is_cascade", False),
            "liquidation_cascade_count": cascade.get("count", 0),
            "liquidation_cascade_volume": cascade.get("total_volume", 0.0),
            "liquidation_event_count": summary.get("event_count", 0),
        }
