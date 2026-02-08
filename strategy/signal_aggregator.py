"""Signal aggregator: collects sub-strategy signals for meta-strategy consumption.

Stores the latest signal per strategy plus a ring buffer of recent history.
Used by MetaStrategy subclasses to observe what the sub-strategies are doing.
"""

import json
import logging
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Optional

from core.models import Signal
from core.types import SignalType

logger = logging.getLogger(__name__)

# Direction mapping for consensus calculation
_DIRECTION_MAP = {
    SignalType.ENTER_LONG: "bullish",
    SignalType.EXIT_SHORT: "bullish",
    SignalType.ENTER_SHORT: "bearish",
    SignalType.EXIT_LONG: "bearish",
    SignalType.HOLD: "neutral",
}


class SignalAggregator:
    """Collects signals from sub-strategies for meta-strategy consumption.

    Stores latest signal per strategy + ring buffer of history in Redis.
    """

    def __init__(self, strategy_names: list[str], redis_store=None,
                 timescale=None, max_history: int = 100):
        self.strategy_names = list(strategy_names)
        self.redis_store = redis_store
        self.timescale = timescale
        self.max_history = max_history

        # In-memory latest signal per strategy
        self._latest: dict[str, dict] = {}
        # In-memory ring buffer per strategy
        self._history: dict[str, deque] = {
            name: deque(maxlen=max_history) for name in strategy_names
        }

    def record_signal(self, strategy_name: str, signal: Signal,
                      market_state: Any = None,
                      health_score: float = 50.0) -> None:
        """Record a signal from a sub-strategy.

        Args:
            strategy_name: Name of the strategy that generated the signal.
            signal: The Signal object from the strategy.
            market_state: Optional MarketState for context.
            health_score: Current health score of the strategy (0-100).
        """
        if strategy_name not in self._history:
            self._history[strategy_name] = deque(maxlen=self.max_history)

        direction = _DIRECTION_MAP.get(signal.signal_type, "neutral")
        record = {
            "strategy_name": strategy_name,
            "signal_type": signal.signal_type.value,
            "direction": direction,
            "price": signal.price,
            "size": signal.size,
            "timestamp": signal.timestamp.isoformat() if signal.timestamp else None,
            "health_score": health_score,
            "metadata": signal.metadata or {},
            "recorded_at": time.time(),
        }

        self._latest[strategy_name] = record
        self._history[strategy_name].append(record)

        # Persist to Redis if available
        if self.redis_store:
            try:
                self.redis_store._r.set(
                    f"signal_agg:{strategy_name}:latest",
                    json.dumps(record),
                )
            except Exception:
                logger.debug("Failed to persist signal to Redis for %s", strategy_name)

    def get_latest_signals(self) -> dict[str, dict]:
        """Return the latest signal from each sub-strategy.

        Returns:
            Dict mapping strategy_name -> signal record dict.
        """
        return dict(self._latest)

    def get_signal_history(self, lookback_ticks: int = 20) -> dict[str, list[dict]]:
        """Return recent signal history for each sub-strategy.

        Args:
            lookback_ticks: Number of recent ticks to return per strategy.

        Returns:
            Dict mapping strategy_name -> list of signal records.
        """
        result = {}
        for name, buf in self._history.items():
            items = list(buf)
            result[name] = items[-lookback_ticks:] if len(items) > lookback_ticks else items
        return result

    def get_consensus(self) -> dict:
        """Compute consensus across latest sub-strategy signals.

        Returns:
            Dict with bullish_count, bearish_count, neutral_count,
            weighted_direction (-1 to +1), total_weight.
        """
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        weighted_sum = 0.0
        total_weight = 0.0

        for name, record in self._latest.items():
            direction = record.get("direction", "neutral")
            health = record.get("health_score", 50.0)
            weight = max(health / 100.0, 0.01)  # Minimum weight to avoid zero

            if direction == "bullish":
                bullish_count += 1
                weighted_sum += weight
            elif direction == "bearish":
                bearish_count += 1
                weighted_sum -= weight
            else:
                neutral_count += 1

            total_weight += weight

        weighted_direction = weighted_sum / total_weight if total_weight > 0 else 0.0

        return {
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "neutral_count": neutral_count,
            "weighted_direction": weighted_direction,
            "total_weight": total_weight,
        }
