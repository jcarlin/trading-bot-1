"""Abstract base for meta-strategies that consume sub-strategy signals.

Meta-strategies aggregate signals from multiple sub-strategies and make
higher-level trading decisions based on consensus, voting, or learned models.
"""

import logging
from typing import Any

from core.models import Signal
from core.types import SignalType
from strategy.live_strategy import LiveStrategy, MarketState
from strategy.signal_aggregator import SignalAggregator

logger = logging.getLogger(__name__)


class MetaStrategy(LiveStrategy):
    """Abstract base for meta-strategies consuming sub-strategy signals.

    Subclasses must implement on_tick(), get_metadata(), and optionally
    get_state()/set_state().
    """

    def __init__(self, params: dict[str, Any],
                 signal_aggregator: SignalAggregator):
        super().__init__(params)
        self.aggregator = signal_aggregator

    def _get_sub_signals(self) -> dict[str, dict]:
        """Retrieve the latest signals from all sub-strategies.

        Returns:
            Dict mapping strategy_name -> signal record dict.
        """
        return self.aggregator.get_latest_signals()

    def _compute_weighted_vote(self, signals: dict[str, dict],
                               weights: dict[str, float]) -> tuple[str, float]:
        """Compute a weighted directional vote across sub-strategy signals.

        Args:
            signals: Dict mapping strategy_name -> signal record.
            weights: Dict mapping strategy_name -> weight (0-1).

        Returns:
            Tuple of (direction: "long"/"short"/"neutral", confidence: 0-1).
        """
        bullish_weight = 0.0
        bearish_weight = 0.0
        total_weight = 0.0

        for name, record in signals.items():
            w = weights.get(name, 0.0)
            if w <= 0:
                continue
            direction = record.get("direction", "neutral")
            if direction == "bullish":
                bullish_weight += w
            elif direction == "bearish":
                bearish_weight += w
            total_weight += w

        if total_weight <= 0:
            return "neutral", 0.0

        bullish_pct = bullish_weight / total_weight
        bearish_pct = bearish_weight / total_weight

        if bullish_pct > bearish_pct:
            direction = "long"
            confidence = bullish_pct
        elif bearish_pct > bullish_pct:
            direction = "short"
            confidence = bearish_pct
        else:
            direction = "neutral"
            confidence = 0.0

        return direction, confidence
