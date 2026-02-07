"""Decision logging for live strategies."""

import logging
from datetime import datetime, timezone

from core.models import Signal
from strategy.live_strategy import MarketState

logger = logging.getLogger(__name__)


class StrategyDecisionLogger:
    """Wraps TimescaleStore.insert_decision() with rich market context."""

    def __init__(self, timescale, strategy_name: str):
        self.timescale = timescale
        self.strategy_name = strategy_name
        self._hold_count = 0

    def log_signal(self, signal: Signal, market_state: MarketState,
                   hypothesis: str, confidence: float = 0.5) -> None:
        """Log a non-HOLD signal with full market context."""
        try:
            self.timescale.insert_decision({
                "time": datetime.now(timezone.utc),
                "decision_type": "signal_generated",
                "strategy": self.strategy_name,
                "context": {
                    "funding_rate": market_state.funding_rate,
                    "mark_price": market_state.mark_price,
                    "open_interest": market_state.open_interest,
                    "equity": market_state.equity,
                    "drawdown_pct": market_state.drawdown_pct,
                    "position": market_state.position,
                    "spread": market_state.spread,
                },
                "hypothesis": hypothesis,
                "action": {
                    "signal_type": signal.signal_type.value,
                    "price": signal.price,
                    "size": signal.size,
                },
                "alternatives": [],
                "confidence": confidence,
                "outcome": {},
            })
        except Exception:
            logger.exception("Failed to log signal decision")

    def log_hold(self, market_state: MarketState, reason: str,
                 every_n: int = 10) -> None:
        """Log HOLD decisions periodically (not every tick)."""
        self._hold_count += 1
        if self._hold_count % every_n != 0:
            return
        try:
            self.timescale.insert_decision({
                "time": datetime.now(timezone.utc),
                "decision_type": "hold",
                "strategy": self.strategy_name,
                "context": {
                    "funding_rate": market_state.funding_rate,
                    "mark_price": market_state.mark_price,
                    "reason": reason,
                    "hold_count": self._hold_count,
                },
                "hypothesis": f"Holding: {reason}",
                "action": {"signal_type": "hold"},
                "alternatives": [],
                "confidence": 0.0,
                "outcome": {},
            })
        except Exception:
            logger.exception("Failed to log hold decision")

    def log_state_change(self, change_type: str, old_state: dict,
                         new_state: dict, reason: str) -> None:
        """Log strategy state transitions."""
        try:
            self.timescale.insert_decision({
                "time": datetime.now(timezone.utc),
                "decision_type": "state_change",
                "strategy": self.strategy_name,
                "context": {
                    "change_type": change_type,
                    "old_state": old_state,
                    "new_state": new_state,
                    "reason": reason,
                },
                "hypothesis": f"State change ({change_type}): {reason}",
                "action": {"new_state": new_state},
                "alternatives": [],
                "confidence": 1.0,
                "outcome": {},
            })
        except Exception:
            logger.exception("Failed to log state change")
