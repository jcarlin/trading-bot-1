"""Funding rate arbitrage strategy for Hyperliquid perpetuals."""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from core.models import Signal
from core.types import SignalType
from strategy.live_strategy import LiveStrategy, MarketState

logger = logging.getLogger(__name__)


class FundingRateArbStrategy(LiveStrategy):
    """Collects funding payments by positioning opposite to prevailing rate.

    Hyperliquid settles funding every 1 hour. When rate is negative,
    shorts pay longs. When positive, longs pay shorts.
    """

    def __init__(self, params: dict[str, Any]):
        super().__init__(params)
        self.entry_threshold = float(params.get("entry_threshold", 0.0001))
        self.exit_threshold = float(params.get("exit_threshold", 0.00005))
        self.max_hold_periods = int(params.get("max_hold_periods", 24))
        self.position_size_pct = float(params.get("position_size_pct", 0.03))
        self.min_open_interest = float(params.get("min_open_interest", 1000000))
        self.cooldown_periods = int(params.get("cooldown_periods", 2))

        # Internal state
        self._entry_rate: Optional[float] = None
        self._entry_time: Optional[datetime] = None
        self._periods_held: int = 0
        self._cumulative_funding: float = 0.0
        self._last_exit_time: Optional[datetime] = None
        self._in_position: bool = False
        self._position_side: Optional[str] = None  # "long" or "short"

    def on_tick(self, market_state: MarketState) -> Signal:
        rate = market_state.funding_rate
        mark_price = market_state.mark_price
        timestamp = market_state.timestamp

        if mark_price <= 0:
            return self._hold_signal(mark_price, timestamp)

        position_size = market_state.equity * self.position_size_pct / mark_price

        if self._in_position:
            return self._check_exit(rate, mark_price, timestamp, position_size)

        return self._check_entry(rate, mark_price, timestamp, position_size, market_state)

    def _check_exit(self, rate: float, mark_price: float,
                    timestamp: datetime, position_size: float) -> Signal:
        """Check exit conditions for an open position."""
        self._periods_held += 1
        self._cumulative_funding += rate

        # Exit conditions
        exit_reason = None

        if abs(rate) < self.exit_threshold:
            exit_reason = "funding_normalized"
        elif self._entry_rate is not None:
            # Rate reversal: we entered because rate was negative (long), now it's positive
            if self._position_side == "long" and rate > 0:
                exit_reason = "rate_reversed"
            elif self._position_side == "short" and rate < 0:
                exit_reason = "rate_reversed"

        if self._periods_held >= self.max_hold_periods:
            exit_reason = "max_hold_exceeded"

        if exit_reason:
            signal_type = (SignalType.EXIT_LONG if self._position_side == "long"
                          else SignalType.EXIT_SHORT)
            self._reset_position_state()
            self._last_exit_time = timestamp
            return Signal(
                signal_type=signal_type,
                price=mark_price,
                timestamp=timestamp,
                size=position_size,
                metadata={"exit_reason": exit_reason, "periods_held": self._periods_held},
            )

        return self._hold_signal(mark_price, timestamp)

    def _check_entry(self, rate: float, mark_price: float,
                     timestamp: datetime, position_size: float,
                     market_state: MarketState) -> Signal:
        """Check entry conditions."""
        # Cooldown check
        if self._last_exit_time is not None:
            hours_since_exit = (timestamp - self._last_exit_time).total_seconds() / 3600
            if hours_since_exit < self.cooldown_periods:
                return self._hold_signal(mark_price, timestamp)

        # Open interest check
        if market_state.open_interest < self.min_open_interest:
            return self._hold_signal(mark_price, timestamp)

        # Entry conditions
        if rate < -self.entry_threshold:
            # Negative funding: shorts pay longs -> go long to collect
            self._enter_position("long", rate, timestamp)
            return Signal(
                signal_type=SignalType.ENTER_LONG,
                price=mark_price,
                timestamp=timestamp,
                size=position_size,
                metadata={"funding_rate": rate, "entry_reason": "negative_funding"},
            )

        if rate > self.entry_threshold:
            # Positive funding: longs pay shorts -> go short to collect
            self._enter_position("short", rate, timestamp)
            return Signal(
                signal_type=SignalType.ENTER_SHORT,
                price=mark_price,
                timestamp=timestamp,
                size=position_size,
                metadata={"funding_rate": rate, "entry_reason": "positive_funding"},
            )

        return self._hold_signal(mark_price, timestamp)

    def _enter_position(self, side: str, rate: float, timestamp: datetime) -> None:
        self._in_position = True
        self._position_side = side
        self._entry_rate = rate
        self._entry_time = timestamp
        self._periods_held = 0
        self._cumulative_funding = 0.0

    def _reset_position_state(self) -> None:
        self._in_position = False
        self._position_side = None
        self._entry_rate = None
        self._entry_time = None
        self._periods_held = 0
        self._cumulative_funding = 0.0

    def _hold_signal(self, price: float, timestamp: datetime) -> Signal:
        return Signal(
            signal_type=SignalType.HOLD,
            price=price,
            timestamp=timestamp,
        )

    def get_metadata(self) -> dict:
        return {
            "name": "funding_rate_arb",
            "version": "1.0",
            "params": {
                "entry_threshold": self.entry_threshold,
                "exit_threshold": self.exit_threshold,
                "max_hold_periods": self.max_hold_periods,
                "position_size_pct": self.position_size_pct,
                "min_open_interest": self.min_open_interest,
                "cooldown_periods": self.cooldown_periods,
            },
        }

    def get_state(self) -> dict:
        return {
            "entry_rate": self._entry_rate,
            "entry_time": self._entry_time.isoformat() if self._entry_time else None,
            "periods_held": self._periods_held,
            "cumulative_funding": self._cumulative_funding,
            "last_exit_time": self._last_exit_time.isoformat() if self._last_exit_time else None,
            "in_position": self._in_position,
            "position_side": self._position_side,
        }

    def set_state(self, state: dict) -> None:
        self._entry_rate = state.get("entry_rate")
        entry_time_str = state.get("entry_time")
        self._entry_time = datetime.fromisoformat(entry_time_str) if entry_time_str else None
        self._periods_held = state.get("periods_held", 0)
        self._cumulative_funding = state.get("cumulative_funding", 0.0)
        exit_time_str = state.get("last_exit_time")
        self._last_exit_time = datetime.fromisoformat(exit_time_str) if exit_time_str else None
        self._in_position = state.get("in_position", False)
        self._position_side = state.get("position_side")
