"""Smart money strategy: follows high-confidence wallet signals.

Entry: When N wallets (weighted by score) agree on direction within time window.
Exit: Consensus reversal, time limit, or drawdown.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pandas as pd

from core.models import Signal
from core.types import SignalType
from strategy.base import BaseStrategy
from strategy.live_strategy import LiveStrategy, MarketState

logger = logging.getLogger(__name__)


class SmartMoneyStrategy(LiveStrategy):
    """Follows high-confidence wallet signals as trading signals.

    Entry logic:
        Aggregates recent wallet signals within a time window. If at least
        ``min_wallets_agree`` wallets (weighted by score) agree on a direction,
        a trade signal is generated.

    Exit logic:
        - Consensus reversal (wallets flip direction).
        - Max hold period exceeded.
        - Drawdown from entry exceeds threshold.

    Params:
        min_wallets_agree: Minimum wallet count agreeing for entry (default 2).
        agreement_window_s: Seconds to look back for signals (default 3600).
        min_wallet_score: Minimum wallet score for inclusion (default 70).
        unusual_size_boost: Extra confidence for unusual-size signals (default 0.2).
        max_hold_periods: Max 1h periods to hold (default 48).
        position_size_pct: Fraction of equity per trade (default 0.015).
        drawdown_exit_pct: Max drawdown % before exit (default 3.0).
    """

    def __init__(self, params: dict[str, Any], wallet_monitor=None):
        super().__init__(params)
        self.wallet_monitor = wallet_monitor

        self.min_wallets_agree = int(params.get("min_wallets_agree", 2))
        self.agreement_window_s = int(params.get("agreement_window_s", 3600))
        self.min_wallet_score = float(params.get("min_wallet_score", 70))
        self.unusual_size_boost = float(params.get("unusual_size_boost", 0.2))
        self.max_hold_periods = int(params.get("max_hold_periods", 48))
        self.position_size_pct = float(params.get("position_size_pct", 0.015))
        self.drawdown_exit_pct = float(params.get("drawdown_exit_pct", 3.0))

        # Position state
        self._position_side: Optional[str] = None
        self._entry_price: Optional[float] = None
        self._entry_time: Optional[datetime] = None
        self._hold_periods: int = 0

    def on_tick(self, market_state: MarketState) -> Signal:
        price = market_state.mark_price
        timestamp = market_state.timestamp

        if price <= 0:
            return self._hold(price, timestamp)

        # Check exit first
        if self._position_side is not None:
            return self._check_exit(price, timestamp)

        # Get wallet signals and compute aggregate direction
        signals = self._get_wallet_signals()
        if not signals:
            return self._hold(price, timestamp)

        direction, confidence = self._compute_aggregate_direction(signals)

        if direction is None:
            return self._hold(price, timestamp)

        position_size = (market_state.equity * self.position_size_pct / price
                         if price > 0 else 0)

        if direction == "long":
            self._position_side = "long"
            self._entry_price = price
            self._entry_time = timestamp
            self._hold_periods = 0
            return Signal(
                signal_type=SignalType.ENTER_LONG,
                price=price,
                timestamp=timestamp,
                size=position_size,
                metadata={
                    "direction": "long",
                    "confidence": round(confidence, 3),
                    "wallet_signals": len(signals),
                    "entry_reason": "smart_money_consensus_long",
                },
            )
        elif direction == "short":
            self._position_side = "short"
            self._entry_price = price
            self._entry_time = timestamp
            self._hold_periods = 0
            return Signal(
                signal_type=SignalType.ENTER_SHORT,
                price=price,
                timestamp=timestamp,
                size=position_size,
                metadata={
                    "direction": "short",
                    "confidence": round(confidence, 3),
                    "wallet_signals": len(signals),
                    "entry_reason": "smart_money_consensus_short",
                },
            )

        return self._hold(price, timestamp)

    def _get_wallet_signals(self) -> list[dict]:
        """Get recent wallet signals from monitor or metadata."""
        if self.wallet_monitor:
            try:
                all_signals = self.wallet_monitor.get_recent_signals(limit=100)
                now = datetime.now(timezone.utc)
                cutoff = now - timedelta(seconds=self.agreement_window_s)

                filtered = []
                for sig in all_signals:
                    ts_str = sig.get("timestamp", "")
                    try:
                        ts = datetime.fromisoformat(ts_str)
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                    except (ValueError, TypeError):
                        continue

                    if ts >= cutoff and sig.get("wallet_score", 0) >= self.min_wallet_score:
                        # Only include actionable signal types
                        if sig.get("signal_type") in ("new_position", "size_increase"):
                            filtered.append(sig)
                return filtered
            except Exception:
                logger.debug("Failed to get wallet signals from monitor")
                return []
        return []

    def _compute_aggregate_direction(self, signals: list[dict]) -> tuple[Optional[str], float]:
        """Compute aggregate direction weighted by wallet score.

        Args:
            signals: List of wallet signal dicts.

        Returns:
            (direction, confidence) or (None, 0.0) if no agreement.
        """
        if not signals:
            return None, 0.0

        # Deduplicate by wallet address (keep latest per wallet)
        wallet_latest: dict[str, dict] = {}
        for sig in signals:
            addr = sig.get("wallet_address", "")
            existing = wallet_latest.get(addr)
            if existing is None:
                wallet_latest[addr] = sig
            else:
                try:
                    existing_ts = datetime.fromisoformat(existing.get("timestamp", ""))
                    new_ts = datetime.fromisoformat(sig.get("timestamp", ""))
                    if new_ts > existing_ts:
                        wallet_latest[addr] = sig
                except (ValueError, TypeError):
                    wallet_latest[addr] = sig

        unique_signals = list(wallet_latest.values())

        long_weight = 0.0
        short_weight = 0.0
        long_count = 0
        short_count = 0

        for sig in unique_signals:
            score = sig.get("wallet_score", 0) / 100.0
            weight = score
            if sig.get("is_unusual", False):
                weight = min(1.0, weight + self.unusual_size_boost)

            direction = sig.get("direction", "")
            if direction == "long":
                long_weight += weight
                long_count += 1
            elif direction == "short":
                short_weight += weight
                short_count += 1

        # Check agreement threshold
        if long_count >= self.min_wallets_agree and long_weight > short_weight:
            total_weight = long_weight + short_weight
            confidence = long_weight / total_weight if total_weight > 0 else 0
            return "long", confidence

        if short_count >= self.min_wallets_agree and short_weight > long_weight:
            total_weight = long_weight + short_weight
            confidence = short_weight / total_weight if total_weight > 0 else 0
            return "short", confidence

        return None, 0.0

    def _check_exit(self, price: float, timestamp: datetime) -> Signal:
        """Check exit conditions for open position."""
        self._hold_periods += 1
        should_exit = False
        exit_reason = ""

        # Max hold period
        if self._hold_periods >= self.max_hold_periods:
            should_exit = True
            exit_reason = "max_hold_period"

        # Drawdown exit
        if self._entry_price and self._entry_price > 0:
            if self._position_side == "long":
                dd_pct = (self._entry_price - price) / self._entry_price * 100
            else:
                dd_pct = (price - self._entry_price) / self._entry_price * 100

            if dd_pct >= self.drawdown_exit_pct:
                should_exit = True
                exit_reason = "drawdown_exit"

        # Consensus reversal
        if not should_exit:
            signals = self._get_wallet_signals()
            if signals:
                direction, _ = self._compute_aggregate_direction(signals)
                if direction is not None and direction != self._position_side:
                    should_exit = True
                    exit_reason = "consensus_reversal"

        if should_exit:
            if self._position_side == "long":
                signal_type = SignalType.EXIT_LONG
            else:
                signal_type = SignalType.EXIT_SHORT

            self._position_side = None
            self._entry_price = None
            self._entry_time = None
            self._hold_periods = 0

            return Signal(
                signal_type=signal_type,
                price=price,
                timestamp=timestamp,
                metadata={"exit_reason": exit_reason},
            )

        return self._hold(price, timestamp)

    def _hold(self, price: float, timestamp: datetime) -> Signal:
        return Signal(
            signal_type=SignalType.HOLD,
            price=price,
            timestamp=timestamp,
        )

    def get_metadata(self) -> dict:
        return {
            "name": "smart_money",
            "version": "1.0",
            "category": "smart_money",
            "timeframes": ["1h"],
            "description": "Smart money strategy following high-confidence wallet signals",
            "params": {
                "min_wallets_agree": self.min_wallets_agree,
                "agreement_window_s": self.agreement_window_s,
                "min_wallet_score": self.min_wallet_score,
                "unusual_size_boost": self.unusual_size_boost,
                "max_hold_periods": self.max_hold_periods,
                "position_size_pct": self.position_size_pct,
                "drawdown_exit_pct": self.drawdown_exit_pct,
            },
        }

    def get_state(self) -> dict:
        return {
            "position_side": self._position_side,
            "entry_price": self._entry_price,
            "entry_time": self._entry_time.isoformat() if self._entry_time else None,
            "hold_periods": self._hold_periods,
        }

    def set_state(self, state: dict) -> None:
        self._position_side = state.get("position_side")
        self._entry_price = state.get("entry_price")
        entry_time_str = state.get("entry_time")
        if entry_time_str:
            try:
                self._entry_time = datetime.fromisoformat(entry_time_str)
            except (ValueError, TypeError):
                self._entry_time = None
        else:
            self._entry_time = None
        self._hold_periods = state.get("hold_periods", 0)


# ---------------------------------------------------------------------------
# Backtest adapter
# ---------------------------------------------------------------------------


class SmartMoneyBacktestAdapter(BaseStrategy):
    """Backtest adapter for SmartMoneyStrategy.

    In backtest mode, smart money signals are simulated from the
    wallet_signals parameter (list of dicts with bar indices and directions).
    This enables walk-forward validation of the strategy logic.
    """

    def __init__(self, params: dict[str, Any]):
        super().__init__(params)
        self.live_strategy = SmartMoneyStrategy(params)
        # Simulated wallet signals for backtest: list of
        # {"index": int, "direction": str, "wallet_score": float, "is_unusual": bool}
        self._sim_signals: list[dict] = params.get("simulated_signals", [])

    def setup(self, df: pd.DataFrame) -> None:
        """No pre-computation needed."""
        pass

    def generate_signal(self, index: int, df: pd.DataFrame) -> Signal:
        """Convert bar-based call to event-based tick."""
        row = df.iloc[index]

        market_state = MarketState(
            mark_price=float(row["close"]),
            mid_price=float(row["close"]),
            bid=float(row["close"]) - 0.5,
            ask=float(row["close"]) + 0.5,
            funding_rate=0.0,
            premium=0.0,
            open_interest=0.0,
            equity=10000.0,
            cash=5000.0,
            timestamp=df.index[index] if hasattr(df.index[index], 'tzinfo') else datetime.now(timezone.utc),
        )

        signal = self.live_strategy.on_tick(market_state)
        signal.timestamp = df.index[index] if hasattr(df.index, '__getitem__') else signal.timestamp
        return signal
