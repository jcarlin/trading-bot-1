"""Voting ensemble strategy: weighted majority voting across sub-strategies.

Aggregates signals from multiple sub-strategies using configurable weighting
modes (equal, health-based, sharpe-based, custom) and enters positions when
agreement exceeds a configurable threshold.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from core.models import Signal
from core.types import SignalType
from strategy.base import BaseStrategy
from strategy.live_strategy import MarketState
from strategy.meta_strategy import MetaStrategy
from strategy.signal_aggregator import SignalAggregator

logger = logging.getLogger(__name__)

# Regime boost mapping: regimes where trending strategies get boosted
_REGIME_BOOST_MAP = {
    "trending_up": {"bullish": 1.3, "bearish": 0.7},
    "trending_down": {"bearish": 1.3, "bullish": 0.7},
    "ranging": {"bullish": 1.0, "bearish": 1.0},
    "volatile": {"bullish": 0.8, "bearish": 0.8},
}


class VotingEnsembleStrategy(MetaStrategy):
    """Weighted majority voting across sub-strategy signals.

    Params:
        min_agreement_pct: Minimum agreement percentage to enter (0-100).
        weighting_mode: "equal", "health", "sharpe", or "custom".
        custom_weights: Dict of strategy_name -> weight (for custom mode).
        position_size_pct: Position size as fraction of equity.
        cooldown_ticks: Ticks to wait after exiting before re-entering.
        regime_boost: Whether to boost weights based on market regime.
    """

    def __init__(self, params: dict[str, Any],
                 signal_aggregator: SignalAggregator):
        super().__init__(params, signal_aggregator)
        self.min_agreement_pct = float(params.get("min_agreement_pct", 60))
        self.weighting_mode = params.get("weighting_mode", "equal")
        self.custom_weights = params.get("custom_weights", {})
        self.position_size_pct = float(params.get("position_size_pct", 0.03))
        self.cooldown_ticks = int(params.get("cooldown_ticks", 5))
        self.regime_boost = bool(params.get("regime_boost", True))

        # State
        self._position_side: Optional[str] = None
        self._entry_price: Optional[float] = None
        self._ticks_since_exit: int = self.cooldown_ticks  # Start ready
        self._tick_count: int = 0

    def on_tick(self, market_state: MarketState) -> Signal:
        """Process signals from sub-strategies and make ensemble decision."""
        price = market_state.mark_price
        timestamp = market_state.timestamp
        self._tick_count += 1

        if price <= 0:
            return self._hold(price, timestamp)

        signals = self._get_sub_signals()
        if not signals:
            return self._hold(price, timestamp)

        # Compute weights
        weights = self._compute_weights(signals)

        # Apply regime boost if enabled
        regime = market_state.metadata.get("regime", "unknown")
        if self.regime_boost and regime != "unknown":
            weights = self._apply_regime_boost(weights, regime, signals)

        # Compute weighted vote
        direction, confidence = self._compute_weighted_vote(signals, weights)
        agreement_pct = confidence * 100.0

        position_size = (market_state.equity * self.position_size_pct / price
                         if price > 0 and market_state.equity > 0 else 0)

        # Track cooldown
        if self._position_side is None:
            self._ticks_since_exit += 1

        # Update Prometheus metrics
        try:
            from monitoring.metrics import (
                meta_strategy_agreement_pct,
                meta_strategy_confidence,
                meta_strategy_sub_signal_count,
            )
            meta_strategy_agreement_pct.labels(strategy_name="voting_ensemble").set(agreement_pct)
            meta_strategy_confidence.labels(strategy_name="voting_ensemble").set(confidence)
            meta_strategy_sub_signal_count.labels(strategy_name="voting_ensemble").set(len(signals))
        except Exception:
            pass

        # Check exit first if in position
        if self._position_side is not None:
            return self._check_exit(direction, confidence, price, timestamp, position_size)

        # Check entry
        if self._ticks_since_exit < self.cooldown_ticks:
            return self._hold(price, timestamp)

        if agreement_pct >= self.min_agreement_pct and direction != "neutral":
            return self._enter(direction, price, timestamp, position_size,
                               agreement_pct, confidence)

        return self._hold(price, timestamp)

    def _compute_weights(self, signals: dict[str, dict]) -> dict[str, float]:
        """Compute strategy weights based on weighting mode.

        Args:
            signals: Latest signals from sub-strategies.

        Returns:
            Dict mapping strategy_name -> weight.
        """
        if self.weighting_mode == "custom":
            return {name: self.custom_weights.get(name, 1.0)
                    for name in signals}

        if self.weighting_mode == "health":
            return {name: max(record.get("health_score", 50.0) / 100.0, 0.01)
                    for name, record in signals.items()}

        if self.weighting_mode == "sharpe":
            weights = {}
            for name, record in signals.items():
                sharpe = record.get("metadata", {}).get("sharpe", 1.0)
                weights[name] = max(float(sharpe), 0.01)
            return weights

        # Default: equal weighting
        return {name: 1.0 for name in signals}

    def _apply_regime_boost(self, weights: dict[str, float], regime: str,
                            signals: dict[str, dict]) -> dict[str, float]:
        """Apply regime-based boost/penalty to weights.

        Args:
            weights: Current strategy weights.
            regime: Current market regime string.
            signals: Latest signals for direction context.

        Returns:
            Adjusted weights dict.
        """
        boost_map = _REGIME_BOOST_MAP.get(regime)
        if not boost_map:
            return weights

        adjusted = {}
        for name, w in weights.items():
            record = signals.get(name, {})
            direction = record.get("direction", "neutral")
            multiplier = boost_map.get(direction, 1.0)
            adjusted[name] = w * multiplier

        return adjusted

    def _enter(self, direction: str, price: float, timestamp: datetime,
               size: float, agreement_pct: float,
               confidence: float) -> Signal:
        """Generate entry signal."""
        if direction == "long":
            self._position_side = "long"
            self._entry_price = price
            return Signal(
                signal_type=SignalType.ENTER_LONG,
                price=price,
                timestamp=timestamp,
                size=size,
                metadata={
                    "strategy": "voting_ensemble",
                    "agreement_pct": round(agreement_pct, 1),
                    "confidence": round(confidence, 4),
                    "weighting_mode": self.weighting_mode,
                },
            )
        elif direction == "short":
            self._position_side = "short"
            self._entry_price = price
            return Signal(
                signal_type=SignalType.ENTER_SHORT,
                price=price,
                timestamp=timestamp,
                size=size,
                metadata={
                    "strategy": "voting_ensemble",
                    "agreement_pct": round(agreement_pct, 1),
                    "confidence": round(confidence, 4),
                    "weighting_mode": self.weighting_mode,
                },
            )
        return self._hold(price, timestamp)

    def _check_exit(self, direction: str, confidence: float, price: float,
                    timestamp: datetime, size: float) -> Signal:
        """Check for exit conditions."""
        should_exit = False
        exit_reason = ""

        # Exit if consensus flips
        if self._position_side == "long" and direction == "short":
            should_exit = True
            exit_reason = "consensus_flip"
        elif self._position_side == "short" and direction == "long":
            should_exit = True
            exit_reason = "consensus_flip"
        # Exit if consensus drops to neutral with low confidence
        elif direction == "neutral" and confidence < 0.3:
            should_exit = True
            exit_reason = "consensus_neutral"

        if should_exit:
            if self._position_side == "long":
                signal_type = SignalType.EXIT_LONG
            else:
                signal_type = SignalType.EXIT_SHORT

            self._position_side = None
            self._entry_price = None
            self._ticks_since_exit = 0

            return Signal(
                signal_type=signal_type,
                price=price,
                timestamp=timestamp,
                size=size,
                metadata={"exit_reason": exit_reason, "strategy": "voting_ensemble"},
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
            "name": "voting_ensemble",
            "version": "1.0",
            "category": "voting_ensemble",
            "timeframes": ["1h"],
            "description": "Weighted majority voting ensemble across sub-strategies",
            "params": {
                "min_agreement_pct": self.min_agreement_pct,
                "weighting_mode": self.weighting_mode,
                "position_size_pct": self.position_size_pct,
                "cooldown_ticks": self.cooldown_ticks,
                "regime_boost": self.regime_boost,
            },
        }

    def get_state(self) -> dict:
        return {
            "position_side": self._position_side,
            "entry_price": self._entry_price,
            "ticks_since_exit": self._ticks_since_exit,
            "tick_count": self._tick_count,
        }

    def set_state(self, state: dict) -> None:
        self._position_side = state.get("position_side")
        self._entry_price = state.get("entry_price")
        self._ticks_since_exit = state.get("ticks_since_exit", self.cooldown_ticks)
        self._tick_count = state.get("tick_count", 0)


# ---------------------------------------------------------------------------
# Backtest adapter
# ---------------------------------------------------------------------------


class VotingEnsembleBacktestAdapter(BaseStrategy):
    """Backtest adapter for VotingEnsembleStrategy.

    In backtest mode, the adapter simulates sub-strategy signals using simple
    indicator-based heuristics (EMA crossover, RSI, BB) to approximate what
    the live sub-strategies would produce.
    """

    def __init__(self, params: dict[str, Any]):
        super().__init__(params)
        self.min_agreement_pct = float(params.get("min_agreement_pct", 60))
        self.position_size_pct = float(params.get("position_size_pct", 0.03))
        self.cooldown_ticks = int(params.get("cooldown_ticks", 5))

        # State
        self._position_side: Optional[str] = None
        self._ticks_since_exit: int = self.cooldown_ticks

    def setup(self, df: pd.DataFrame) -> None:
        """Compute proxy indicators for backtest sub-strategy signals."""
        closes = df["close"]

        # Proxy 1: EMA crossover (momentum proxy)
        ema_fast = closes.ewm(span=12, adjust=False).mean()
        ema_slow = closes.ewm(span=26, adjust=False).mean()
        self.indicators["ema_signal"] = (ema_fast > ema_slow).astype(float)

        # Proxy 2: RSI (mean-reversion proxy)
        delta = closes.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss.replace(0, float("inf"))
        rsi = 100.0 - (100.0 / (1.0 + rs))
        self.indicators["rsi"] = rsi
        # RSI signal: > 50 = bullish, < 50 = bearish
        self.indicators["rsi_signal"] = (rsi > 50).astype(float)

        # Proxy 3: BB position (vol proxy)
        bb_mid = closes.rolling(20).mean()
        bb_std = closes.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        # Above mid = bullish, below = bearish
        self.indicators["bb_signal"] = (closes > bb_mid).astype(float)

    def generate_signal(self, index: int, df: pd.DataFrame) -> Signal:
        """Generate signal based on proxy sub-strategy consensus."""
        if index < 26:  # Need enough warmup
            return self.hold_signal(df, index)

        price = float(df["close"].iloc[index])
        timestamp = df.index[index]

        # Count bullish proxy signals
        ema_bull = self.indicators["ema_signal"].iloc[index]
        rsi_bull = self.indicators["rsi_signal"].iloc[index]
        bb_bull = self.indicators["bb_signal"].iloc[index]

        bullish = int(ema_bull) + int(rsi_bull) + int(bb_bull)
        bearish = 3 - bullish
        agreement_pct = max(bullish, bearish) / 3.0 * 100.0

        if self._position_side is None:
            self._ticks_since_exit += 1

        size = 10000 * self.position_size_pct / price if price > 0 else 0

        # Check exit
        if self._position_side is not None:
            should_exit = False
            if self._position_side == "long" and bearish > bullish:
                should_exit = True
            elif self._position_side == "short" and bullish > bearish:
                should_exit = True

            if should_exit:
                st = (SignalType.EXIT_LONG if self._position_side == "long"
                      else SignalType.EXIT_SHORT)
                self._position_side = None
                self._ticks_since_exit = 0
                return Signal(signal_type=st, price=price, timestamp=timestamp,
                              size=size, metadata={"exit_reason": "consensus_flip"})

            return self.hold_signal(df, index)

        # Check entry
        if self._ticks_since_exit < self.cooldown_ticks:
            return self.hold_signal(df, index)

        if agreement_pct >= self.min_agreement_pct:
            if bullish > bearish:
                self._position_side = "long"
                return Signal(signal_type=SignalType.ENTER_LONG, price=price,
                              timestamp=timestamp, size=size,
                              metadata={"agreement_pct": agreement_pct})
            elif bearish > bullish:
                self._position_side = "short"
                return Signal(signal_type=SignalType.ENTER_SHORT, price=price,
                              timestamp=timestamp, size=size,
                              metadata={"agreement_pct": agreement_pct})

        return self.hold_signal(df, index)
