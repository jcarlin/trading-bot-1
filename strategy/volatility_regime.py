"""Volatility regime strategy: trades vol expansion/contraction.

Uses ATR ratio (fast/slow) for regime detection, Bollinger Band width
for vol classification, Keltner Channel breakouts for expansion entries,
and RSI for direction confirmation.
"""

import logging
import math
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from core.models import Signal
from core.types import SignalType
from strategy.live_strategy import LiveStrategy, MarketState

logger = logging.getLogger(__name__)

# Volatility regime states
LOW_VOL = "low_vol"
NORMAL = "normal"
HIGH_VOL = "high_vol"
EXPANDING = "expanding"
CONTRACTING = "contracting"


class VolatilityRegimeStrategy(LiveStrategy):
    """Volatility regime strategy: trades vol expansion/contraction.

    Uses ATR ratio (fast/slow) for regime detection, Bollinger Band width
    for vol classification, Keltner Channel breakouts for expansion entries,
    and RSI for direction confirmation.

    Regime states: LOW_VOL, NORMAL, HIGH_VOL, EXPANDING, CONTRACTING
    """

    def __init__(self, params: dict[str, Any]):
        super().__init__(params)
        self.atr_fast = int(params.get("atr_fast", 7))
        self.atr_slow = int(params.get("atr_slow", 21))
        self.bb_period = int(params.get("bb_period", 20))
        self.bb_std = float(params.get("bb_std", 2.0))
        self.kc_period = int(params.get("kc_period", 20))
        self.kc_atr_mult = float(params.get("kc_atr_mult", 1.5))
        self.expansion_threshold = float(params.get("expansion_threshold", 1.5))
        self.contraction_threshold = float(params.get("contraction_threshold", 0.7))
        self.rsi_period = int(params.get("rsi_period", 14))
        self.position_size_pct = float(params.get("position_size_pct", 0.02))

        # State
        self._position_side: Optional[str] = None
        self._entry_price: Optional[float] = None
        self._current_regime: str = NORMAL
        self._entry_regime: Optional[str] = None
        self._bb_width_history: list[float] = []

    def on_tick(self, market_state: MarketState) -> Signal:
        candles = market_state.recent_candles
        price = market_state.mark_price
        timestamp = market_state.timestamp

        if price <= 0:
            return self._hold(price, timestamp)

        # Need enough candles for all indicators
        min_candles = max(self.atr_slow, self.bb_period, self.kc_period, self.rsi_period + 1) + 5
        if len(candles) < min_candles:
            return self._hold(price, timestamp)

        closes = np.array([float(c.get("close", c.get("Close", 0))) for c in candles])
        highs = np.array([float(c.get("high", c.get("High", 0))) for c in candles])
        lows = np.array([float(c.get("low", c.get("Low", 0))) for c in candles])

        # Compute indicators
        atr_fast = self._compute_atr(highs, lows, closes, self.atr_fast)
        atr_slow = self._compute_atr(highs, lows, closes, self.atr_slow)
        if atr_fast is None or atr_slow is None or atr_slow[-1] == 0:
            return self._hold(price, timestamp)

        atr_ratio = atr_fast[-1] / atr_slow[-1] if atr_slow[-1] > 0 else 1.0

        bb_width = self._compute_bb_width(closes, self.bb_period, self.bb_std)
        if bb_width is None:
            return self._hold(price, timestamp)

        # Track BB width history for percentile
        curr_bb_width = bb_width[-1]
        self._bb_width_history.append(curr_bb_width)
        if len(self._bb_width_history) > 100:
            self._bb_width_history = self._bb_width_history[-100:]

        bb_width_pctl = self._percentile(curr_bb_width, self._bb_width_history)

        # Classify regime
        self._current_regime = self._classify_vol_regime(atr_ratio, curr_bb_width, bb_width_pctl)

        # Compute additional indicators for entries
        kc_upper, kc_lower, kc_mid = self._compute_keltner_channels(
            closes, highs, lows, self.kc_period, self.kc_atr_mult)
        if kc_upper is None:
            return self._hold(price, timestamp)

        rsi = self._compute_rsi(closes, self.rsi_period)
        if rsi is None:
            return self._hold(price, timestamp)

        # Bollinger Bands for contraction entries
        bb_upper, bb_lower, bb_mid = self._compute_bollinger_bands(
            closes, self.bb_period, self.bb_std)
        if bb_upper is None:
            return self._hold(price, timestamp)

        position_size = market_state.equity * self.position_size_pct / price if price > 0 else 0

        # Check exit first if in position
        if self._position_side is not None:
            return self._check_exit(
                price, atr_ratio, rsi[-1], timestamp, position_size)

        # Check entries based on regime
        if self._current_regime == EXPANDING:
            return self._check_expansion_entry(
                price, kc_upper[-1], kc_lower[-1], rsi[-1],
                atr_ratio, timestamp, position_size)
        elif self._current_regime in (LOW_VOL, CONTRACTING):
            return self._check_contraction_entry(
                price, bb_lower[-1], bb_upper[-1], rsi[-1],
                timestamp, position_size)

        return self._hold(price, timestamp)

    def _classify_vol_regime(self, atr_ratio: float, bb_width: float,
                             bb_width_pctl: float) -> str:
        """Classify the current volatility regime.

        Args:
            atr_ratio: Fast ATR / Slow ATR ratio.
            bb_width: Current Bollinger Band width.
            bb_width_pctl: BB width percentile (0-100).

        Returns:
            One of: LOW_VOL, NORMAL, HIGH_VOL, EXPANDING, CONTRACTING
        """
        if atr_ratio >= self.expansion_threshold:
            return EXPANDING
        elif atr_ratio <= self.contraction_threshold:
            return CONTRACTING
        elif bb_width_pctl <= 20:
            return LOW_VOL
        elif bb_width_pctl >= 80:
            return HIGH_VOL
        else:
            return NORMAL

    def _check_expansion_entry(self, price: float, kc_upper: float,
                               kc_lower: float, rsi: float,
                               atr_ratio: float, timestamp: datetime,
                               size: float) -> Signal:
        """Check for Keltner Channel breakout entries during vol expansion."""
        # Long: price breaks above KC upper + RSI not overbought
        if price > kc_upper and rsi < 70:
            self._position_side = "long"
            self._entry_price = price
            self._entry_regime = EXPANDING
            return Signal(
                signal_type=SignalType.ENTER_LONG,
                price=price,
                timestamp=timestamp,
                size=size,
                metadata={
                    "regime": EXPANDING,
                    "atr_ratio": round(atr_ratio, 4),
                    "rsi": round(rsi, 2),
                    "kc_upper": round(kc_upper, 2),
                    "entry_reason": "kc_breakout_long",
                },
            )

        # Short: price breaks below KC lower + RSI not oversold
        if price < kc_lower and rsi > 30:
            self._position_side = "short"
            self._entry_price = price
            self._entry_regime = EXPANDING
            return Signal(
                signal_type=SignalType.ENTER_SHORT,
                price=price,
                timestamp=timestamp,
                size=size,
                metadata={
                    "regime": EXPANDING,
                    "atr_ratio": round(atr_ratio, 4),
                    "rsi": round(rsi, 2),
                    "kc_lower": round(kc_lower, 2),
                    "entry_reason": "kc_breakout_short",
                },
            )

        return self._hold(price, timestamp)

    def _check_contraction_entry(self, price: float, bb_lower: float,
                                 bb_upper: float, rsi: float,
                                 timestamp: datetime,
                                 size: float) -> Signal:
        """Check for mean-reversion entries during vol contraction/low vol."""
        # Long: price below lower BB + RSI oversold
        if price < bb_lower and rsi < 30:
            self._position_side = "long"
            self._entry_price = price
            self._entry_regime = self._current_regime
            return Signal(
                signal_type=SignalType.ENTER_LONG,
                price=price,
                timestamp=timestamp,
                size=size,
                metadata={
                    "regime": self._current_regime,
                    "rsi": round(rsi, 2),
                    "bb_lower": round(bb_lower, 2),
                    "entry_reason": "bb_oversold_long",
                },
            )

        # Short: price above upper BB + RSI overbought
        if price > bb_upper and rsi > 70:
            self._position_side = "short"
            self._entry_price = price
            self._entry_regime = self._current_regime
            return Signal(
                signal_type=SignalType.ENTER_SHORT,
                price=price,
                timestamp=timestamp,
                size=size,
                metadata={
                    "regime": self._current_regime,
                    "rsi": round(rsi, 2),
                    "bb_upper": round(bb_upper, 2),
                    "entry_reason": "bb_overbought_short",
                },
            )

        return self._hold(price, timestamp)

    def _check_exit(self, price: float, atr_ratio: float, rsi: float,
                    timestamp: datetime, size: float) -> Signal:
        """Check for exit conditions on open position."""
        should_exit = False
        exit_reason = ""

        # Exit if regime changes significantly from entry
        if self._entry_regime == EXPANDING and self._current_regime in (CONTRACTING, LOW_VOL):
            should_exit = True
            exit_reason = "regime_change"
        elif self._entry_regime in (CONTRACTING, LOW_VOL) and self._current_regime == EXPANDING:
            should_exit = True
            exit_reason = "regime_change"

        # Exit long on RSI overbought (mean-reversion target hit)
        if self._position_side == "long" and rsi > 70 and self._entry_regime != EXPANDING:
            should_exit = True
            exit_reason = "rsi_target"

        # Exit short on RSI oversold
        if self._position_side == "short" and rsi < 30 and self._entry_regime != EXPANDING:
            should_exit = True
            exit_reason = "rsi_target"

        if should_exit:
            if self._position_side == "long":
                signal_type = SignalType.EXIT_LONG
            else:
                signal_type = SignalType.EXIT_SHORT

            self._position_side = None
            self._entry_price = None
            self._entry_regime = None
            return Signal(
                signal_type=signal_type,
                price=price,
                timestamp=timestamp,
                size=size,
                metadata={"exit_reason": exit_reason, "rsi": round(rsi, 2),
                          "regime": self._current_regime},
            )

        return self._hold(price, timestamp)

    @staticmethod
    def _compute_atr(highs: np.ndarray, lows: np.ndarray,
                     closes: np.ndarray, period: int) -> Optional[np.ndarray]:
        """Compute Average True Range."""
        n = len(closes)
        if n < period + 1:
            return None

        tr = np.zeros(n)
        tr[0] = highs[0] - lows[0]
        for i in range(1, n):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )

        atr = np.zeros(n)
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

        # Fill initial values
        for i in range(period - 1):
            atr[i] = atr[period - 1]

        return atr

    @staticmethod
    def _compute_keltner_channels(closes: np.ndarray, highs: np.ndarray,
                                  lows: np.ndarray, period: int,
                                  atr_mult: float) -> tuple:
        """Compute Keltner Channels (EMA +/- ATR multiplier).

        Returns: (upper, lower, mid) arrays or (None, None, None).
        """
        n = len(closes)
        if n < period + 1:
            return None, None, None

        # Mid = EMA of closes
        multiplier = 2.0 / (period + 1)
        mid = np.zeros(n)
        mid[period - 1] = np.mean(closes[:period])
        for i in range(period, n):
            mid[i] = (closes[i] - mid[i - 1]) * multiplier + mid[i - 1]
        for i in range(period - 1):
            mid[i] = mid[period - 1]

        # ATR
        tr = np.zeros(n)
        tr[0] = highs[0] - lows[0]
        for i in range(1, n):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )

        atr = np.zeros(n)
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        for i in range(period - 1):
            atr[i] = atr[period - 1]

        upper = mid + atr_mult * atr
        lower = mid - atr_mult * atr

        return upper, lower, mid

    @staticmethod
    def _compute_bb_width(closes: np.ndarray, period: int,
                          num_std: float) -> Optional[np.ndarray]:
        """Compute Bollinger Band width (upper - lower) / mid.

        Returns: array of BB widths or None if insufficient data.
        """
        n = len(closes)
        if n < period:
            return None

        width = np.zeros(n)
        for i in range(period - 1, n):
            window = closes[i - period + 1:i + 1]
            mid = np.mean(window)
            if mid == 0:
                continue
            std = np.std(window, ddof=0)
            upper = mid + num_std * std
            lower = mid - num_std * std
            width[i] = (upper - lower) / mid

        # Fill initial values
        for i in range(period - 1):
            width[i] = width[period - 1]

        return width

    @staticmethod
    def _compute_bollinger_bands(closes: np.ndarray, period: int,
                                 num_std: float) -> tuple:
        """Compute Bollinger Bands.

        Returns: (upper, lower, mid) arrays or (None, None, None).
        """
        n = len(closes)
        if n < period:
            return None, None, None

        upper = np.zeros(n)
        lower = np.zeros(n)
        mid = np.zeros(n)

        for i in range(period - 1, n):
            window = closes[i - period + 1:i + 1]
            m = np.mean(window)
            std = np.std(window, ddof=0)
            mid[i] = m
            upper[i] = m + num_std * std
            lower[i] = m - num_std * std

        # Fill initial
        for i in range(period - 1):
            upper[i] = upper[period - 1]
            lower[i] = lower[period - 1]
            mid[i] = mid[period - 1]

        return upper, lower, mid

    @staticmethod
    def _compute_rsi(closes: np.ndarray, period: int) -> Optional[np.ndarray]:
        """Compute RSI indicator."""
        if len(closes) < period + 1:
            return None
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        rsi = np.zeros(len(closes))

        if avg_loss == 0:
            rsi[period] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[period] = 100.0 - (100.0 / (1.0 + rs))

        for i in range(period + 1, len(closes)):
            delta = deltas[i - 1]
            gain = max(delta, 0.0)
            loss = max(-delta, 0.0)
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period

            if avg_loss == 0:
                rsi[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100.0 - (100.0 / (1.0 + rs))

        for i in range(period):
            rsi[i] = 50.0

        return rsi

    @staticmethod
    def _percentile(value: float, history: list[float]) -> float:
        """Compute percentile of value within history."""
        if not history:
            return 50.0
        count_below = sum(1 for v in history if v < value)
        return count_below / len(history) * 100.0

    def _hold(self, price: float, timestamp: datetime) -> Signal:
        return Signal(
            signal_type=SignalType.HOLD,
            price=price,
            timestamp=timestamp,
        )

    def get_metadata(self) -> dict:
        return {
            "name": "volatility_regime",
            "version": "1.0",
            "category": "volatility",
            "timeframes": ["1h"],
            "description": "Volatility regime strategy (ATR + BB + Keltner)",
            "params": {
                "atr_fast": self.atr_fast,
                "atr_slow": self.atr_slow,
                "bb_period": self.bb_period,
                "bb_std": self.bb_std,
                "kc_period": self.kc_period,
                "kc_atr_mult": self.kc_atr_mult,
                "expansion_threshold": self.expansion_threshold,
                "contraction_threshold": self.contraction_threshold,
                "rsi_period": self.rsi_period,
                "position_size_pct": self.position_size_pct,
            },
        }

    def get_state(self) -> dict:
        return {
            "position_side": self._position_side,
            "entry_price": self._entry_price,
            "current_regime": self._current_regime,
            "entry_regime": self._entry_regime,
            "bb_width_history": self._bb_width_history[-50:],  # Keep last 50
        }

    def set_state(self, state: dict) -> None:
        self._position_side = state.get("position_side")
        self._entry_price = state.get("entry_price")
        self._current_regime = state.get("current_regime", NORMAL)
        self._entry_regime = state.get("entry_regime")
        self._bb_width_history = state.get("bb_width_history", [])


# ---------------------------------------------------------------------------
# Backtest adapter
# ---------------------------------------------------------------------------

import pandas as pd
from strategy.base import BaseStrategy


class VolatilityRegimeBacktestAdapter(BaseStrategy):
    """Backtest adapter wrapping VolatilityRegimeStrategy for the bar-based engine."""

    def __init__(self, params: dict[str, Any]):
        super().__init__(params)
        self.live_strategy = VolatilityRegimeStrategy(params)

    def setup(self, df: pd.DataFrame) -> None:
        """No pre-computation needed -- indicators computed per-tick."""
        pass

    def generate_signal(self, index: int, df: pd.DataFrame) -> Signal:
        """Convert a bar-based call into an event-based MarketState tick."""
        row = df.iloc[index]

        # Build candle history up to current bar
        lookback = min(index + 1, 100)
        candle_slice = df.iloc[max(0, index - lookback + 1):index + 1]
        recent_candles = [
            {
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
                "volume": float(r.get("volume", 0)),
            }
            for _, r in candle_slice.iterrows()
        ]

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
            recent_candles=recent_candles,
            timestamp=df.index[index] if hasattr(df.index[index], 'tzinfo') else datetime.now(timezone.utc),
        )

        signal = self.live_strategy.on_tick(market_state)
        signal.timestamp = df.index[index] if hasattr(df.index, '__getitem__') else signal.timestamp
        return signal
