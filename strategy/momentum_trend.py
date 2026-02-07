"""Momentum/trend following strategy using EMA crossover + RSI + ADX."""

import logging
import math
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from core.models import Signal
from core.types import SignalType
from strategy.live_strategy import LiveStrategy, MarketState

logger = logging.getLogger(__name__)


class MomentumTrendStrategy(LiveStrategy):
    """Multi-indicator trend following strategy.

    Entry: ADX > threshold (trend exists) + EMA crossover + RSI confirmation
    Exit: EMA cross back or RSI extreme reversal
    """

    def __init__(self, params: dict[str, Any]):
        super().__init__(params)
        self.ema_fast = int(params.get("ema_fast", 12))
        self.ema_slow = int(params.get("ema_slow", 26))
        self.rsi_period = int(params.get("rsi_period", 14))
        self.adx_period = int(params.get("adx_period", 14))
        self.rsi_overbought = float(params.get("rsi_overbought", 70))
        self.rsi_oversold = float(params.get("rsi_oversold", 30))
        self.adx_threshold = float(params.get("adx_threshold", 25))
        self.position_size_pct = float(params.get("position_size_pct", 0.02))
        self._position_side: Optional[str] = None
        self._entry_price: Optional[float] = None
        self._prev_ema_fast: Optional[float] = None
        self._prev_ema_slow: Optional[float] = None

    def on_tick(self, market_state: MarketState) -> Signal:
        candles = market_state.recent_candles
        price = market_state.mark_price
        timestamp = market_state.timestamp

        if price <= 0:
            return self._hold(price, timestamp)

        # Need enough candles for indicators
        min_candles = max(self.ema_slow, self.adx_period + 1, self.rsi_period + 1) + 5
        if len(candles) < min_candles:
            return self._hold(price, timestamp)

        closes = np.array([float(c.get("close", c.get("Close", 0))) for c in candles])
        highs = np.array([float(c.get("high", c.get("High", 0))) for c in candles])
        lows = np.array([float(c.get("low", c.get("Low", 0))) for c in candles])

        # Compute indicators
        ema_fast = self._compute_ema(closes, self.ema_fast)
        ema_slow = self._compute_ema(closes, self.ema_slow)
        rsi = self._compute_rsi(closes, self.rsi_period)
        adx = self._compute_adx(highs, lows, closes, self.adx_period)

        if ema_fast is None or ema_slow is None or rsi is None or adx is None:
            return self._hold(price, timestamp)

        position_size = market_state.equity * self.position_size_pct / price if price > 0 else 0

        # Current values
        curr_ema_fast = ema_fast[-1]
        curr_ema_slow = ema_slow[-1]
        prev_ema_fast = ema_fast[-2] if len(ema_fast) >= 2 else curr_ema_fast
        prev_ema_slow = ema_slow[-2] if len(ema_slow) >= 2 else curr_ema_slow
        curr_rsi = rsi[-1]
        curr_adx = adx[-1]

        # Check position exit
        if self._position_side is not None:
            return self._check_exit(
                curr_ema_fast, curr_ema_slow, prev_ema_fast, prev_ema_slow,
                curr_rsi, price, timestamp, position_size)

        # Check entry
        return self._check_entry(
            curr_ema_fast, curr_ema_slow, prev_ema_fast, prev_ema_slow,
            curr_rsi, curr_adx, price, timestamp, position_size)

    def _check_entry(self, ema_fast, ema_slow, prev_fast, prev_slow,
                     rsi, adx, price, timestamp, size) -> Signal:
        """Check for trend entry conditions."""
        # Must have trend (ADX > threshold)
        if adx < self.adx_threshold:
            return self._hold(price, timestamp)

        # Bullish crossover: fast crosses above slow
        if prev_fast <= prev_slow and ema_fast > ema_slow and rsi < self.rsi_overbought:
            self._position_side = "long"
            self._entry_price = price
            return Signal(
                signal_type=SignalType.ENTER_LONG,
                price=price,
                timestamp=timestamp,
                size=size,
                metadata={"adx": round(adx, 2), "rsi": round(rsi, 2),
                          "ema_fast": round(ema_fast, 4), "ema_slow": round(ema_slow, 4),
                          "entry_reason": "bullish_crossover"},
            )

        # Bearish crossover: fast crosses below slow
        if prev_fast >= prev_slow and ema_fast < ema_slow and rsi > self.rsi_oversold:
            self._position_side = "short"
            self._entry_price = price
            return Signal(
                signal_type=SignalType.ENTER_SHORT,
                price=price,
                timestamp=timestamp,
                size=size,
                metadata={"adx": round(adx, 2), "rsi": round(rsi, 2),
                          "ema_fast": round(ema_fast, 4), "ema_slow": round(ema_slow, 4),
                          "entry_reason": "bearish_crossover"},
            )

        return self._hold(price, timestamp)

    def _check_exit(self, ema_fast, ema_slow, prev_fast, prev_slow,
                    rsi, price, timestamp, size) -> Signal:
        """Check for exit conditions on open position."""
        if self._position_side == "long":
            # Exit long: EMA cross back (fast below slow) or RSI overbought reversal
            if (prev_fast >= prev_slow and ema_fast < ema_slow) or rsi > self.rsi_overbought:
                exit_reason = "ema_cross_back" if ema_fast < ema_slow else "rsi_overbought"
                self._position_side = None
                self._entry_price = None
                return Signal(
                    signal_type=SignalType.EXIT_LONG,
                    price=price,
                    timestamp=timestamp,
                    size=size,
                    metadata={"exit_reason": exit_reason, "rsi": round(rsi, 2)},
                )

        elif self._position_side == "short":
            # Exit short: EMA cross back (fast above slow) or RSI oversold reversal
            if (prev_fast <= prev_slow and ema_fast > ema_slow) or rsi < self.rsi_oversold:
                exit_reason = "ema_cross_back" if ema_fast > ema_slow else "rsi_oversold"
                self._position_side = None
                self._entry_price = None
                return Signal(
                    signal_type=SignalType.EXIT_SHORT,
                    price=price,
                    timestamp=timestamp,
                    size=size,
                    metadata={"exit_reason": exit_reason, "rsi": round(rsi, 2)},
                )

        return self._hold(price, timestamp)

    @staticmethod
    def _compute_ema(data: np.ndarray, period: int) -> Optional[np.ndarray]:
        """Compute exponential moving average."""
        if len(data) < period:
            return None
        multiplier = 2.0 / (period + 1)
        ema = np.zeros(len(data))
        ema[period - 1] = np.mean(data[:period])
        for i in range(period, len(data)):
            ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1]
        # Fill initial values
        for i in range(period - 1):
            ema[i] = ema[period - 1]
        return ema

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

        # Fill initial values
        for i in range(period):
            rsi[i] = 50.0  # neutral default

        return rsi

    @staticmethod
    def _compute_adx(highs: np.ndarray, lows: np.ndarray,
                     closes: np.ndarray, period: int) -> Optional[np.ndarray]:
        """Compute ADX indicator."""
        n = len(closes)
        if n < period + 1:
            return None

        # True Range
        tr = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )

        # +DM and -DM
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        for i in range(1, n):
            up = highs[i] - highs[i - 1]
            down = lows[i - 1] - lows[i]
            if up > down and up > 0:
                plus_dm[i] = up
            if down > up and down > 0:
                minus_dm[i] = down

        # Smoothed TR, +DM, -DM
        atr = np.zeros(n)
        smooth_plus = np.zeros(n)
        smooth_minus = np.zeros(n)

        atr[period] = np.sum(tr[1:period + 1])
        smooth_plus[period] = np.sum(plus_dm[1:period + 1])
        smooth_minus[period] = np.sum(minus_dm[1:period + 1])

        for i in range(period + 1, n):
            atr[i] = atr[i - 1] - atr[i - 1] / period + tr[i]
            smooth_plus[i] = smooth_plus[i - 1] - smooth_plus[i - 1] / period + plus_dm[i]
            smooth_minus[i] = smooth_minus[i - 1] - smooth_minus[i - 1] / period + minus_dm[i]

        # +DI and -DI
        plus_di = np.zeros(n)
        minus_di = np.zeros(n)
        dx = np.zeros(n)

        for i in range(period, n):
            if atr[i] > 0:
                plus_di[i] = 100.0 * smooth_plus[i] / atr[i]
                minus_di[i] = 100.0 * smooth_minus[i] / atr[i]
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0:
                dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum

        # ADX: smoothed DX
        adx = np.zeros(n)
        # Initial ADX is the average of the first period DX values
        start_idx = 2 * period
        if start_idx < n:
            adx[start_idx] = np.mean(dx[period:start_idx + 1])
            for i in range(start_idx + 1, n):
                adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

        # Fill earlier values
        for i in range(start_idx):
            adx[i] = adx[start_idx] if start_idx < n else 0.0

        return adx

    def _hold(self, price: float, timestamp: datetime) -> Signal:
        return Signal(
            signal_type=SignalType.HOLD,
            price=price,
            timestamp=timestamp,
        )

    def get_metadata(self) -> dict:
        return {
            "name": "momentum_trend",
            "version": "1.0",
            "category": "momentum",
            "timeframes": ["1h"],
            "description": "Multi-indicator trend following (EMA + RSI + ADX)",
            "params": {
                "ema_fast": self.ema_fast,
                "ema_slow": self.ema_slow,
                "rsi_period": self.rsi_period,
                "adx_period": self.adx_period,
                "rsi_overbought": self.rsi_overbought,
                "rsi_oversold": self.rsi_oversold,
                "adx_threshold": self.adx_threshold,
                "position_size_pct": self.position_size_pct,
            },
        }

    def get_state(self) -> dict:
        return {
            "position_side": self._position_side,
            "entry_price": self._entry_price,
        }

    def set_state(self, state: dict) -> None:
        self._position_side = state.get("position_side")
        self._entry_price = state.get("entry_price")


# ---------------------------------------------------------------------------
# Backtest adapter
# ---------------------------------------------------------------------------

import pandas as pd
from strategy.base import BaseStrategy


class MomentumTrendBacktestAdapter(BaseStrategy):
    """Backtest adapter wrapping MomentumTrendStrategy for the bar-based engine."""

    def __init__(self, params: dict[str, Any]):
        super().__init__(params)
        self.live_strategy = MomentumTrendStrategy(params)

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
        # Override timestamp with bar timestamp
        signal.timestamp = df.index[index] if hasattr(df.index, '__getitem__') else signal.timestamp
        return signal
