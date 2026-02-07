"""Mean reversion strategy using Bollinger Bands + RSI with regime filter."""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from core.models import Signal
from core.types import SignalType
from strategy.live_strategy import LiveStrategy, MarketState

logger = logging.getLogger(__name__)


class MeanReversionStrategy(LiveStrategy):
    """Bollinger Band mean reversion with RSI confirmation and regime filter.

    Only enters positions during 'ranging' market regime (from MarketRegimeClassifier).
    Buys at lower band with oversold RSI, sells at upper band with overbought RSI.
    Exits when price crosses the middle band (SMA).
    """

    def __init__(self, params: dict[str, Any]):
        super().__init__(params)
        self.bb_period = int(params.get("bb_period", 20))
        self.bb_std = float(params.get("bb_std", 2.0))
        self.rsi_period = int(params.get("rsi_period", 14))
        self.rsi_entry_low = float(params.get("rsi_entry_low", 30))
        self.rsi_entry_high = float(params.get("rsi_entry_high", 70))
        self.position_size_pct = float(params.get("position_size_pct", 0.02))
        self.regime_filter = bool(params.get("regime_filter", True))
        self._position_side: Optional[str] = None
        self._entry_price: Optional[float] = None

    def on_tick(self, market_state: MarketState) -> Signal:
        price = market_state.mark_price
        timestamp = market_state.timestamp
        candles = market_state.recent_candles

        if price <= 0:
            return self._hold(price, timestamp)

        min_candles = max(self.bb_period, self.rsi_period + 1) + 5
        if len(candles) < min_candles:
            return self._hold(price, timestamp)

        closes = np.array([float(c.get("close", c.get("Close", 0))) for c in candles])

        # Compute indicators
        sma, upper_band, lower_band = self._compute_bollinger_bands(closes, self.bb_period, self.bb_std)
        rsi = self._compute_rsi(closes, self.rsi_period)

        if sma is None or rsi is None:
            return self._hold(price, timestamp)

        curr_sma = sma[-1]
        curr_upper = upper_band[-1]
        curr_lower = lower_band[-1]
        curr_rsi = rsi[-1]

        position_size = market_state.equity * self.position_size_pct / price if price > 0 else 0

        # Check exit first
        if self._position_side is not None:
            return self._check_exit(price, curr_sma, timestamp, position_size)

        # Regime filter: only trade in ranging market
        if self.regime_filter:
            metadata = getattr(market_state, 'metadata', {}) or {}
            regime = metadata.get("regime", "unknown")
            if regime not in ("ranging", "unknown"):
                return self._hold(price, timestamp)

        # Check entry
        return self._check_entry(
            price, curr_lower, curr_upper, curr_rsi,
            timestamp, position_size)

    def _check_entry(self, price, lower_band, upper_band, rsi,
                     timestamp, size) -> Signal:
        """Check for mean reversion entry conditions."""
        # Long entry: price below lower BB + RSI oversold
        if price < lower_band and rsi < self.rsi_entry_low:
            self._position_side = "long"
            self._entry_price = price
            return Signal(
                signal_type=SignalType.ENTER_LONG,
                price=price,
                timestamp=timestamp,
                size=size,
                metadata={
                    "entry_reason": "lower_band_oversold",
                    "rsi": round(rsi, 2),
                    "lower_band": round(lower_band, 2),
                },
            )

        # Short entry: price above upper BB + RSI overbought
        if price > upper_band and rsi > self.rsi_entry_high:
            self._position_side = "short"
            self._entry_price = price
            return Signal(
                signal_type=SignalType.ENTER_SHORT,
                price=price,
                timestamp=timestamp,
                size=size,
                metadata={
                    "entry_reason": "upper_band_overbought",
                    "rsi": round(rsi, 2),
                    "upper_band": round(upper_band, 2),
                },
            )

        return self._hold(price, timestamp)

    def _check_exit(self, price, sma, timestamp, size) -> Signal:
        """Exit when price crosses the middle band (SMA)."""
        if self._position_side == "long" and price >= sma:
            self._position_side = None
            self._entry_price = None
            return Signal(
                signal_type=SignalType.EXIT_LONG,
                price=price,
                timestamp=timestamp,
                size=size,
                metadata={"exit_reason": "crossed_midline"},
            )

        if self._position_side == "short" and price <= sma:
            self._position_side = None
            self._entry_price = None
            return Signal(
                signal_type=SignalType.EXIT_SHORT,
                price=price,
                timestamp=timestamp,
                size=size,
                metadata={"exit_reason": "crossed_midline"},
            )

        return self._hold(price, timestamp)

    @staticmethod
    def _compute_bollinger_bands(closes: np.ndarray, period: int,
                                  num_std: float) -> tuple:
        """Compute Bollinger Bands (SMA, upper, lower).
        
        Returns (sma, upper, lower) as numpy arrays, or (None, None, None) if insufficient data.
        """
        if len(closes) < period:
            return None, None, None

        sma = np.zeros(len(closes))
        upper = np.zeros(len(closes))
        lower = np.zeros(len(closes))

        for i in range(period - 1, len(closes)):
            window = closes[i - period + 1:i + 1]
            mean = np.mean(window)
            std = np.std(window)
            sma[i] = mean
            upper[i] = mean + num_std * std
            lower[i] = mean - num_std * std

        # Fill initial values
        for i in range(period - 1):
            sma[i] = sma[period - 1]
            upper[i] = upper[period - 1]
            lower[i] = lower[period - 1]

        return sma, upper, lower

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

    def _hold(self, price: float, timestamp: datetime) -> Signal:
        return Signal(
            signal_type=SignalType.HOLD,
            price=price,
            timestamp=timestamp,
        )

    def get_metadata(self) -> dict:
        return {
            "name": "mean_reversion",
            "version": "1.0",
            "category": "mean_reversion",
            "timeframes": ["1h"],
            "description": "Bollinger Band mean reversion with regime filter",
            "params": {
                "bb_period": self.bb_period,
                "bb_std": self.bb_std,
                "rsi_period": self.rsi_period,
                "rsi_entry_low": self.rsi_entry_low,
                "rsi_entry_high": self.rsi_entry_high,
                "position_size_pct": self.position_size_pct,
                "regime_filter": self.regime_filter,
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


class MeanReversionBacktestAdapter(BaseStrategy):
    """Backtest adapter wrapping MeanReversionStrategy for bar-based engine."""

    def __init__(self, params: dict[str, Any]):
        super().__init__(params)
        self.live_strategy = MeanReversionStrategy(params)

    def setup(self, df: pd.DataFrame) -> None:
        pass

    def generate_signal(self, index: int, df: pd.DataFrame) -> Signal:
        row = df.iloc[index]
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
            metadata={"regime": "ranging"},  # Default to ranging for backtest
        )

        signal = self.live_strategy.on_tick(market_state)
        signal.timestamp = df.index[index] if hasattr(df.index, '__getitem__') else signal.timestamp
        return signal
