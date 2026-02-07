"""Market regime classification using technical indicators."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketRegimeClassifier:
    """Classifies market regime as trending_up, trending_down, ranging, or volatile.

    Uses SMA crossover and ATR-based volatility measurement.
    """

    def __init__(self, params: Optional[dict] = None):
        params = params or {}
        self.sma_fast_period = params.get("sma_fast_period", 20)
        self.sma_slow_period = params.get("sma_slow_period", 50)
        self.atr_period = params.get("atr_period", 14)
        self.volatility_threshold = params.get("volatility_threshold", 2.0)
        self.trend_strength_threshold = params.get("trend_strength_threshold", 0.02)

    def classify(self, candles: pd.DataFrame) -> dict:
        """Classify market regime from OHLCV candle data.

        Args:
            candles: DataFrame with columns [open, high, low, close, volume]
                     and a DatetimeIndex (UTC).

        Returns:
            dict with keys: regime, confidence, indicators
        """
        if candles is None or len(candles) < self.sma_slow_period + 1:
            return {
                "regime": "unknown",
                "confidence": 0.0,
                "indicators": {},
            }

        close = candles["close"].astype(float)
        high = candles["high"].astype(float)
        low = candles["low"].astype(float)

        # Compute indicators
        sma_fast = close.rolling(window=self.sma_fast_period).mean()
        sma_slow = close.rolling(window=self.sma_slow_period).mean()
        atr = self._compute_atr(high, low, close)
        atr_mean = atr.rolling(window=self.sma_slow_period).mean()

        # Get latest values
        current_price = float(close.iloc[-1])
        current_sma_fast = float(sma_fast.iloc[-1])
        current_sma_slow = float(sma_slow.iloc[-1])
        current_atr = float(atr.iloc[-1])
        current_atr_mean = float(atr_mean.iloc[-1]) if not np.isnan(atr_mean.iloc[-1]) else current_atr

        # ATR ratio for volatility detection
        atr_ratio = current_atr / current_atr_mean if current_atr_mean > 0 else 1.0

        # Trend strength: distance of price from slow SMA as a fraction
        trend_strength = (current_price - current_sma_slow) / current_sma_slow if current_sma_slow > 0 else 0.0

        # Classification logic
        regime, confidence = self._classify_regime(
            current_price, current_sma_fast, current_sma_slow,
            atr_ratio, trend_strength,
        )

        return {
            "regime": regime,
            "confidence": round(confidence, 4),
            "indicators": {
                "sma_fast": round(current_sma_fast, 4),
                "sma_slow": round(current_sma_slow, 4),
                "atr": round(current_atr, 4),
                "atr_ratio": round(atr_ratio, 4),
                "trend_strength": round(trend_strength, 6),
                "price": round(current_price, 4),
            },
        }

    def _compute_atr(self, high: pd.Series, low: pd.Series,
                     close: pd.Series) -> pd.Series:
        """Compute Average True Range."""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=self.atr_period).mean()

    def _classify_regime(self, price: float, sma_fast: float, sma_slow: float,
                         atr_ratio: float, trend_strength: float) -> tuple[str, float]:
        """Apply classification rules.

        Returns:
            (regime, confidence) tuple
        """
        # Rule 1: High volatility overrides trend classification
        if atr_ratio > self.volatility_threshold:
            confidence = min(1.0, (atr_ratio - self.volatility_threshold) / self.volatility_threshold + 0.5)
            return "volatile", confidence

        # Rule 2: Trending up — price > SMA_fast > SMA_slow with strength
        if (price > sma_fast > sma_slow and
                trend_strength > self.trend_strength_threshold):
            confidence = min(1.0, abs(trend_strength) / 0.05 * 0.5 + 0.5)
            return "trending_up", confidence

        # Rule 3: Trending down — price < SMA_fast < SMA_slow with strength
        if (price < sma_fast < sma_slow and
                trend_strength < -self.trend_strength_threshold):
            confidence = min(1.0, abs(trend_strength) / 0.05 * 0.5 + 0.5)
            return "trending_down", confidence

        # Rule 4: Ranging (default)
        confidence = max(0.3, 1.0 - abs(trend_strength) / 0.05)
        return "ranging", min(1.0, confidence)
