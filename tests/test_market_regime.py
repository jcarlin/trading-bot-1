"""Tests for analysis.market_regime.MarketRegimeClassifier."""

import os
import sys
import unittest
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.market_regime import MarketRegimeClassifier


class TestMarketRegimeClassifier(unittest.TestCase):
    """Tests for MarketRegimeClassifier."""

    def setUp(self):
        self.classifier = MarketRegimeClassifier()

    def _make_candles(self, prices: list[float], volatility: float = 0.5) -> pd.DataFrame:
        """Create a mock OHLCV DataFrame from a list of close prices."""
        n = len(prices)
        dates = pd.date_range(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            periods=n, freq="1h")
        df = pd.DataFrame({
            "open": [p - volatility for p in prices],
            "high": [p + volatility for p in prices],
            "low": [p - volatility for p in prices],
            "close": prices,
            "volume": [1000.0] * n,
        }, index=dates)
        return df

    def test_trending_up(self):
        """Steadily rising prices should be classified as trending_up."""
        # Create 60 candles of steadily rising prices
        prices = [100 + i * 0.5 for i in range(60)]
        candles = self._make_candles(prices, volatility=0.3)

        result = self.classifier.classify(candles)
        self.assertEqual(result["regime"], "trending_up")
        self.assertGreater(result["confidence"], 0.3)
        self.assertIn("sma_fast", result["indicators"])
        self.assertIn("sma_slow", result["indicators"])

    def test_trending_down(self):
        """Steadily falling prices should be classified as trending_down."""
        prices = [200 - i * 0.5 for i in range(60)]
        candles = self._make_candles(prices, volatility=0.3)

        result = self.classifier.classify(candles)
        self.assertEqual(result["regime"], "trending_down")
        self.assertGreater(result["confidence"], 0.3)

    def test_ranging(self):
        """Oscillating prices around a mean should be classified as ranging."""
        # Sine wave around 100 with small amplitude
        prices = [100 + 0.5 * np.sin(i * 0.3) for i in range(60)]
        candles = self._make_candles(prices, volatility=0.3)

        result = self.classifier.classify(candles)
        self.assertEqual(result["regime"], "ranging")

    def test_volatile(self):
        """High volatility should be classified as volatile."""
        # Long calm period so ATR mean stays low, then a sharp volatility spike
        prices = [100.0] * 80
        # Add large swings at the end
        for i in range(15):
            prices.append(100 + (10 * ((-1) ** i)))

        candles = self._make_candles(prices, volatility=0.3)
        # Override high/low on the last 15 bars to create large ATR spike
        candles.loc[candles.index[-15:], "high"] = candles.loc[candles.index[-15:], "close"] + 15.0
        candles.loc[candles.index[-15:], "low"] = candles.loc[candles.index[-15:], "close"] - 15.0

        result = self.classifier.classify(candles)
        self.assertEqual(result["regime"], "volatile")

    def test_insufficient_data(self):
        """Too few candles should return unknown."""
        prices = [100.0] * 10
        candles = self._make_candles(prices)

        result = self.classifier.classify(candles)
        self.assertEqual(result["regime"], "unknown")
        self.assertEqual(result["confidence"], 0.0)

    def test_none_input(self):
        """None input should return unknown."""
        result = self.classifier.classify(None)
        self.assertEqual(result["regime"], "unknown")

    def test_custom_params(self):
        """Custom parameters should be used."""
        classifier = MarketRegimeClassifier({
            "sma_fast_period": 10,
            "sma_slow_period": 30,
            "atr_period": 7,
            "volatility_threshold": 3.0,
        })
        self.assertEqual(classifier.sma_fast_period, 10)
        self.assertEqual(classifier.sma_slow_period, 30)
        self.assertEqual(classifier.atr_period, 7)
        self.assertEqual(classifier.volatility_threshold, 3.0)

    def test_result_structure(self):
        """Result should have correct structure."""
        prices = [100 + i * 0.5 for i in range(60)]
        candles = self._make_candles(prices, volatility=0.3)

        result = self.classifier.classify(candles)
        self.assertIn("regime", result)
        self.assertIn("confidence", result)
        self.assertIn("indicators", result)
        self.assertIn("atr", result["indicators"])
        self.assertIn("atr_ratio", result["indicators"])
        self.assertIn("trend_strength", result["indicators"])
        self.assertIn("price", result["indicators"])

    def test_confidence_bounded(self):
        """Confidence should be between 0 and 1."""
        prices = [100 + i * 2 for i in range(60)]
        candles = self._make_candles(prices, volatility=0.3)

        result = self.classifier.classify(candles)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()
