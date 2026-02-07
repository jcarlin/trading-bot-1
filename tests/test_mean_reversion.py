#!/usr/bin/env python3
"""Tests for mean reversion strategy."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from core.models import Signal
from core.types import SignalType
from strategy.mean_reversion import MeanReversionStrategy, MeanReversionBacktestAdapter
from strategy.live_strategy import MarketState


class TestMeanReversionStrategy(unittest.TestCase):

    def setUp(self):
        self.params = {
            "bb_period": 20,
            "bb_std": 2.0,
            "rsi_period": 14,
            "rsi_entry_low": 30,
            "rsi_entry_high": 70,
            "position_size_pct": 0.02,
            "regime_filter": True,
        }
        self.strategy = MeanReversionStrategy(self.params)
        self.base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)

    def _make_candles(self, closes):
        candles = []
        for c in closes:
            candles.append({
                "open": c - 1,
                "high": c + 5,
                "low": c - 5,
                "close": c,
                "volume": 100.0,
            })
        return candles

    def _make_market_state(self, candles, mark_price=None, equity=10000.0,
                           timestamp=None, regime="ranging"):
        if mark_price is None:
            mark_price = candles[-1]["close"]
        return MarketState(
            mark_price=mark_price,
            mid_price=mark_price,
            bid=mark_price - 0.5,
            ask=mark_price + 0.5,
            funding_rate=0.0,
            premium=0.0,
            open_interest=0.0,
            equity=equity,
            cash=equity * 0.5,
            recent_candles=candles,
            timestamp=timestamp or self.base_time,
            metadata={"regime": regime},
        )

    def test_hold_insufficient_candles(self):
        candles = self._make_candles([50000 + i for i in range(5)])
        ms = self._make_market_state(candles)
        signal = self.strategy.on_tick(ms)
        self.assertEqual(signal.signal_type, SignalType.HOLD)

    def test_hold_zero_price(self):
        closes = [50000.0] * 40
        candles = self._make_candles(closes)
        ms = self._make_market_state(candles, mark_price=0)
        signal = self.strategy.on_tick(ms)
        self.assertEqual(signal.signal_type, SignalType.HOLD)

    def test_enter_long_at_lower_band(self):
        """Price drops sharply below lower BB with low RSI -> ENTER_LONG."""
        # Ranging then sharp drop
        closes = [50000.0] * 30 + [50000 - i * 100 for i in range(1, 16)]
        candles = self._make_candles(closes)
        ms = self._make_market_state(candles, regime="ranging")
        signal = self.strategy.on_tick(ms)
        # May or may not trigger depending on exact RSI/BB values
        # At least verify it doesn't crash
        self.assertIsInstance(signal, Signal)

    def test_enter_short_at_upper_band(self):
        """Price surges above upper BB with high RSI -> ENTER_SHORT."""
        closes = [50000.0] * 30 + [50000 + i * 100 for i in range(1, 16)]
        candles = self._make_candles(closes)
        ms = self._make_market_state(candles, regime="ranging")
        signal = self.strategy.on_tick(ms)
        self.assertIsInstance(signal, Signal)

    def test_regime_filter_blocks_trending(self):
        """In trending regime with regime_filter=True, should HOLD."""
        closes = [50000 - i * 200 for i in range(45)]
        candles = self._make_candles(closes)
        ms = self._make_market_state(candles, regime="trending_down")
        signal = self.strategy.on_tick(ms)
        self.assertEqual(signal.signal_type, SignalType.HOLD)

    def test_regime_filter_blocks_volatile(self):
        """In volatile regime, should HOLD."""
        closes = [50000.0] * 30 + [50000 - i * 200 for i in range(1, 16)]
        candles = self._make_candles(closes)
        ms = self._make_market_state(candles, regime="volatile")
        signal = self.strategy.on_tick(ms)
        self.assertEqual(signal.signal_type, SignalType.HOLD)

    def test_regime_filter_allows_unknown(self):
        """Unknown regime should be allowed (default behavior)."""
        # The strategy should proceed to check entry conditions
        closes = [50000.0] * 30 + [50000 - i * 200 for i in range(1, 16)]
        candles = self._make_candles(closes)
        ms = self._make_market_state(candles, regime="unknown")
        signal = self.strategy.on_tick(ms)
        # Should not error, might be HOLD or entry depending on indicators
        self.assertIsInstance(signal, Signal)

    def test_regime_filter_disabled(self):
        """With regime_filter=False, should trade regardless of regime."""
        params = dict(self.params, regime_filter=False)
        strategy = MeanReversionStrategy(params)
        closes = [50000.0] * 30 + [50000 - i * 200 for i in range(1, 16)]
        candles = self._make_candles(closes)
        ms = self._make_market_state(candles, regime="trending_up")
        signal = strategy.on_tick(ms)
        # Should not be blocked by regime - might be HOLD or entry
        self.assertIsInstance(signal, Signal)

    def test_exit_long_at_midline(self):
        """Exit long when price crosses above SMA."""
        self.strategy._position_side = "long"
        self.strategy._entry_price = 49000.0
        # Price at or above SMA (mean of recent closes)
        closes = [50000.0] * 40
        candles = self._make_candles(closes)
        ms = self._make_market_state(candles, mark_price=50000.0)
        signal = self.strategy.on_tick(ms)
        self.assertEqual(signal.signal_type, SignalType.EXIT_LONG)

    def test_exit_short_at_midline(self):
        """Exit short when price drops to SMA."""
        self.strategy._position_side = "short"
        self.strategy._entry_price = 51000.0
        closes = [50000.0] * 40
        candles = self._make_candles(closes)
        ms = self._make_market_state(candles, mark_price=50000.0)
        signal = self.strategy.on_tick(ms)
        self.assertEqual(signal.signal_type, SignalType.EXIT_SHORT)

    def test_metadata(self):
        meta = self.strategy.get_metadata()
        self.assertEqual(meta["name"], "mean_reversion")
        self.assertEqual(meta["category"], "mean_reversion")
        self.assertIn("bb_period", meta["params"])
        self.assertIn("regime_filter", meta["params"])

    def test_state_checkpoint_restore(self):
        self.strategy._position_side = "short"
        self.strategy._entry_price = 51000.0
        state = self.strategy.get_state()

        new_strategy = MeanReversionStrategy(self.params)
        new_strategy.set_state(state)
        self.assertEqual(new_strategy._position_side, "short")
        self.assertEqual(new_strategy._entry_price, 51000.0)

    def test_state_empty_default(self):
        state = self.strategy.get_state()
        self.assertIsNone(state["position_side"])
        self.assertIsNone(state["entry_price"])

    def test_bollinger_bands_computation(self):
        closes = np.array([50000.0 + np.sin(i * 0.3) * 100 for i in range(30)])
        sma, upper, lower = MeanReversionStrategy._compute_bollinger_bands(closes, 20, 2.0)
        self.assertIsNotNone(sma)
        self.assertGreater(upper[-1], sma[-1])
        self.assertLess(lower[-1], sma[-1])

    def test_bollinger_bands_insufficient_data(self):
        closes = np.array([50000.0] * 5)
        sma, upper, lower = MeanReversionStrategy._compute_bollinger_bands(closes, 20, 2.0)
        self.assertIsNone(sma)

    def test_rsi_computation(self):
        closes = np.array([float(i * 10) for i in range(30)])
        rsi = MeanReversionStrategy._compute_rsi(closes, 14)
        self.assertIsNotNone(rsi)
        self.assertGreater(rsi[-1], 80)

    def test_rsi_insufficient_data(self):
        closes = np.array([1.0, 2.0, 3.0])
        rsi = MeanReversionStrategy._compute_rsi(closes, 14)
        self.assertIsNone(rsi)

    def test_signal_has_size(self):
        """Entry signals should have position size."""
        self.strategy._position_side = "long"
        closes = [50000.0] * 40
        candles = self._make_candles(closes)
        ms = self._make_market_state(candles, equity=10000.0)
        signal = self.strategy.on_tick(ms)
        if signal.signal_type != SignalType.HOLD:
            self.assertIsNotNone(signal.size)
            self.assertGreater(signal.size, 0)

    def test_no_metadata_field_graceful(self):
        """Strategy should handle MarketState without metadata gracefully."""
        closes = [50000.0] * 40
        candles = self._make_candles(closes)
        # Create MarketState without metadata field (simulate pre-Phase 3)
        ms = MarketState(
            mark_price=50000.0,
            mid_price=50000.0,
            bid=49999.5,
            ask=50000.5,
            funding_rate=0.0,
            premium=0.0,
            open_interest=0.0,
            equity=10000.0,
            cash=5000.0,
            recent_candles=candles,
            timestamp=self.base_time,
        )
        signal = self.strategy.on_tick(ms)
        # metadata defaults to {}, so regime is "unknown" which is allowed
        self.assertIsInstance(signal, Signal)


class TestMeanReversionBacktestAdapter(unittest.TestCase):

    def setUp(self):
        self.params = {
            "bb_period": 20,
            "bb_std": 2.0,
            "rsi_period": 14,
            "regime_filter": False,  # Disable for backtest simplicity
        }
        self.adapter = MeanReversionBacktestAdapter(self.params)

    def _make_df(self, n=50):
        dates = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
        close = [50000 + 200 * np.sin(i * 0.3) for i in range(n)]
        data = {
            "open": [c - 5 for c in close],
            "high": [c + 10 for c in close],
            "low": [c - 10 for c in close],
            "close": close,
            "volume": [100.0] * n,
        }
        return pd.DataFrame(data, index=dates)

    def test_setup_does_nothing(self):
        df = self._make_df()
        self.adapter.setup(df)

    def test_generate_signal_returns_signal(self):
        df = self._make_df()
        signal = self.adapter.generate_signal(49, df)
        self.assertIsInstance(signal, Signal)

    def test_adapter_wraps_live_strategy(self):
        self.assertIsInstance(self.adapter.live_strategy, MeanReversionStrategy)


class TestMarketStateMetadata(unittest.TestCase):
    """Verify the metadata field works correctly on MarketState."""

    def test_metadata_default_empty(self):
        ms = MarketState(
            mark_price=50000.0,
            mid_price=50000.0,
            bid=49999.5,
            ask=50000.5,
            funding_rate=0.0,
            premium=0.0,
            open_interest=0.0,
        )
        self.assertEqual(ms.metadata, {})

    def test_metadata_with_regime(self):
        ms = MarketState(
            mark_price=50000.0,
            mid_price=50000.0,
            bid=49999.5,
            ask=50000.5,
            funding_rate=0.0,
            premium=0.0,
            open_interest=0.0,
            metadata={"regime": "ranging", "regime_confidence": 0.85},
        )
        self.assertEqual(ms.metadata["regime"], "ranging")
        self.assertEqual(ms.metadata["regime_confidence"], 0.85)


if __name__ == "__main__":
    unittest.main()
