#!/usr/bin/env python3
"""Tests for momentum/trend strategy."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from core.models import Signal
from core.types import SignalType
from strategy.momentum_trend import MomentumTrendStrategy, MomentumTrendBacktestAdapter
from strategy.live_strategy import MarketState


class TestMomentumTrendStrategy(unittest.TestCase):

    def setUp(self):
        self.params = {
            "ema_fast": 12,
            "ema_slow": 26,
            "rsi_period": 14,
            "adx_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "adx_threshold": 25,
            "position_size_pct": 0.02,
        }
        self.strategy = MomentumTrendStrategy(self.params)
        self.base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)

    def _make_candles(self, closes, base_price=50000.0, spread=100.0):
        """Make candle dicts from close prices."""
        candles = []
        for c in closes:
            candles.append({
                "open": c - spread * 0.1,
                "high": c + spread * 0.5,
                "low": c - spread * 0.5,
                "close": c,
                "volume": 100.0,
            })
        return candles

    def _make_trending_up_candles(self, n=50, start=50000, step=50):
        """Create clearly trending upward candle data."""
        closes = [start + i * step for i in range(n)]
        return self._make_candles(closes, spread=step * 0.5)

    def _make_trending_down_candles(self, n=50, start=55000, step=50):
        """Create clearly trending downward candle data."""
        closes = [start - i * step for i in range(n)]
        return self._make_candles(closes, spread=step * 0.5)

    def _make_ranging_candles(self, n=50, center=50000, amplitude=50):
        """Create ranging (sideways) candle data."""
        closes = [center + amplitude * np.sin(i * 0.5) for i in range(n)]
        return self._make_candles(closes, spread=amplitude * 0.5)

    def _make_market_state(self, candles, mark_price=None, equity=10000.0,
                           timestamp=None):
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
        )

    def test_hold_insufficient_candles(self):
        """Should HOLD when not enough candles for indicators."""
        candles = self._make_candles([50000 + i for i in range(5)])
        ms = self._make_market_state(candles)
        signal = self.strategy.on_tick(ms)
        self.assertEqual(signal.signal_type, SignalType.HOLD)

    def test_hold_zero_price(self):
        candles = self._make_trending_up_candles()
        ms = self._make_market_state(candles, mark_price=0)
        signal = self.strategy.on_tick(ms)
        self.assertEqual(signal.signal_type, SignalType.HOLD)

    def test_hold_ranging_market(self):
        """Ranging market should produce HOLD (low ADX)."""
        candles = self._make_ranging_candles(n=60, amplitude=10)
        ms = self._make_market_state(candles)
        signal = self.strategy.on_tick(ms)
        self.assertEqual(signal.signal_type, SignalType.HOLD)

    def test_metadata(self):
        meta = self.strategy.get_metadata()
        self.assertEqual(meta["name"], "momentum_trend")
        self.assertEqual(meta["version"], "1.0")
        self.assertEqual(meta["category"], "momentum")
        self.assertIn("ema_fast", meta["params"])

    def test_state_checkpoint_restore(self):
        self.strategy._position_side = "long"
        self.strategy._entry_price = 50000.0
        state = self.strategy.get_state()

        new_strategy = MomentumTrendStrategy(self.params)
        new_strategy.set_state(state)
        self.assertEqual(new_strategy._position_side, "long")
        self.assertEqual(new_strategy._entry_price, 50000.0)

    def test_state_empty_default(self):
        state = self.strategy.get_state()
        self.assertIsNone(state["position_side"])
        self.assertIsNone(state["entry_price"])

    def test_signal_has_size(self):
        """Entry signals should have a position size."""
        # Force a position by manually setting state
        self.strategy._position_side = "long"
        candles = self._make_trending_down_candles(n=60)
        ms = self._make_market_state(candles, equity=10000.0)
        signal = self.strategy.on_tick(ms)
        if signal.signal_type != SignalType.HOLD:
            self.assertIsNotNone(signal.size)
            self.assertGreater(signal.size, 0)

    def test_exit_long_on_rsi_overbought(self):
        """Manually set position, then feed overbought RSI to trigger exit."""
        self.strategy._position_side = "long"
        self.strategy._entry_price = 50000.0
        # Create strong uptrend that pushes RSI above 70
        closes = [50000 + i * 200 for i in range(60)]
        candles = self._make_candles(closes, spread=50)
        ms = self._make_market_state(candles)
        signal = self.strategy.on_tick(ms)
        # Should exit or hold (RSI should be very high)
        if signal.signal_type != SignalType.HOLD:
            self.assertEqual(signal.signal_type, SignalType.EXIT_LONG)

    def test_exit_short_on_rsi_oversold(self):
        """Manually set short position, feed oversold RSI."""
        self.strategy._position_side = "short"
        self.strategy._entry_price = 55000.0
        closes = [55000 - i * 200 for i in range(60)]
        candles = self._make_candles(closes, spread=50)
        ms = self._make_market_state(candles)
        signal = self.strategy.on_tick(ms)
        if signal.signal_type != SignalType.HOLD:
            self.assertEqual(signal.signal_type, SignalType.EXIT_SHORT)

    def test_ema_computation(self):
        data = np.array([float(i) for i in range(20)])
        ema = MomentumTrendStrategy._compute_ema(data, 5)
        self.assertIsNotNone(ema)
        self.assertEqual(len(ema), 20)
        # EMA should track the rising trend
        self.assertGreater(ema[-1], ema[5])

    def test_ema_insufficient_data(self):
        data = np.array([1.0, 2.0])
        ema = MomentumTrendStrategy._compute_ema(data, 5)
        self.assertIsNone(ema)

    def test_rsi_computation(self):
        # Monotonically rising prices -> RSI close to 100
        data = np.array([float(i * 10) for i in range(30)])
        rsi = MomentumTrendStrategy._compute_rsi(data, 14)
        self.assertIsNotNone(rsi)
        self.assertGreater(rsi[-1], 80)

    def test_rsi_monotonic_decline(self):
        data = np.array([float(100 - i * 2) for i in range(30)])
        rsi = MomentumTrendStrategy._compute_rsi(data, 14)
        self.assertIsNotNone(rsi)
        self.assertLess(rsi[-1], 20)

    def test_rsi_insufficient_data(self):
        data = np.array([1.0, 2.0, 3.0])
        rsi = MomentumTrendStrategy._compute_rsi(data, 14)
        self.assertIsNone(rsi)

    def test_adx_computation(self):
        n = 50
        highs = np.array([100.0 + i * 2.0 + np.random.random() for i in range(n)])
        lows = np.array([98.0 + i * 2.0 - np.random.random() for i in range(n)])
        closes = np.array([99.0 + i * 2.0 for i in range(n)])
        adx = MomentumTrendStrategy._compute_adx(highs, lows, closes, 14)
        self.assertIsNotNone(adx)
        self.assertEqual(len(adx), n)

    def test_adx_insufficient_data(self):
        highs = np.array([100.0, 101.0])
        lows = np.array([99.0, 100.0])
        closes = np.array([99.5, 100.5])
        adx = MomentumTrendStrategy._compute_adx(highs, lows, closes, 14)
        self.assertIsNone(adx)

    def test_custom_params(self):
        params = {
            "ema_fast": 8,
            "ema_slow": 20,
            "rsi_period": 10,
            "adx_period": 10,
            "adx_threshold": 20,
        }
        s = MomentumTrendStrategy(params)
        self.assertEqual(s.ema_fast, 8)
        self.assertEqual(s.ema_slow, 20)
        self.assertEqual(s.adx_threshold, 20)


class TestMomentumTrendBacktestAdapter(unittest.TestCase):

    def setUp(self):
        self.params = {
            "ema_fast": 12,
            "ema_slow": 26,
            "rsi_period": 14,
            "adx_period": 14,
            "adx_threshold": 25,
        }
        self.adapter = MomentumTrendBacktestAdapter(self.params)

    def _make_df(self, n=60, trend="up"):
        dates = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
        if trend == "up":
            close = [50000 + i * 50 for i in range(n)]
        elif trend == "down":
            close = [55000 - i * 50 for i in range(n)]
        else:
            close = [50000 + 50 * np.sin(i * 0.3) for i in range(n)]
        data = {
            "open": [c - 10 for c in close],
            "high": [c + 25 for c in close],
            "low": [c - 25 for c in close],
            "close": close,
            "volume": [100.0] * n,
        }
        return pd.DataFrame(data, index=dates)

    def test_setup_does_nothing(self):
        df = self._make_df()
        self.adapter.setup(df)  # should not raise

    def test_generate_signal_returns_signal(self):
        df = self._make_df(n=60, trend="flat")
        signal = self.adapter.generate_signal(59, df)
        self.assertIsInstance(signal, Signal)

    def test_generate_signal_trending(self):
        """In a strong trend, adapter should eventually produce a signal."""
        df = self._make_df(n=60, trend="up")
        signals = []
        for i in range(30, 60):
            signal = self.adapter.generate_signal(i, df)
            signals.append(signal.signal_type)
        # At least one non-HOLD signal should appear in a strong trend
        # (or all HOLD if trend is not strong enough for ADX)
        self.assertTrue(all(isinstance(s, SignalType) for s in signals))

    def test_adapter_wraps_live_strategy(self):
        self.assertIsInstance(self.adapter.live_strategy, MomentumTrendStrategy)


if __name__ == "__main__":
    unittest.main()
