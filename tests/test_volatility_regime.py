#!/usr/bin/env python3
"""Tests for volatility regime strategy."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from core.models import Signal
from core.types import SignalType
from strategy.volatility_regime import (
    VolatilityRegimeStrategy, VolatilityRegimeBacktestAdapter,
    LOW_VOL, NORMAL, HIGH_VOL, EXPANDING, CONTRACTING,
)
from strategy.live_strategy import MarketState


class TestVolatilityRegimeStrategy(unittest.TestCase):

    def setUp(self):
        self.params = {
            "atr_fast": 7,
            "atr_slow": 21,
            "bb_period": 20,
            "bb_std": 2.0,
            "kc_period": 20,
            "kc_atr_mult": 1.5,
            "expansion_threshold": 1.5,
            "contraction_threshold": 0.7,
            "rsi_period": 14,
            "position_size_pct": 0.02,
        }
        self.strategy = VolatilityRegimeStrategy(self.params)
        self.base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)

    def _make_candles(self, closes, spread=100.0):
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

    def _make_stable_candles(self, n=50, center=50000, spread=10):
        """Candles with very low volatility."""
        closes = [center + spread * np.sin(i * 0.1) for i in range(n)]
        return self._make_candles(closes, spread=spread * 0.5)

    def _make_expanding_vol_candles(self, n=50, start=50000):
        """Candles where recent vol is much higher than historical vol."""
        # First 30 bars: low vol, last 20 bars: high vol
        closes = []
        for i in range(30):
            closes.append(start + np.random.uniform(-10, 10))
        for i in range(20):
            closes.append(start + np.random.uniform(-500, 500))
        return self._make_candles(closes, spread=200)

    # ---------------------------------------------------------------
    # Regime classification tests
    # ---------------------------------------------------------------

    def test_classify_expanding(self):
        """ATR ratio above expansion threshold -> EXPANDING."""
        regime = self.strategy._classify_vol_regime(
            atr_ratio=2.0, bb_width=0.05, bb_width_pctl=50)
        self.assertEqual(regime, EXPANDING)

    def test_classify_contracting(self):
        """ATR ratio below contraction threshold -> CONTRACTING."""
        regime = self.strategy._classify_vol_regime(
            atr_ratio=0.5, bb_width=0.02, bb_width_pctl=30)
        self.assertEqual(regime, CONTRACTING)

    def test_classify_low_vol(self):
        """Low BB width percentile -> LOW_VOL."""
        regime = self.strategy._classify_vol_regime(
            atr_ratio=1.0, bb_width=0.01, bb_width_pctl=10)
        self.assertEqual(regime, LOW_VOL)

    def test_classify_high_vol(self):
        """High BB width percentile -> HIGH_VOL."""
        regime = self.strategy._classify_vol_regime(
            atr_ratio=1.0, bb_width=0.1, bb_width_pctl=90)
        self.assertEqual(regime, HIGH_VOL)

    def test_classify_normal(self):
        """Moderate values -> NORMAL."""
        regime = self.strategy._classify_vol_regime(
            atr_ratio=1.0, bb_width=0.04, bb_width_pctl=50)
        self.assertEqual(regime, NORMAL)

    # ---------------------------------------------------------------
    # Entry/exit tests
    # ---------------------------------------------------------------

    def test_expansion_entry_long(self):
        """Price above KC upper with RSI < 70 -> ENTER_LONG."""
        signal = self.strategy._check_expansion_entry(
            price=51000, kc_upper=50500, kc_lower=49500,
            rsi=55, atr_ratio=2.0,
            timestamp=self.base_time, size=0.01)
        self.assertEqual(signal.signal_type, SignalType.ENTER_LONG)
        self.assertEqual(signal.metadata["entry_reason"], "kc_breakout_long")

    def test_expansion_entry_short(self):
        """Price below KC lower with RSI > 30 -> ENTER_SHORT."""
        signal = self.strategy._check_expansion_entry(
            price=49000, kc_upper=50500, kc_lower=49500,
            rsi=45, atr_ratio=2.0,
            timestamp=self.base_time, size=0.01)
        self.assertEqual(signal.signal_type, SignalType.ENTER_SHORT)
        self.assertEqual(signal.metadata["entry_reason"], "kc_breakout_short")

    def test_expansion_no_entry_rsi_overbought(self):
        """Price above KC upper but RSI > 70 -> HOLD."""
        signal = self.strategy._check_expansion_entry(
            price=51000, kc_upper=50500, kc_lower=49500,
            rsi=75, atr_ratio=2.0,
            timestamp=self.base_time, size=0.01)
        self.assertEqual(signal.signal_type, SignalType.HOLD)

    def test_contraction_entry_long(self):
        """Price below BB lower + RSI < 30 -> ENTER_LONG."""
        self.strategy._current_regime = CONTRACTING
        signal = self.strategy._check_contraction_entry(
            price=49000, bb_lower=49500, bb_upper=50500,
            rsi=25, timestamp=self.base_time, size=0.01)
        self.assertEqual(signal.signal_type, SignalType.ENTER_LONG)
        self.assertEqual(signal.metadata["entry_reason"], "bb_oversold_long")

    def test_contraction_entry_short(self):
        """Price above BB upper + RSI > 70 -> ENTER_SHORT."""
        self.strategy._current_regime = CONTRACTING
        signal = self.strategy._check_contraction_entry(
            price=51000, bb_lower=49500, bb_upper=50500,
            rsi=75, timestamp=self.base_time, size=0.01)
        self.assertEqual(signal.signal_type, SignalType.ENTER_SHORT)
        self.assertEqual(signal.metadata["entry_reason"], "bb_overbought_short")

    def test_contraction_no_entry_mid_rsi(self):
        """Price below BB but RSI not oversold -> HOLD."""
        self.strategy._current_regime = CONTRACTING
        signal = self.strategy._check_contraction_entry(
            price=49000, bb_lower=49500, bb_upper=50500,
            rsi=50, timestamp=self.base_time, size=0.01)
        self.assertEqual(signal.signal_type, SignalType.HOLD)

    def test_exit_on_regime_change(self):
        """Exit when regime changes from entry regime."""
        self.strategy._position_side = "long"
        self.strategy._entry_price = 50000
        self.strategy._entry_regime = EXPANDING
        self.strategy._current_regime = CONTRACTING
        signal = self.strategy._check_exit(
            price=50500, atr_ratio=0.5, rsi=50,
            timestamp=self.base_time, size=0.01)
        self.assertEqual(signal.signal_type, SignalType.EXIT_LONG)
        self.assertEqual(signal.metadata["exit_reason"], "regime_change")

    def test_exit_short_on_regime_change(self):
        """Exit short when regime changes."""
        self.strategy._position_side = "short"
        self.strategy._entry_price = 50000
        self.strategy._entry_regime = EXPANDING
        self.strategy._current_regime = LOW_VOL
        signal = self.strategy._check_exit(
            price=49500, atr_ratio=0.5, rsi=50,
            timestamp=self.base_time, size=0.01)
        self.assertEqual(signal.signal_type, SignalType.EXIT_SHORT)

    def test_hold_in_normal_regime(self):
        """Should HOLD when regime is NORMAL and no position."""
        candles = self._make_stable_candles(n=50)
        ms = self._make_market_state(candles)
        signal = self.strategy.on_tick(ms)
        self.assertEqual(signal.signal_type, SignalType.HOLD)

    def test_hold_insufficient_candles(self):
        """Should HOLD when not enough candles."""
        candles = self._make_candles([50000 + i for i in range(5)])
        ms = self._make_market_state(candles)
        signal = self.strategy.on_tick(ms)
        self.assertEqual(signal.signal_type, SignalType.HOLD)

    def test_hold_zero_price(self):
        candles = self._make_stable_candles(n=50)
        ms = self._make_market_state(candles, mark_price=0)
        signal = self.strategy.on_tick(ms)
        self.assertEqual(signal.signal_type, SignalType.HOLD)

    def test_position_state_tracking_entry(self):
        """Entering a position should update state."""
        self.strategy._check_expansion_entry(
            price=51000, kc_upper=50500, kc_lower=49500,
            rsi=50, atr_ratio=2.0,
            timestamp=self.base_time, size=0.01)
        self.assertEqual(self.strategy._position_side, "long")
        self.assertEqual(self.strategy._entry_price, 51000)
        self.assertEqual(self.strategy._entry_regime, EXPANDING)

    def test_position_state_tracking_exit(self):
        """Exiting should clear position state."""
        self.strategy._position_side = "long"
        self.strategy._entry_price = 50000
        self.strategy._entry_regime = EXPANDING
        self.strategy._current_regime = CONTRACTING
        self.strategy._check_exit(
            price=50500, atr_ratio=0.5, rsi=50,
            timestamp=self.base_time, size=0.01)
        self.assertIsNone(self.strategy._position_side)
        self.assertIsNone(self.strategy._entry_price)

    def test_params_from_config(self):
        custom_params = {"atr_fast": 5, "atr_slow": 30, "bb_std": 2.5}
        s = VolatilityRegimeStrategy(custom_params)
        self.assertEqual(s.atr_fast, 5)
        self.assertEqual(s.atr_slow, 30)
        self.assertEqual(s.bb_std, 2.5)


# ===================================================================
# TestVolatilityIndicators
# ===================================================================

class TestVolatilityIndicators(unittest.TestCase):

    def test_atr_computation(self):
        n = 30
        highs = np.array([100.0 + i * 0.5 for i in range(n)])
        lows = np.array([99.0 + i * 0.5 for i in range(n)])
        closes = np.array([99.5 + i * 0.5 for i in range(n)])
        atr = VolatilityRegimeStrategy._compute_atr(highs, lows, closes, 7)
        self.assertIsNotNone(atr)
        self.assertEqual(len(atr), n)
        # ATR should be positive
        self.assertGreater(atr[-1], 0)

    def test_atr_insufficient_data(self):
        highs = np.array([100.0, 101.0])
        lows = np.array([99.0, 100.0])
        closes = np.array([99.5, 100.5])
        atr = VolatilityRegimeStrategy._compute_atr(highs, lows, closes, 7)
        self.assertIsNone(atr)

    def test_keltner_channels(self):
        n = 30
        closes = np.array([100.0 + i * 0.1 for i in range(n)])
        highs = closes + 1.0
        lows = closes - 1.0
        upper, lower, mid = VolatilityRegimeStrategy._compute_keltner_channels(
            closes, highs, lows, 20, 1.5)
        self.assertIsNotNone(upper)
        self.assertIsNotNone(lower)
        self.assertEqual(len(upper), n)
        # Upper should be above mid, lower below
        self.assertGreater(upper[-1], mid[-1])
        self.assertLess(lower[-1], mid[-1])

    def test_keltner_insufficient_data(self):
        closes = np.array([100.0, 101.0])
        highs = closes + 1
        lows = closes - 1
        upper, lower, mid = VolatilityRegimeStrategy._compute_keltner_channels(
            closes, highs, lows, 20, 1.5)
        self.assertIsNone(upper)

    def test_bb_width(self):
        n = 30
        closes = np.array([100.0 + np.sin(i * 0.3) * 2 for i in range(n)])
        width = VolatilityRegimeStrategy._compute_bb_width(closes, 20, 2.0)
        self.assertIsNotNone(width)
        self.assertEqual(len(width), n)
        self.assertGreater(width[-1], 0)

    def test_bb_width_insufficient_data(self):
        closes = np.array([100.0, 101.0])
        width = VolatilityRegimeStrategy._compute_bb_width(closes, 20, 2.0)
        self.assertIsNone(width)

    def test_bollinger_bands(self):
        n = 30
        closes = np.array([100.0 + np.sin(i * 0.3) * 2 for i in range(n)])
        upper, lower, mid = VolatilityRegimeStrategy._compute_bollinger_bands(
            closes, 20, 2.0)
        self.assertIsNotNone(upper)
        self.assertGreater(upper[-1], mid[-1])
        self.assertLess(lower[-1], mid[-1])

    def test_rsi_computation(self):
        # Rising prices -> high RSI
        closes = np.array([float(i * 10) for i in range(30)])
        rsi = VolatilityRegimeStrategy._compute_rsi(closes, 14)
        self.assertIsNotNone(rsi)
        self.assertGreater(rsi[-1], 80)

    def test_rsi_declining(self):
        closes = np.array([float(100 - i * 2) for i in range(30)])
        rsi = VolatilityRegimeStrategy._compute_rsi(closes, 14)
        self.assertIsNotNone(rsi)
        self.assertLess(rsi[-1], 20)

    def test_rsi_insufficient_data(self):
        closes = np.array([1.0, 2.0, 3.0])
        rsi = VolatilityRegimeStrategy._compute_rsi(closes, 14)
        self.assertIsNone(rsi)

    def test_atr_flat_prices(self):
        """ATR with flat prices should have small values."""
        n = 30
        highs = np.array([100.5] * n)
        lows = np.array([99.5] * n)
        closes = np.array([100.0] * n)
        atr = VolatilityRegimeStrategy._compute_atr(highs, lows, closes, 7)
        self.assertIsNotNone(atr)
        self.assertAlmostEqual(atr[-1], 1.0, places=1)

    def test_percentile(self):
        history = [1.0, 2.0, 3.0, 4.0, 5.0]
        pctl = VolatilityRegimeStrategy._percentile(3.0, history)
        self.assertAlmostEqual(pctl, 40.0)  # 2 values below 3.0 out of 5

    def test_percentile_empty(self):
        pctl = VolatilityRegimeStrategy._percentile(5.0, [])
        self.assertAlmostEqual(pctl, 50.0)


# ===================================================================
# TestVolatilityRegimeBacktestAdapter
# ===================================================================

class TestVolatilityRegimeBacktestAdapter(unittest.TestCase):

    def setUp(self):
        self.params = {
            "atr_fast": 7,
            "atr_slow": 21,
            "bb_period": 20,
            "bb_std": 2.0,
            "kc_period": 20,
            "kc_atr_mult": 1.5,
            "rsi_period": 14,
        }
        self.adapter = VolatilityRegimeBacktestAdapter(self.params)

    def _make_df(self, n=60):
        dates = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
        close = [50000 + np.sin(i * 0.1) * 500 for i in range(n)]
        data = {
            "open": [c - 10 for c in close],
            "high": [c + 50 for c in close],
            "low": [c - 50 for c in close],
            "close": close,
            "volume": [100.0] * n,
        }
        return pd.DataFrame(data, index=dates)

    def test_setup_does_nothing(self):
        df = self._make_df()
        self.adapter.setup(df)  # should not raise

    def test_generate_signal_returns_signal(self):
        df = self._make_df(n=60)
        signal = self.adapter.generate_signal(59, df)
        self.assertIsInstance(signal, Signal)

    def test_generate_signal_type(self):
        df = self._make_df(n=60)
        signal = self.adapter.generate_signal(59, df)
        self.assertIn(signal.signal_type, list(SignalType))

    def test_adapter_wraps_live_strategy(self):
        self.assertIsInstance(self.adapter.live_strategy, VolatilityRegimeStrategy)

    def test_bar_to_market_state_conversion(self):
        """Adapter should pass correct data to live strategy."""
        df = self._make_df(n=60)
        # Should not raise
        for i in range(30, 60):
            signal = self.adapter.generate_signal(i, df)
            self.assertIsInstance(signal, Signal)


# ===================================================================
# TestVolatilityRegimeMetadata
# ===================================================================

class TestVolatilityRegimeMetadata(unittest.TestCase):

    def setUp(self):
        self.params = {
            "atr_fast": 7,
            "atr_slow": 21,
            "bb_period": 20,
            "bb_std": 2.0,
            "kc_period": 20,
            "kc_atr_mult": 1.5,
            "rsi_period": 14,
            "position_size_pct": 0.02,
        }
        self.strategy = VolatilityRegimeStrategy(self.params)

    def test_get_metadata(self):
        meta = self.strategy.get_metadata()
        self.assertEqual(meta["name"], "volatility_regime")
        self.assertEqual(meta["version"], "1.0")
        self.assertEqual(meta["category"], "volatility")
        self.assertIn("atr_fast", meta["params"])
        self.assertIn("bb_std", meta["params"])

    def test_state_round_trip(self):
        self.strategy._position_side = "long"
        self.strategy._entry_price = 50000.0
        self.strategy._current_regime = EXPANDING
        self.strategy._entry_regime = EXPANDING
        self.strategy._bb_width_history = [0.01, 0.02, 0.03]

        state = self.strategy.get_state()

        new_strategy = VolatilityRegimeStrategy(self.params)
        new_strategy.set_state(state)

        self.assertEqual(new_strategy._position_side, "long")
        self.assertEqual(new_strategy._entry_price, 50000.0)
        self.assertEqual(new_strategy._current_regime, EXPANDING)
        self.assertEqual(new_strategy._entry_regime, EXPANDING)
        self.assertEqual(len(new_strategy._bb_width_history), 3)

    def test_state_empty_default(self):
        state = self.strategy.get_state()
        self.assertIsNone(state["position_side"])
        self.assertIsNone(state["entry_price"])
        self.assertEqual(state["current_regime"], NORMAL)


if __name__ == "__main__":
    unittest.main()
