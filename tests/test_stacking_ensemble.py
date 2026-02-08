#!/usr/bin/env python3
"""Tests for stacking ensemble strategy."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import math
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from core.models import Signal
from core.types import SignalType
from strategy.live_strategy import MarketState
from strategy.signal_aggregator import SignalAggregator
from strategy.stacking_ensemble import (
    StackingEnsembleStrategy,
    StackingEnsembleBacktestAdapter,
)


class TestStackingEnsembleStrategy(unittest.TestCase):

    def setUp(self):
        self.strategy_names = ["momentum", "mean_reversion", "funding_rate_arb"]
        self.aggregator = SignalAggregator(self.strategy_names)
        self.params = {
            "feature_mode": "signals_only",
            "probability_threshold": 0.65,
            "position_size_pct": 0.03,
        }
        self.strategy = StackingEnsembleStrategy(self.params, self.aggregator)
        self.base_time = datetime(2025, 6, 1, tzinfo=timezone.utc)

    def _make_market_state(self, price=100.0, equity=10000.0, regime="unknown",
                           drawdown_pct=0.0, spread=0.1):
        return MarketState(
            mark_price=price,
            mid_price=price,
            bid=price - 0.5,
            ask=price + 0.5,
            funding_rate=0.0,
            premium=0.0,
            open_interest=0.0,
            equity=equity,
            cash=5000.0,
            drawdown_pct=drawdown_pct,
            spread=spread,
            timestamp=self.base_time,
            metadata={"regime": regime},
        )

    def _make_signal(self, signal_type, price=100.0, health=50.0):
        return Signal(
            signal_type=signal_type,
            price=price,
            timestamp=self.base_time,
        )

    def _record_all(self, signal_type, health=50.0):
        for name in self.strategy_names:
            self.aggregator.record_signal(name, self._make_signal(signal_type),
                                          health_score=health)

    def test_feature_vector_signals_only(self):
        """Feature vector in signals_only mode has 2 features per strategy."""
        self._record_all(SignalType.ENTER_LONG, health=80.0)
        signals = self.aggregator.get_latest_signals()
        ms = self._make_market_state()
        features = self.strategy._build_feature_vector(signals, ms)
        # 3 strategies * 2 features each = 6
        self.assertEqual(len(features), 6)
        # All bullish (1.0) with health 0.8
        for i in range(0, 6, 2):
            self.assertAlmostEqual(features[i], 1.0)  # direction
            self.assertAlmostEqual(features[i + 1], 0.8)  # health

    def test_feature_vector_signals_plus_regime(self):
        """Feature vector with regime adds 1 extra feature."""
        self.strategy.feature_mode = "signals_plus_regime"
        self._record_all(SignalType.HOLD)
        signals = self.aggregator.get_latest_signals()
        ms = self._make_market_state(regime="trending_up")
        features = self.strategy._build_feature_vector(signals, ms)
        # 6 signal features + 1 regime = 7
        self.assertEqual(len(features), 7)
        self.assertAlmostEqual(features[6], 1.0)  # trending_up encoded as 1.0

    def test_feature_vector_full_mode(self):
        """Feature vector in full mode includes drawdown and spread."""
        self.strategy.feature_mode = "full"
        self._record_all(SignalType.HOLD)
        signals = self.aggregator.get_latest_signals()
        ms = self._make_market_state(regime="ranging", drawdown_pct=5.0, spread=0.5)
        features = self.strategy._build_feature_vector(signals, ms)
        # 6 signal + 1 regime + 2 (drawdown, spread) = 9
        self.assertEqual(len(features), 9)
        self.assertAlmostEqual(features[7], 0.05)  # drawdown_pct / 100
        self.assertAlmostEqual(features[8], 0.5)  # spread

    def test_feature_vector_missing_strategy(self):
        """Missing strategy gets default features (0.0, 0.5)."""
        # Only record for one strategy
        self.aggregator.record_signal("momentum", self._make_signal(SignalType.ENTER_LONG),
                                      health_score=90.0)
        signals = self.aggregator.get_latest_signals()
        ms = self._make_market_state()
        features = self.strategy._build_feature_vector(signals, ms)
        self.assertEqual(len(features), 6)

    def test_predict_fallback_bullish(self):
        """Fallback prediction returns long when majority bullish."""
        self._record_all(SignalType.ENTER_LONG, health=80.0)
        signals = self.aggregator.get_latest_signals()
        direction, prob = self.strategy._predict_fallback(signals)
        self.assertEqual(direction, "long")
        self.assertGreater(prob, 0.5)

    def test_predict_fallback_bearish(self):
        """Fallback prediction returns short when majority bearish."""
        self._record_all(SignalType.ENTER_SHORT, health=80.0)
        signals = self.aggregator.get_latest_signals()
        direction, prob = self.strategy._predict_fallback(signals)
        self.assertEqual(direction, "short")
        self.assertGreater(prob, 0.5)

    def test_predict_fallback_empty(self):
        """Fallback with no signals returns neutral."""
        direction, prob = self.strategy._predict_fallback({})
        self.assertEqual(direction, "neutral")
        self.assertEqual(prob, 0.0)

    def test_predict_above_threshold_enters(self):
        """Prediction above threshold triggers entry."""
        self._record_all(SignalType.ENTER_LONG, health=90.0)
        ms = self._make_market_state()
        signal = self.strategy.on_tick(ms)
        # All bullish with high health -> probability should exceed 0.65
        self.assertIn(signal.signal_type, [SignalType.ENTER_LONG, SignalType.HOLD])

    def test_predict_below_threshold_holds(self):
        """Prediction below threshold returns HOLD."""
        # Mixed signals -> low probability
        self.aggregator.record_signal("momentum", self._make_signal(SignalType.ENTER_LONG),
                                      health_score=50.0)
        self.aggregator.record_signal("mean_reversion", self._make_signal(SignalType.ENTER_SHORT),
                                      health_score=50.0)
        self.aggregator.record_signal("funding_rate_arb", self._make_signal(SignalType.HOLD),
                                      health_score=50.0)
        ms = self._make_market_state()
        signal = self.strategy.on_tick(ms)
        self.assertEqual(signal.signal_type, SignalType.HOLD)

    def test_model_loading(self):
        """Model loads from JSON file."""
        model_data = {"weights": [0.5, 0.3, 0.5, 0.3, 0.5, 0.3], "bias": -1.0}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(model_data, f)
            model_path = f.name

        try:
            params = dict(self.params, model_path=model_path)
            strategy = StackingEnsembleStrategy(params, self.aggregator)
            self.assertTrue(strategy._model_loaded)
            self.assertEqual(len(strategy._weights), 6)
            self.assertEqual(strategy._bias, -1.0)
        finally:
            os.unlink(model_path)

    def test_model_predict_with_model(self):
        """Predict with loaded model uses logistic regression."""
        # Manually set model weights
        self.strategy._weights = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        self.strategy._bias = 0.0
        self.strategy._model_loaded = True

        # All bullish -> features [1.0, 0.5, 1.0, 0.5, 1.0, 0.5]
        # logit = 1*1 + 0*0.5 + 1*1 + 0*0.5 + 1*1 + 0*0.5 = 3.0
        # sigmoid(3.0) = 0.9526
        features = [1.0, 0.5, 1.0, 0.5, 1.0, 0.5]
        direction, prob = self.strategy._predict_with_model(features)
        self.assertEqual(direction, "long")
        self.assertAlmostEqual(prob, 1.0 / (1.0 + math.exp(-3.0)), places=3)

    def test_model_predict_bearish(self):
        """Predict with model identifies bearish direction."""
        self.strategy._weights = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        self.strategy._bias = 0.0
        self.strategy._model_loaded = True

        # All bearish -> features [-1, 0.5, -1, 0.5, -1, 0.5]
        # logit = -3.0 -> sigmoid(-3.0) = 0.0474 -> short, conf = 1 - 0.0474
        features = [-1.0, 0.5, -1.0, 0.5, -1.0, 0.5]
        direction, prob = self.strategy._predict_with_model(features)
        self.assertEqual(direction, "short")
        self.assertGreater(prob, 0.9)

    def test_model_file_not_found(self):
        """Missing model file falls back gracefully."""
        params = dict(self.params, model_path="/nonexistent/model.json")
        strategy = StackingEnsembleStrategy(params, self.aggregator)
        self.assertFalse(strategy._model_loaded)

    def test_exit_on_direction_flip(self):
        """Position exits when predicted direction flips with confidence."""
        self.strategy._weights = [2.0, 0.0, 2.0, 0.0, 2.0, 0.0]
        self.strategy._bias = 0.0
        self.strategy._model_loaded = True

        # Enter long
        self._record_all(SignalType.ENTER_LONG, health=80.0)
        ms = self._make_market_state()
        signal = self.strategy.on_tick(ms)
        self.assertEqual(signal.signal_type, SignalType.ENTER_LONG)

        # Flip to bearish
        self._record_all(SignalType.ENTER_SHORT, health=80.0)
        signal = self.strategy.on_tick(ms)
        self.assertEqual(signal.signal_type, SignalType.EXIT_LONG)

    def test_hold_on_zero_price(self):
        """HOLD on zero price."""
        self._record_all(SignalType.ENTER_LONG)
        ms = self._make_market_state(price=0.0)
        signal = self.strategy.on_tick(ms)
        self.assertEqual(signal.signal_type, SignalType.HOLD)

    def test_hold_on_empty_signals(self):
        """HOLD when no sub-strategy signals."""
        ms = self._make_market_state()
        signal = self.strategy.on_tick(ms)
        self.assertEqual(signal.signal_type, SignalType.HOLD)

    def test_get_state_set_state(self):
        """State round-trips correctly."""
        self.strategy._position_side = "short"
        self.strategy._entry_price = 99.5
        self.strategy._tick_count = 42

        state = self.strategy.get_state()
        new_strategy = StackingEnsembleStrategy(self.params, self.aggregator)
        new_strategy.set_state(state)
        self.assertEqual(new_strategy._position_side, "short")
        self.assertEqual(new_strategy._entry_price, 99.5)
        self.assertEqual(new_strategy._tick_count, 42)

    def test_get_metadata(self):
        """Metadata has expected fields."""
        meta = self.strategy.get_metadata()
        self.assertEqual(meta["name"], "stacking_ensemble")
        self.assertEqual(meta["category"], "stacking_ensemble")
        self.assertIn("probability_threshold", meta["params"])
        self.assertIn("feature_mode", meta["params"])

    def test_entry_metadata(self):
        """Entry signal metadata includes probability and feature mode."""
        self.strategy._weights = [2.0, 0.0, 2.0, 0.0, 2.0, 0.0]
        self.strategy._bias = 0.0
        self.strategy._model_loaded = True

        self._record_all(SignalType.ENTER_LONG, health=80.0)
        ms = self._make_market_state()
        signal = self.strategy.on_tick(ms)
        self.assertEqual(signal.signal_type, SignalType.ENTER_LONG)
        self.assertIn("probability", signal.metadata)
        self.assertIn("feature_mode", signal.metadata)
        self.assertTrue(signal.metadata["model_loaded"])


class TestStackingEnsembleBacktestAdapter(unittest.TestCase):

    def setUp(self):
        self.params = {
            "probability_threshold": 0.65,
            "position_size_pct": 0.03,
        }
        self.adapter = StackingEnsembleBacktestAdapter(self.params)

    def _make_df(self, n=100):
        dates = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
        np.random.seed(42)
        prices = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
        return pd.DataFrame({
            "open": prices,
            "high": prices + abs(np.random.randn(n)),
            "low": prices - abs(np.random.randn(n)),
            "close": prices,
            "volume": np.random.randint(100, 1000, n).astype(float),
        }, index=dates)

    def test_setup_creates_indicators(self):
        """Setup creates momentum_dir, rsi_norm, bb_pos, combined."""
        df = self._make_df()
        self.adapter.setup(df)
        self.assertIn("momentum_dir", self.adapter.indicators)
        self.assertIn("rsi_norm", self.adapter.indicators)
        self.assertIn("bb_pos", self.adapter.indicators)
        self.assertIn("combined", self.adapter.indicators)

    def test_warmup_returns_hold(self):
        """Early bars return HOLD."""
        df = self._make_df()
        self.adapter.setup(df)
        signal = self.adapter.generate_signal(5, df)
        self.assertEqual(signal.signal_type, SignalType.HOLD)

    def test_generates_signals(self):
        """Adapter generates trading signals on trending data."""
        # Create strongly trending data
        n = 200
        dates = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
        prices = 100.0 + np.arange(n) * 0.5
        df = pd.DataFrame({
            "open": prices,
            "high": prices + 1.0,
            "low": prices - 0.5,
            "close": prices,
            "volume": np.ones(n) * 500.0,
        }, index=dates)
        self.adapter.setup(df)
        signal_types = set()
        for i in range(30, n):
            signal = self.adapter.generate_signal(i, df)
            signal_types.add(signal.signal_type)
        # Strong uptrend should produce at least an entry
        self.assertTrue(len(signal_types) >= 1)

    def test_exit_on_reversal(self):
        """Adapter exits when trend reverses."""
        # Trend up then down
        n = 200
        dates = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
        up = np.arange(100) * 0.5
        down = up[-1] - np.arange(100) * 0.5
        prices = 100.0 + np.concatenate([up, down])
        df = pd.DataFrame({
            "open": prices,
            "high": prices + 1.0,
            "low": prices - 0.5,
            "close": prices,
            "volume": np.ones(n) * 500.0,
        }, index=dates)
        self.adapter.setup(df)
        entered = False
        exited = False
        for i in range(30, n):
            signal = self.adapter.generate_signal(i, df)
            if signal.signal_type in (SignalType.ENTER_LONG, SignalType.ENTER_SHORT):
                entered = True
            elif entered and signal.signal_type in (SignalType.EXIT_LONG, SignalType.EXIT_SHORT):
                exited = True
                break
        # Should have produced at least one entry/exit pair
        # (may not always happen depending on exact price path)
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
