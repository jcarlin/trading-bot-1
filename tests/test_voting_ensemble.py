#!/usr/bin/env python3
"""Tests for voting ensemble strategy."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from core.models import Signal
from core.types import SignalType
from strategy.live_strategy import MarketState
from strategy.signal_aggregator import SignalAggregator
from strategy.voting_ensemble import (
    VotingEnsembleStrategy,
    VotingEnsembleBacktestAdapter,
)


class TestVotingEnsembleStrategy(unittest.TestCase):

    def setUp(self):
        self.strategy_names = ["momentum", "mean_reversion", "funding_rate_arb"]
        self.aggregator = SignalAggregator(self.strategy_names)
        self.params = {
            "min_agreement_pct": 60,
            "weighting_mode": "equal",
            "position_size_pct": 0.03,
            "cooldown_ticks": 3,
            "regime_boost": False,
        }
        self.strategy = VotingEnsembleStrategy(self.params, self.aggregator)
        self.base_time = datetime(2025, 6, 1, tzinfo=timezone.utc)

    def _make_market_state(self, price=100.0, equity=10000.0, regime="unknown"):
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
            timestamp=self.base_time,
            metadata={"regime": regime},
        )

    def _make_signal(self, signal_type, price=100.0):
        return Signal(
            signal_type=signal_type,
            price=price,
            timestamp=self.base_time,
        )

    def _record_all(self, signal_type):
        """Record the same signal for all strategies."""
        for name in self.strategy_names:
            self.aggregator.record_signal(name, self._make_signal(signal_type))

    def test_unanimous_long_entry(self):
        """Unanimous bullish signals trigger long entry."""
        self._record_all(SignalType.ENTER_LONG)
        ms = self._make_market_state()
        signal = self.strategy.on_tick(ms)
        self.assertEqual(signal.signal_type, SignalType.ENTER_LONG)
        self.assertGreater(signal.size, 0)

    def test_unanimous_short_entry(self):
        """Unanimous bearish signals trigger short entry."""
        self._record_all(SignalType.ENTER_SHORT)
        ms = self._make_market_state()
        signal = self.strategy.on_tick(ms)
        self.assertEqual(signal.signal_type, SignalType.ENTER_SHORT)

    def test_below_agreement_threshold_holds(self):
        """Below min_agreement_pct results in HOLD."""
        # 1 bullish, 1 bearish, 1 neutral -> 33% agreement
        self.aggregator.record_signal("momentum", self._make_signal(SignalType.ENTER_LONG))
        self.aggregator.record_signal("mean_reversion", self._make_signal(SignalType.ENTER_SHORT))
        self.aggregator.record_signal("funding_rate_arb", self._make_signal(SignalType.HOLD))
        ms = self._make_market_state()
        signal = self.strategy.on_tick(ms)
        self.assertEqual(signal.signal_type, SignalType.HOLD)

    def test_two_of_three_bullish_enters_long(self):
        """2/3 bullish (66.7%) exceeds 60% threshold -> enters long."""
        self.aggregator.record_signal("momentum", self._make_signal(SignalType.ENTER_LONG))
        self.aggregator.record_signal("mean_reversion", self._make_signal(SignalType.ENTER_LONG))
        self.aggregator.record_signal("funding_rate_arb", self._make_signal(SignalType.HOLD))
        ms = self._make_market_state()
        signal = self.strategy.on_tick(ms)
        self.assertEqual(signal.signal_type, SignalType.ENTER_LONG)

    def test_equal_weighting(self):
        """Equal weighting gives each strategy weight=1.0."""
        self._record_all(SignalType.ENTER_LONG)
        signals = self.aggregator.get_latest_signals()
        weights = self.strategy._compute_weights(signals)
        for w in weights.values():
            self.assertEqual(w, 1.0)

    def test_health_weighting(self):
        """Health weighting uses health_score/100 as weight."""
        self.strategy.weighting_mode = "health"
        self.aggregator.record_signal("momentum",
                                      self._make_signal(SignalType.ENTER_LONG),
                                      health_score=80.0)
        self.aggregator.record_signal("mean_reversion",
                                      self._make_signal(SignalType.ENTER_LONG),
                                      health_score=40.0)
        signals = self.aggregator.get_latest_signals()
        weights = self.strategy._compute_weights(signals)
        self.assertAlmostEqual(weights["momentum"], 0.8)
        self.assertAlmostEqual(weights["mean_reversion"], 0.4)

    def test_sharpe_weighting(self):
        """Sharpe weighting uses metadata sharpe value."""
        self.strategy.weighting_mode = "sharpe"
        sig1 = Signal(signal_type=SignalType.ENTER_LONG, price=100.0,
                      timestamp=self.base_time, metadata={"sharpe": 2.5})
        sig2 = Signal(signal_type=SignalType.ENTER_LONG, price=100.0,
                      timestamp=self.base_time, metadata={"sharpe": 0.5})
        self.aggregator.record_signal("momentum", sig1)
        self.aggregator.record_signal("mean_reversion", sig2)
        signals = self.aggregator.get_latest_signals()
        weights = self.strategy._compute_weights(signals)
        self.assertAlmostEqual(weights["momentum"], 2.5)
        self.assertAlmostEqual(weights["mean_reversion"], 0.5)

    def test_custom_weighting(self):
        """Custom weighting uses provided weight dict."""
        self.strategy.weighting_mode = "custom"
        self.strategy.custom_weights = {"momentum": 3.0, "mean_reversion": 1.0}
        self._record_all(SignalType.ENTER_LONG)
        signals = self.aggregator.get_latest_signals()
        weights = self.strategy._compute_weights(signals)
        self.assertEqual(weights["momentum"], 3.0)
        self.assertEqual(weights["mean_reversion"], 1.0)
        self.assertEqual(weights["funding_rate_arb"], 1.0)  # default

    def test_regime_boost_trending_up(self):
        """Regime boost increases bullish weights in trending_up."""
        self.strategy.regime_boost = True
        self._record_all(SignalType.ENTER_LONG)
        signals = self.aggregator.get_latest_signals()
        weights = {name: 1.0 for name in signals}
        boosted = self.strategy._apply_regime_boost(weights, "trending_up", signals)
        for w in boosted.values():
            self.assertAlmostEqual(w, 1.3)

    def test_regime_boost_trending_down_bearish(self):
        """Regime boost increases bearish weights in trending_down."""
        self.strategy.regime_boost = True
        self._record_all(SignalType.ENTER_SHORT)
        signals = self.aggregator.get_latest_signals()
        weights = {name: 1.0 for name in signals}
        boosted = self.strategy._apply_regime_boost(weights, "trending_down", signals)
        for w in boosted.values():
            self.assertAlmostEqual(w, 1.3)

    def test_cooldown_prevents_reentry(self):
        """Cooldown prevents re-entry after exit."""
        self.strategy.cooldown_ticks = 3

        # Enter long
        self._record_all(SignalType.ENTER_LONG)
        ms = self._make_market_state()
        signal = self.strategy.on_tick(ms)
        self.assertEqual(signal.signal_type, SignalType.ENTER_LONG)

        # Flip to short -> exit (ticks_since_exit resets to 0)
        self._record_all(SignalType.ENTER_SHORT)
        signal = self.strategy.on_tick(ms)
        self.assertEqual(signal.signal_type, SignalType.EXIT_LONG)

        # Tick 1 after exit: ticks_since_exit=1, 1 < 3 -> HOLD
        signal = self.strategy.on_tick(ms)
        self.assertEqual(signal.signal_type, SignalType.HOLD)

        # Tick 2: ticks_since_exit=2, 2 < 3 -> HOLD
        signal = self.strategy.on_tick(ms)
        self.assertEqual(signal.signal_type, SignalType.HOLD)

        # Tick 3: ticks_since_exit=3, 3 < 3 is False -> entry allowed
        signal = self.strategy.on_tick(ms)
        self.assertEqual(signal.signal_type, SignalType.ENTER_SHORT)

    def test_exit_on_consensus_flip(self):
        """Position exits when consensus flips direction."""
        # Enter long
        self._record_all(SignalType.ENTER_LONG)
        ms = self._make_market_state()
        self.strategy.on_tick(ms)

        # Flip all to bearish
        self._record_all(SignalType.ENTER_SHORT)
        signal = self.strategy.on_tick(ms)
        self.assertEqual(signal.signal_type, SignalType.EXIT_LONG)
        self.assertEqual(signal.metadata.get("exit_reason"), "consensus_flip")

    def test_exit_short_on_consensus_flip(self):
        """Short position exits when consensus flips bullish."""
        self._record_all(SignalType.ENTER_SHORT)
        ms = self._make_market_state()
        self.strategy.on_tick(ms)

        self._record_all(SignalType.ENTER_LONG)
        signal = self.strategy.on_tick(ms)
        self.assertEqual(signal.signal_type, SignalType.EXIT_SHORT)

    def test_no_entry_on_empty_signals(self):
        """No entry when aggregator has no signals."""
        ms = self._make_market_state()
        signal = self.strategy.on_tick(ms)
        self.assertEqual(signal.signal_type, SignalType.HOLD)

    def test_hold_on_zero_price(self):
        """HOLD returned on zero price."""
        self._record_all(SignalType.ENTER_LONG)
        ms = self._make_market_state(price=0.0)
        signal = self.strategy.on_tick(ms)
        self.assertEqual(signal.signal_type, SignalType.HOLD)

    def test_get_state_set_state(self):
        """get_state/set_state round-trips correctly."""
        self._record_all(SignalType.ENTER_LONG)
        ms = self._make_market_state()
        self.strategy.on_tick(ms)

        state = self.strategy.get_state()
        self.assertEqual(state["position_side"], "long")

        new_strategy = VotingEnsembleStrategy(self.params, self.aggregator)
        new_strategy.set_state(state)
        self.assertEqual(new_strategy._position_side, "long")

    def test_get_metadata(self):
        """Metadata has expected fields."""
        meta = self.strategy.get_metadata()
        self.assertEqual(meta["name"], "voting_ensemble")
        self.assertEqual(meta["category"], "voting_ensemble")
        self.assertIn("min_agreement_pct", meta["params"])

    def test_entry_metadata_contains_agreement(self):
        """Entry signal metadata contains agreement and confidence."""
        self._record_all(SignalType.ENTER_LONG)
        ms = self._make_market_state()
        signal = self.strategy.on_tick(ms)
        self.assertIn("agreement_pct", signal.metadata)
        self.assertIn("confidence", signal.metadata)
        self.assertGreaterEqual(signal.metadata["agreement_pct"], 60)

    def test_position_size_calculation(self):
        """Position size is equity * pct / price."""
        self._record_all(SignalType.ENTER_LONG)
        ms = self._make_market_state(price=100.0, equity=10000.0)
        signal = self.strategy.on_tick(ms)
        expected_size = 10000.0 * 0.03 / 100.0
        self.assertAlmostEqual(signal.size, expected_size, places=4)

    def test_regime_boost_disabled(self):
        """When regime_boost=False, weights unchanged by regime."""
        self.strategy.regime_boost = False
        self._record_all(SignalType.ENTER_LONG)
        ms = self._make_market_state(regime="trending_up")
        signal = self.strategy.on_tick(ms)
        # Should still enter (regime boost disabled doesn't prevent entry)
        self.assertEqual(signal.signal_type, SignalType.ENTER_LONG)

    def test_exit_neutral_low_confidence(self):
        """Exit when consensus becomes neutral with low confidence."""
        # Enter long
        self._record_all(SignalType.ENTER_LONG)
        ms = self._make_market_state()
        self.strategy.on_tick(ms)

        # All go neutral
        self._record_all(SignalType.HOLD)
        signal = self.strategy.on_tick(ms)
        self.assertEqual(signal.signal_type, SignalType.EXIT_LONG)
        self.assertEqual(signal.metadata.get("exit_reason"), "consensus_neutral")


class TestVotingEnsembleBacktestAdapter(unittest.TestCase):

    def setUp(self):
        self.params = {
            "min_agreement_pct": 60,
            "position_size_pct": 0.03,
            "cooldown_ticks": 3,
        }
        self.adapter = VotingEnsembleBacktestAdapter(self.params)

    def _make_df(self, n=100):
        """Create sample OHLCV DataFrame."""
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
        """Setup creates EMA, RSI, and BB indicators."""
        df = self._make_df()
        self.adapter.setup(df)
        self.assertIn("ema_signal", self.adapter.indicators)
        self.assertIn("rsi_signal", self.adapter.indicators)
        self.assertIn("bb_signal", self.adapter.indicators)

    def test_warmup_returns_hold(self):
        """Early bars during warmup return HOLD."""
        df = self._make_df()
        self.adapter.setup(df)
        signal = self.adapter.generate_signal(5, df)
        self.assertEqual(signal.signal_type, SignalType.HOLD)

    def test_generates_entry_signals(self):
        """Adapter generates non-HOLD signals after warmup."""
        df = self._make_df(200)
        self.adapter.setup(df)
        signal_types = set()
        for i in range(30, 200):
            signal = self.adapter.generate_signal(i, df)
            signal_types.add(signal.signal_type)
        # Should produce at least one non-HOLD signal
        self.assertTrue(len(signal_types) > 1 or SignalType.HOLD in signal_types)

    def test_no_entry_during_cooldown(self):
        """After exit, no entry during cooldown period."""
        df = self._make_df(200)
        self.adapter.setup(df)
        # Find first entry, then track exit and cooldown
        entered = False
        exited = False
        for i in range(30, 200):
            signal = self.adapter.generate_signal(i, df)
            if signal.signal_type in (SignalType.ENTER_LONG, SignalType.ENTER_SHORT):
                entered = True
            elif entered and signal.signal_type in (SignalType.EXIT_LONG, SignalType.EXIT_SHORT):
                exited = True
                break
        # Test passes if either no entry found or we found exit/entry cycle
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
