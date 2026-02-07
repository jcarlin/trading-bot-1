#!/usr/bin/env python3
"""Tests for the funding rate arbitrage strategy."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from datetime import datetime, timedelta, timezone

from core.models import Signal
from core.types import SignalType
from strategy.funding_rate_arb import FundingRateArbStrategy
from strategy.live_strategy import MarketState


class TestFundingRateArbStrategy(unittest.TestCase):

    def setUp(self):
        self.params = {
            "entry_threshold": 0.0001,
            "exit_threshold": 0.00005,
            "max_hold_periods": 24,
            "position_size_pct": 0.03,
            "min_open_interest": 1000000,
            "cooldown_periods": 2,
        }
        self.strategy = FundingRateArbStrategy(self.params)
        self.base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)

    def _make_market_state(self, funding_rate=0.0, mark_price=50000.0,
                           open_interest=5000000.0, equity=10000.0,
                           timestamp=None, position=None):
        return MarketState(
            mark_price=mark_price,
            mid_price=mark_price,
            bid=mark_price - 0.5,
            ask=mark_price + 0.5,
            funding_rate=funding_rate,
            premium=0.0,
            open_interest=open_interest,
            equity=equity,
            cash=equity * 0.5,
            timestamp=timestamp or self.base_time,
            position=position,
        )

    def test_enter_long_negative_funding(self):
        ms = self._make_market_state(funding_rate=-0.0005)
        signal = self.strategy.on_tick(ms)
        self.assertEqual(signal.signal_type, SignalType.ENTER_LONG)
        self.assertEqual(signal.price, 50000.0)
        self.assertIsNotNone(signal.size)
        self.assertGreater(signal.size, 0)

    def test_enter_short_positive_funding(self):
        ms = self._make_market_state(funding_rate=0.0005)
        signal = self.strategy.on_tick(ms)
        self.assertEqual(signal.signal_type, SignalType.ENTER_SHORT)

    def test_hold_below_threshold(self):
        ms = self._make_market_state(funding_rate=0.00003)
        signal = self.strategy.on_tick(ms)
        self.assertEqual(signal.signal_type, SignalType.HOLD)

    def test_hold_negative_below_threshold(self):
        ms = self._make_market_state(funding_rate=-0.00003)
        signal = self.strategy.on_tick(ms)
        self.assertEqual(signal.signal_type, SignalType.HOLD)

    def test_hold_during_cooldown(self):
        # Enter and exit first
        ms1 = self._make_market_state(funding_rate=-0.0005, timestamp=self.base_time)
        self.strategy.on_tick(ms1)
        # Force exit
        ms2 = self._make_market_state(
            funding_rate=0.00001,
            timestamp=self.base_time + timedelta(hours=1))
        self.strategy.on_tick(ms2)
        # Try to enter again during cooldown (< 2 hours)
        ms3 = self._make_market_state(
            funding_rate=-0.0005,
            timestamp=self.base_time + timedelta(hours=1, minutes=30))
        signal = self.strategy.on_tick(ms3)
        self.assertEqual(signal.signal_type, SignalType.HOLD)

    def test_hold_low_open_interest(self):
        ms = self._make_market_state(funding_rate=-0.0005, open_interest=500000)
        signal = self.strategy.on_tick(ms)
        self.assertEqual(signal.signal_type, SignalType.HOLD)

    def test_exit_funding_normalized(self):
        # Enter long
        ms1 = self._make_market_state(funding_rate=-0.0005)
        signal1 = self.strategy.on_tick(ms1)
        self.assertEqual(signal1.signal_type, SignalType.ENTER_LONG)
        # Funding normalizes
        ms2 = self._make_market_state(
            funding_rate=-0.00001,
            timestamp=self.base_time + timedelta(hours=1))
        signal2 = self.strategy.on_tick(ms2)
        self.assertEqual(signal2.signal_type, SignalType.EXIT_LONG)

    def test_exit_rate_reversed(self):
        # Enter long (negative funding)
        ms1 = self._make_market_state(funding_rate=-0.0005)
        self.strategy.on_tick(ms1)
        # Rate reverses to positive (above exit threshold)
        ms2 = self._make_market_state(
            funding_rate=0.0002,
            timestamp=self.base_time + timedelta(hours=1))
        signal = self.strategy.on_tick(ms2)
        self.assertEqual(signal.signal_type, SignalType.EXIT_LONG)

    def test_exit_max_hold_exceeded(self):
        # Enter
        ms1 = self._make_market_state(funding_rate=-0.0005)
        self.strategy.on_tick(ms1)
        # Tick 24 times (all still in favorable funding)
        for i in range(1, 25):
            ms = self._make_market_state(
                funding_rate=-0.0003,
                timestamp=self.base_time + timedelta(hours=i))
            signal = self.strategy.on_tick(ms)
        self.assertEqual(signal.signal_type, SignalType.EXIT_LONG)

    def test_state_round_trip(self):
        # Enter a position
        ms = self._make_market_state(funding_rate=-0.0005)
        self.strategy.on_tick(ms)
        # Save state
        state = self.strategy.get_state()
        self.assertTrue(state["in_position"])
        self.assertEqual(state["position_side"], "long")
        # Create new strategy and restore
        new_strategy = FundingRateArbStrategy(self.params)
        new_strategy.set_state(state)
        self.assertTrue(new_strategy._in_position)
        self.assertEqual(new_strategy._position_side, "long")

    def test_position_sizing(self):
        ms = self._make_market_state(
            funding_rate=-0.0005, equity=100000.0, mark_price=50000.0)
        signal = self.strategy.on_tick(ms)
        # 3% of 100000 / 50000 = 0.06
        expected_size = 100000.0 * 0.03 / 50000.0
        self.assertAlmostEqual(signal.size, expected_size, places=6)

    def test_get_metadata(self):
        meta = self.strategy.get_metadata()
        self.assertEqual(meta["name"], "funding_rate_arb")
        self.assertEqual(meta["version"], "1.0")
        self.assertIn("params", meta)
        self.assertEqual(meta["params"]["entry_threshold"], 0.0001)

    def test_hold_zero_mark_price(self):
        ms = self._make_market_state(funding_rate=-0.0005, mark_price=0.0)
        signal = self.strategy.on_tick(ms)
        self.assertEqual(signal.signal_type, SignalType.HOLD)

    def test_exit_short_on_rate_reversal(self):
        # Enter short (positive funding)
        ms1 = self._make_market_state(funding_rate=0.0005)
        signal = self.strategy.on_tick(ms1)
        self.assertEqual(signal.signal_type, SignalType.ENTER_SHORT)
        # Rate reverses to negative
        ms2 = self._make_market_state(
            funding_rate=-0.0002,
            timestamp=self.base_time + timedelta(hours=1))
        signal2 = self.strategy.on_tick(ms2)
        self.assertEqual(signal2.signal_type, SignalType.EXIT_SHORT)

    def test_entry_after_cooldown_expires(self):
        # Enter and exit
        ms1 = self._make_market_state(funding_rate=-0.0005, timestamp=self.base_time)
        self.strategy.on_tick(ms1)
        ms2 = self._make_market_state(
            funding_rate=0.00001,
            timestamp=self.base_time + timedelta(hours=1))
        self.strategy.on_tick(ms2)
        # After cooldown (>2 hours)
        ms3 = self._make_market_state(
            funding_rate=-0.0005,
            timestamp=self.base_time + timedelta(hours=4))
        signal = self.strategy.on_tick(ms3)
        self.assertEqual(signal.signal_type, SignalType.ENTER_LONG)


if __name__ == "__main__":
    unittest.main()
