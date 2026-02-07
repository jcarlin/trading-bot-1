#!/usr/bin/env python3
"""Tests for the LiveStrategy interface and MarketState dataclass."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from datetime import datetime, timezone

from core.models import Signal
from core.types import SignalType
from strategy.live_strategy import LiveStrategy, MarketState


class DummyStrategy(LiveStrategy):
    """Concrete strategy for testing."""

    def on_tick(self, market_state):
        return Signal(
            signal_type=SignalType.HOLD,
            price=market_state.mark_price,
            timestamp=market_state.timestamp,
        )

    def get_metadata(self):
        return {"name": "dummy", "version": "1.0"}


class TestMarketState(unittest.TestCase):

    def test_construction_with_all_fields(self):
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        ms = MarketState(
            mark_price=100.0,
            mid_price=100.5,
            bid=100.0,
            ask=101.0,
            funding_rate=0.0001,
            premium=0.5,
            open_interest=5000000.0,
            funding_history=[{"rate": 0.0001, "ts": "2025-01-01"}],
            position={"size": 1.0, "side": "long"},
            equity=10000.0,
            cash=5000.0,
            drawdown_pct=2.5,
            recent_candles=[{"close": 100.0}],
            spread=1.0,
            bid_depth=500.0,
            ask_depth=600.0,
            timestamp=ts,
        )
        self.assertEqual(ms.mark_price, 100.0)
        self.assertEqual(ms.mid_price, 100.5)
        self.assertEqual(ms.bid, 100.0)
        self.assertEqual(ms.ask, 101.0)
        self.assertEqual(ms.funding_rate, 0.0001)
        self.assertEqual(ms.premium, 0.5)
        self.assertEqual(ms.open_interest, 5000000.0)
        self.assertEqual(ms.funding_history, [{"rate": 0.0001, "ts": "2025-01-01"}])
        self.assertEqual(ms.position, {"size": 1.0, "side": "long"})
        self.assertEqual(ms.equity, 10000.0)
        self.assertEqual(ms.cash, 5000.0)
        self.assertEqual(ms.drawdown_pct, 2.5)
        self.assertEqual(ms.recent_candles, [{"close": 100.0}])
        self.assertEqual(ms.spread, 1.0)
        self.assertEqual(ms.bid_depth, 500.0)
        self.assertEqual(ms.ask_depth, 600.0)
        self.assertEqual(ms.timestamp, ts)

    def test_default_values(self):
        ms = MarketState(
            mark_price=50.0,
            mid_price=50.5,
            bid=50.0,
            ask=51.0,
            funding_rate=0.0002,
            premium=0.1,
            open_interest=1000000.0,
        )
        self.assertEqual(ms.funding_history, [])
        self.assertIsNone(ms.position)
        self.assertEqual(ms.equity, 0.0)
        self.assertEqual(ms.cash, 0.0)
        self.assertEqual(ms.drawdown_pct, 0.0)
        self.assertEqual(ms.recent_candles, [])
        self.assertEqual(ms.spread, 0.0)
        self.assertEqual(ms.bid_depth, 0.0)
        self.assertEqual(ms.ask_depth, 0.0)

    def test_timestamp_default(self):
        before = datetime.now(timezone.utc)
        ms = MarketState(
            mark_price=50.0,
            mid_price=50.5,
            bid=50.0,
            ask=51.0,
            funding_rate=0.0,
            premium=0.0,
            open_interest=0.0,
        )
        after = datetime.now(timezone.utc)
        self.assertGreaterEqual(ms.timestamp, before)
        self.assertLessEqual(ms.timestamp, after)


class TestLiveStrategy(unittest.TestCase):

    def test_cannot_instantiate_abc(self):
        with self.assertRaises(TypeError):
            LiveStrategy({})

    def test_concrete_implementation(self):
        strategy = DummyStrategy({"param1": "value1"})
        ms = MarketState(
            mark_price=100.0,
            mid_price=100.5,
            bid=100.0,
            ask=101.0,
            funding_rate=0.0001,
            premium=0.5,
            open_interest=5000000.0,
        )
        signal = strategy.on_tick(ms)
        self.assertIsInstance(signal, Signal)
        self.assertEqual(signal.signal_type, SignalType.HOLD)
        self.assertEqual(signal.price, 100.0)

    def test_get_state_default(self):
        strategy = DummyStrategy({})
        self.assertEqual(strategy.get_state(), {})

    def test_set_state_default(self):
        strategy = DummyStrategy({})
        strategy.set_state({"key": "value"})  # should not raise

    def test_get_risk_parameters_default(self):
        strategy = DummyStrategy({})
        self.assertEqual(strategy.get_risk_parameters(), {})

    def test_get_metadata_required(self):
        """Verify a concrete class must implement get_metadata."""
        class IncompleteStrategy(LiveStrategy):
            def on_tick(self, market_state):
                return Signal(
                    signal_type=SignalType.HOLD,
                    price=0.0,
                    timestamp=market_state.timestamp,
                )

        with self.assertRaises(TypeError):
            IncompleteStrategy({})


if __name__ == "__main__":
    unittest.main()
