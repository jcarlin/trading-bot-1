#!/usr/bin/env python3
"""Tests for the strategy runner."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from core.models import Signal
from core.types import SignalType
from strategy.live_strategy import LiveStrategy, MarketState
from strategy.runner import StrategyRunner


class MockStrategy(LiveStrategy):
    """Mock strategy for testing."""
    def __init__(self, signal_to_return=None):
        super().__init__({})
        self._signal = signal_to_return or Signal(
            signal_type=SignalType.HOLD,
            price=50000.0,
            timestamp=datetime.now(timezone.utc),
        )
        self._state = {}

    def on_tick(self, market_state):
        return self._signal

    def get_metadata(self):
        return {"name": "test_strategy", "version": "1.0"}

    def get_state(self):
        return self._state

    def set_state(self, state):
        self._state = state


class TestStrategyRunner(unittest.IsolatedAsyncioTestCase):

    def _make_config(self):
        config = MagicMock()
        config.get.side_effect = lambda key, default=None: {
            "strategy_runner.symbol": "BTC/USDC",
            "strategy_runner.tick_interval_s": 1,
            "strategy_runner.funding_lookback_hours": 72,
            "strategy_runner.candle_lookback": 100,
            "strategy_runner.candle_timeframe": "1h",
            "strategy_runner.log_hold_every_n": 10,
        }.get(key, default)
        return config

    def _make_redis(self):
        redis = MagicMock()
        redis.get_price.return_value = {
            "last": 50000.0, "bid": 49999.0, "ask": 50001.0,
            "timestamp": "2025-01-01T00:00:00Z",
        }
        redis.get_funding.return_value = {
            "rate": -0.0005, "premium": 0.001,
            "mark_price": 50000.0, "timestamp": "2025-01-01T00:00:00Z",
        }
        redis.get_position.return_value = None
        redis.get_account_state.return_value = {
            "total_equity": 10000.0, "cash": 5000.0,
            "position_value": 5000.0, "unrealized_pnl": 0.0,
            "drawdown_pct": 0.0, "timestamp": "2025-01-01T00:00:00Z",
        }
        redis.get_orderbook.return_value = {
            "bids": [[49999, 1]], "asks": [[50001, 1]],
            "mid_price": 50000.0, "spread": 2.0,
            "timestamp": "2025-01-01T00:00:00Z",
        }
        redis.get_strategy_state.return_value = None
        redis._r = MagicMock()
        return redis

    def _make_timescale(self):
        timescale = MagicMock()
        timescale.query_funding_rates.return_value = []
        timescale.query_candles.return_value = []
        return timescale

    def setUp(self):
        self.strategy = MockStrategy()
        self.execution_engine = MagicMock()
        self.execution_engine.process_signal = AsyncMock()
        self.timescale = self._make_timescale()
        self.redis = self._make_redis()
        self.config = self._make_config()
        self.decision_logger = MagicMock()

    def _make_runner(self, strategy=None):
        return StrategyRunner(
            strategy=strategy or self.strategy,
            execution_engine=self.execution_engine,
            timescale=self.timescale,
            redis=self.redis,
            config=self.config,
            decision_logger=self.decision_logger,
        )

    def test_build_market_state(self):
        runner = self._make_runner()
        ms = runner._build_market_state()
        self.assertIsNotNone(ms)
        self.assertIsInstance(ms, MarketState)
        self.assertEqual(ms.bid, 49999.0)
        self.assertEqual(ms.ask, 50001.0)
        self.assertEqual(ms.funding_rate, -0.0005)
        self.assertEqual(ms.equity, 10000.0)

    def test_build_market_state_missing_price(self):
        self.redis.get_price.return_value = None
        runner = self._make_runner()
        ms = runner._build_market_state()
        self.assertIsNone(ms)

    def test_build_market_state_missing_funding(self):
        self.redis.get_funding.return_value = None
        runner = self._make_runner()
        ms = runner._build_market_state()
        self.assertIsNone(ms)

    async def test_tick_hold_signal(self):
        runner = self._make_runner()
        await runner._tick()
        self.execution_engine.process_signal.assert_not_called()
        self.decision_logger.log_hold.assert_called_once()

    async def test_tick_enter_signal(self):
        enter_signal = Signal(
            signal_type=SignalType.ENTER_LONG,
            price=50000.0,
            timestamp=datetime.now(timezone.utc),
            size=0.01,
            metadata={},
        )
        strategy = MockStrategy(signal_to_return=enter_signal)
        runner = self._make_runner(strategy=strategy)
        await runner._tick()
        self.execution_engine.process_signal.assert_called_once()
        call_args = self.execution_engine.process_signal.call_args
        self.assertEqual(call_args[0][0].signal_type, SignalType.ENTER_LONG)
        self.assertEqual(call_args[0][1], "test_strategy")
        # Verify symbol was set in metadata
        self.assertEqual(call_args[0][0].metadata["symbol"], "BTC/USDC")

    async def test_tick_logs_non_hold_signal(self):
        enter_signal = Signal(
            signal_type=SignalType.ENTER_SHORT,
            price=50000.0,
            timestamp=datetime.now(timezone.utc),
            metadata={},
        )
        strategy = MockStrategy(signal_to_return=enter_signal)
        runner = self._make_runner(strategy=strategy)
        await runner._tick()
        self.decision_logger.log_signal.assert_called_once()

    def test_checkpoint_state(self):
        runner = self._make_runner()
        self.strategy._state = {"in_position": True, "side": "long"}
        runner._checkpoint_state()
        self.redis.set_strategy_state.assert_called()
        self.redis._r.set.assert_called_once()
        key, value = self.redis._r.set.call_args[0]
        self.assertIn("test_strategy", key)
        state = json.loads(value)
        self.assertTrue(state["in_position"])

    def test_restore_state_no_data(self):
        self.redis.get_strategy_state.return_value = None
        runner = self._make_runner()
        runner._restore_state()
        # Should not crash, state stays default
        self.assertEqual(self.strategy._state, {})

    async def test_run_stops_on_event(self):
        runner = self._make_runner()
        stop_event = asyncio.Event()
        # Set the stop event immediately
        stop_event.set()
        await runner.run(stop_event)
        self.assertFalse(runner._running)

    def test_strategy_name_from_metadata(self):
        runner = self._make_runner()
        self.assertEqual(runner._strategy_name, "test_strategy")

    async def test_tick_handles_market_state_none(self):
        self.redis.get_price.return_value = None
        runner = self._make_runner()
        # Should not raise
        await runner._tick()
        self.execution_engine.process_signal.assert_not_called()


if __name__ == "__main__":
    unittest.main()
