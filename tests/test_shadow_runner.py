#!/usr/bin/env python3
"""Tests for the shadow runner."""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from core.models import Signal
from core.types import SignalType
from strategy.live_strategy import LiveStrategy, MarketState
from orchestration.shadow_runner import ShadowRunner


class MockShadowStrategy(LiveStrategy):
    """Mock strategy for shadow runner tests."""
    def __init__(self, signals=None):
        super().__init__({})
        self._signals = signals or []
        self._call_count = 0

    def on_tick(self, market_state):
        if self._call_count < len(self._signals):
            signal = self._signals[self._call_count]
        else:
            signal = Signal(
                signal_type=SignalType.HOLD,
                price=market_state.mid_price,
                timestamp=datetime.now(timezone.utc),
            )
        self._call_count += 1
        return signal

    def get_metadata(self):
        return {"name": "shadow_test", "version": "1.0"}

    def get_state(self):
        return {}

    def set_state(self, state):
        pass


class TestShadowRunner(unittest.TestCase):

    def _make_config(self):
        config = MagicMock()
        config.get.side_effect = lambda key, default=None: {
            "strategy_runner.symbol": "BTC/USDC",
            "strategy_runner.tick_interval_s": 0,  # immediate for tests
            "strategy_runner.funding_lookback_hours": 72,
            "strategy_runner.candle_lookback": 100,
            "strategy_runner.candle_timeframe": "1h",
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
        redis.get_market_regime.return_value = None
        return redis

    def _make_timescale(self):
        timescale = MagicMock()
        timescale.query_funding_rates.return_value = []
        timescale.query_candles.return_value = []
        return timescale

    def _make_runner(self, strategy=None):
        strategy = strategy or MockShadowStrategy()
        return ShadowRunner(
            strategy=strategy,
            timescale=self._make_timescale(),
            redis=self._make_redis(),
            config=self._make_config(),
            name="shadow_test",
        )

    def _run_async(self, coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(
                    *pending, return_exceptions=True))
            loop.close()

    def test_run_processes_ticks(self):
        """ShadowRunner tick loop processes ticks with HOLD signals."""
        strategy = MockShadowStrategy()
        runner = self._make_runner(strategy)
        stop = asyncio.Event()

        async def _test():
            stop.set()
            await runner.run(stop)

        self._run_async(_test())
        self.assertFalse(runner._running)

    def test_no_execution_engine_called(self):
        """ShadowRunner has no execution engine — signals are never executed."""
        runner = self._make_runner()
        # Verify no execution_engine attribute
        self.assertFalse(hasattr(runner, 'execution_engine'))

    def test_signal_recorded_on_entry(self):
        """Signals should be recorded when strategy emits them."""
        enter_signal = Signal(
            signal_type=SignalType.ENTER_LONG,
            price=50000.0,
            timestamp=datetime.now(timezone.utc),
            size=0.1,
        )
        strategy = MockShadowStrategy(signals=[enter_signal])
        runner = self._make_runner(strategy)

        self._run_async(runner._tick())

        signals = runner.get_signals()
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["signal_type"], "enter_long")

    def test_hypothetical_pnl_long_trade(self):
        """Long trade: enter at mid, exit at higher mid -> positive PnL."""
        enter = Signal(signal_type=SignalType.ENTER_LONG, price=50000.0,
                       timestamp=datetime.now(timezone.utc), size=1.0)
        exit_ = Signal(signal_type=SignalType.EXIT_LONG, price=50100.0,
                       timestamp=datetime.now(timezone.utc))

        strategy = MockShadowStrategy(signals=[enter, exit_])
        runner = self._make_runner(strategy)

        # First tick — entry at mid_price = 50000
        self._run_async(runner._tick())
        self.assertIsNotNone(runner._position)

        # Change price for exit tick
        runner.redis.get_price.return_value = {
            "last": 50100.0, "bid": 50099.0, "ask": 50101.0,
        }

        # Second tick — exit at mid_price = 50100
        self._run_async(runner._tick())
        self.assertIsNone(runner._position)

        summary = runner.get_performance_summary()
        self.assertEqual(summary["trade_count"], 1)
        self.assertGreater(summary["total_pnl"], 0)
        self.assertEqual(summary["wins"], 1)

    def test_hypothetical_pnl_short_trade(self):
        """Short trade: enter at mid, exit at lower mid -> positive PnL."""
        enter = Signal(signal_type=SignalType.ENTER_SHORT, price=50000.0,
                       timestamp=datetime.now(timezone.utc), size=1.0)
        exit_ = Signal(signal_type=SignalType.EXIT_SHORT, price=49900.0,
                       timestamp=datetime.now(timezone.utc))

        strategy = MockShadowStrategy(signals=[enter, exit_])
        runner = self._make_runner(strategy)

        # Entry tick
        self._run_async(runner._tick())
        self.assertIsNotNone(runner._position)
        self.assertEqual(runner._position["side"], "short")

        # Change price for exit
        runner.redis.get_price.return_value = {
            "last": 49900.0, "bid": 49899.0, "ask": 49901.0,
        }

        # Exit tick
        self._run_async(runner._tick())
        self.assertIsNone(runner._position)

        summary = runner.get_performance_summary()
        self.assertEqual(summary["trade_count"], 1)
        self.assertGreater(summary["total_pnl"], 0)

    def test_max_drawdown_tracked(self):
        """Max drawdown should update after a losing trade."""
        # Win first, then lose — should show drawdown
        signals = [
            Signal(signal_type=SignalType.ENTER_LONG, price=50000.0,
                   timestamp=datetime.now(timezone.utc), size=1.0),
            Signal(signal_type=SignalType.EXIT_LONG, price=50100.0,
                   timestamp=datetime.now(timezone.utc)),
            Signal(signal_type=SignalType.ENTER_LONG, price=50100.0,
                   timestamp=datetime.now(timezone.utc), size=1.0),
            Signal(signal_type=SignalType.EXIT_LONG, price=49900.0,
                   timestamp=datetime.now(timezone.utc)),
        ]
        strategy = MockShadowStrategy(signals=signals)
        runner = self._make_runner(strategy)

        # Win trade (entry at 50000, exit at 50100)
        self._run_async(runner._tick())
        runner.redis.get_price.return_value = {
            "last": 50100.0, "bid": 50099.0, "ask": 50101.0,
        }
        self._run_async(runner._tick())

        # Lose trade (entry at 50100, exit at 49900)
        self._run_async(runner._tick())
        runner.redis.get_price.return_value = {
            "last": 49900.0, "bid": 49899.0, "ask": 49901.0,
        }
        self._run_async(runner._tick())

        summary = runner.get_performance_summary()
        self.assertGreater(summary["max_drawdown"], 0)

    def test_performance_summary_structure(self):
        """Performance summary should have all expected keys."""
        runner = self._make_runner()
        summary = runner.get_performance_summary()
        expected_keys = {"total_pnl", "trade_count", "win_rate",
                         "max_drawdown", "sharpe", "wins", "losses"}
        self.assertEqual(set(summary.keys()), expected_keys)

    def test_stop_event_halts_loop(self):
        """Setting stop event should halt the run loop."""
        strategy = MockShadowStrategy()
        runner = self._make_runner(strategy)
        stop = asyncio.Event()

        async def _test():
            stop.set()
            await runner.run(stop)
            return runner._running

        running = self._run_async(_test())
        self.assertFalse(running)

    def test_get_signals_returns_list(self):
        """get_signals should return a list."""
        runner = self._make_runner()
        self.assertIsInstance(runner.get_signals(), list)
        self.assertEqual(len(runner.get_signals()), 0)

    def test_multiple_trades(self):
        """Multiple round-trip trades should all be tracked."""
        signals = [
            Signal(signal_type=SignalType.ENTER_LONG, price=50000.0,
                   timestamp=datetime.now(timezone.utc), size=1.0),
            Signal(signal_type=SignalType.EXIT_LONG, price=50050.0,
                   timestamp=datetime.now(timezone.utc)),
            Signal(signal_type=SignalType.ENTER_SHORT, price=50050.0,
                   timestamp=datetime.now(timezone.utc), size=1.0),
            Signal(signal_type=SignalType.EXIT_SHORT, price=50000.0,
                   timestamp=datetime.now(timezone.utc)),
        ]
        strategy = MockShadowStrategy(signals=signals)
        runner = self._make_runner(strategy)

        # Trade 1: long entry
        self._run_async(runner._tick())
        # Trade 1: long exit (profit)
        runner.redis.get_price.return_value = {
            "last": 50050.0, "bid": 50049.0, "ask": 50051.0,
        }
        self._run_async(runner._tick())
        # Trade 2: short entry
        self._run_async(runner._tick())
        # Trade 2: short exit (profit)
        runner.redis.get_price.return_value = {
            "last": 50000.0, "bid": 49999.0, "ask": 50001.0,
        }
        self._run_async(runner._tick())

        summary = runner.get_performance_summary()
        self.assertEqual(summary["trade_count"], 2)
        self.assertGreater(summary["total_pnl"], 0)

    def test_hold_signal_no_position_change(self):
        """HOLD signal should not change position state."""
        runner = self._make_runner()

        self._run_async(runner._tick())

        self.assertIsNone(runner._position)
        self.assertEqual(runner._trade_count, 0)

    def test_win_rate_calculation(self):
        """Win rate should be correctly calculated."""
        signals = [
            Signal(signal_type=SignalType.ENTER_LONG, price=50000.0,
                   timestamp=datetime.now(timezone.utc), size=1.0),
            Signal(signal_type=SignalType.EXIT_LONG, price=50100.0,
                   timestamp=datetime.now(timezone.utc)),
            Signal(signal_type=SignalType.ENTER_LONG, price=50100.0,
                   timestamp=datetime.now(timezone.utc), size=1.0),
            Signal(signal_type=SignalType.EXIT_LONG, price=49900.0,
                   timestamp=datetime.now(timezone.utc)),
        ]
        strategy = MockShadowStrategy(signals=signals)
        runner = self._make_runner(strategy)

        # Win
        self._run_async(runner._tick())
        runner.redis.get_price.return_value = {
            "last": 50100.0, "bid": 50099.0, "ask": 50101.0,
        }
        self._run_async(runner._tick())

        # Loss
        self._run_async(runner._tick())
        runner.redis.get_price.return_value = {
            "last": 49900.0, "bid": 49899.0, "ask": 49901.0,
        }
        self._run_async(runner._tick())

        summary = runner.get_performance_summary()
        self.assertEqual(summary["win_rate"], 50.0)
        self.assertEqual(summary["wins"], 1)
        self.assertEqual(summary["losses"], 1)

    def test_empty_performance_no_trades(self):
        """Performance summary with no trades should have zero values."""
        runner = self._make_runner()
        summary = runner.get_performance_summary()
        self.assertEqual(summary["total_pnl"], 0.0)
        self.assertEqual(summary["trade_count"], 0)
        self.assertEqual(summary["win_rate"], 0.0)
        self.assertEqual(summary["max_drawdown"], 0.0)
        self.assertEqual(summary["sharpe"], 0.0)


if __name__ == "__main__":
    unittest.main()
