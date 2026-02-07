"""Tests for analysis.backtest_comparator.BacktestLiveComparator."""

import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.models import Signal
from core.types import SignalType
from analysis.backtest_comparator import BacktestLiveComparator


class TestBacktestLiveComparator(unittest.TestCase):
    """Tests for BacktestLiveComparator."""

    def setUp(self):
        self.mock_strategy = MagicMock()
        self.mock_strategy.get_state.return_value = {}
        self.mock_strategy.get_metadata.return_value = {"name": "test"}

        self.mock_timescale = MagicMock()
        self.mock_redis = MagicMock()

        self.comparator = BacktestLiveComparator(
            strategy=self.mock_strategy,
            timescale=self.mock_timescale,
            redis_store=self.mock_redis,
            strategy_name="funding_rate_arb",
            symbol="BTC/USDC",
        )

    def _make_candles(self, start: datetime, hours: int,
                      base_price: float = 100.0) -> list[dict]:
        """Create mock candle data."""
        candles = []
        for i in range(hours):
            t = start + timedelta(hours=i)
            candles.append({
                "time": t,
                "open": base_price,
                "high": base_price + 1,
                "low": base_price - 1,
                "close": base_price,
                "volume": 1000.0,
            })
        return candles

    def test_compare_with_no_candles(self):
        """No candles should return empty comparison."""
        self.mock_timescale.query_candles.return_value = []
        self.mock_timescale.query_funding_rates.return_value = []
        self.mock_timescale.query_fills_by_strategy.return_value = []

        result = self.comparator.compare()
        self.assertEqual(result["backtest_pnl"], 0.0)
        self.assertFalse(result["should_pause"])

    def test_compare_high_decay_triggers_pause(self):
        """Decay > 50% should trigger pause recommendation."""
        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=168)

        candles = self._make_candles(start, 168)
        funding = [{"time": start + timedelta(hours=i), "rate": 0.0001, "premium": 0.0}
                   for i in range(168)]

        # Strategy produces entry + exit signals during replay
        hold_signal = Signal(signal_type=SignalType.HOLD, price=100.0,
                             timestamp=now)
        enter_signal = Signal(signal_type=SignalType.ENTER_LONG, price=100.0,
                              timestamp=now, size=1.0)
        exit_signal = Signal(signal_type=SignalType.EXIT_LONG, price=110.0,
                             timestamp=now, size=1.0)

        # Alternate between signals to simulate trades
        call_count = [0]
        def mock_on_tick(state):
            call_count[0] += 1
            if call_count[0] == 10:
                return enter_signal
            elif call_count[0] == 20:
                return exit_signal
            return hold_signal

        self.mock_strategy.on_tick.side_effect = mock_on_tick

        # Live fills show much less profit
        live_fills = [
            {"time": start + timedelta(hours=10), "side": "buy",
             "price": 100.0, "quantity": 1.0, "closed_pnl": 0.0},
            {"time": start + timedelta(hours=20), "side": "sell",
             "price": 102.0, "quantity": 1.0, "closed_pnl": 2.0},
        ]

        self.mock_timescale.query_candles.return_value = candles
        self.mock_timescale.query_funding_rates.return_value = funding
        self.mock_timescale.query_fills_by_strategy.return_value = live_fills

        result = self.comparator.compare()
        # Backtest PnL should be 10 (110-100), live PnL = 2, decay = 80%
        self.assertGreater(result["decay_pct"], 50.0)
        self.assertTrue(result["should_pause"])

    def test_compare_no_decay(self):
        """No decay when live matches backtest."""
        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=24)

        candles = self._make_candles(start, 24)

        # No signals during replay
        hold_signal = Signal(signal_type=SignalType.HOLD, price=100.0,
                             timestamp=now)
        self.mock_strategy.on_tick.return_value = hold_signal

        self.mock_timescale.query_candles.return_value = candles
        self.mock_timescale.query_funding_rates.return_value = []
        self.mock_timescale.query_fills_by_strategy.return_value = []

        result = self.comparator.compare()
        self.assertEqual(result["decay_pct"], 0.0)
        self.assertFalse(result["should_pause"])

    def test_result_structure(self):
        """Result should have all expected keys."""
        self.mock_timescale.query_candles.return_value = []
        self.mock_timescale.query_funding_rates.return_value = []
        self.mock_timescale.query_fills_by_strategy.return_value = []

        result = self.comparator.compare()
        expected_keys = [
            "backtest_pnl", "live_pnl", "decay_pct", "should_pause",
            "backtest_trade_count", "live_trade_count",
            "entry_price_deviation_pct",
        ]
        for key in expected_keys:
            self.assertIn(key, result)

    def test_strategy_state_preserved(self):
        """Strategy state should be saved and restored after replay."""
        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=24)

        original_state = {"in_position": True, "entry_price": 100.0}
        self.mock_strategy.get_state.return_value = original_state

        candles = self._make_candles(start, 24)
        hold_signal = Signal(signal_type=SignalType.HOLD, price=100.0,
                             timestamp=now)
        self.mock_strategy.on_tick.return_value = hold_signal

        self.mock_timescale.query_candles.return_value = candles
        self.mock_timescale.query_funding_rates.return_value = []
        self.mock_timescale.query_fills_by_strategy.return_value = []

        self.comparator.compare()

        # State should be restored
        self.mock_strategy.set_state.assert_called()
        # Last call should restore original state
        last_call = self.mock_strategy.set_state.call_args_list[-1]
        self.assertEqual(last_call[0][0], original_state)

    def test_exception_handling(self):
        """Exceptions from timescale should be handled gracefully."""
        self.mock_timescale.query_candles.side_effect = Exception("DB error")
        self.mock_timescale.query_funding_rates.side_effect = Exception("DB error")
        self.mock_timescale.query_fills_by_strategy.side_effect = Exception("DB error")

        result = self.comparator.compare()
        self.assertEqual(result["backtest_pnl"], 0.0)
        self.assertFalse(result["should_pause"])


if __name__ == "__main__":
    unittest.main()
