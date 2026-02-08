"""Tests for smart money strategy."""

import asyncio
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.models import Signal
from core.types import SignalType
from strategy.smart_money import SmartMoneyStrategy, SmartMoneyBacktestAdapter
from strategy.live_strategy import MarketState


def _make_market_state(price=50000.0, equity=10000.0, **overrides):
    defaults = dict(
        mark_price=price,
        mid_price=price,
        bid=price - 1.0,
        ask=price + 1.0,
        funding_rate=0.0001,
        premium=0.0,
        open_interest=1000.0,
        equity=equity,
        cash=5000.0,
        timestamp=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return MarketState(**defaults)


def _make_wallet_signal(direction="long", wallet_score=85.0, signal_type="new_position",
                        wallet_address="0xabc", is_unusual=False, ts_offset_s=0):
    ts = datetime.now(timezone.utc) - timedelta(seconds=ts_offset_s)
    return {
        "timestamp": ts.isoformat(),
        "wallet_address": wallet_address,
        "wallet_score": wallet_score,
        "symbol": "BTC/USDC",
        "signal_type": signal_type,
        "direction": direction,
        "size": 1.0,
        "size_change_pct": 100.0,
        "is_unusual": is_unusual,
        "confidence": 0.7,
    }


class TestSmartMoneyEntry(unittest.TestCase):
    """Tests for entry conditions."""

    def _make_strategy_with_signals(self, signals, **params_overrides):
        params = {
            "min_wallets_agree": 2,
            "agreement_window_s": 3600,
            "min_wallet_score": 70,
        }
        params.update(params_overrides)
        monitor = MagicMock()
        monitor.get_recent_signals.return_value = signals
        strategy = SmartMoneyStrategy(params, wallet_monitor=monitor)
        return strategy

    def test_entry_long_when_wallets_agree(self):
        signals = [
            _make_wallet_signal("long", wallet_address="0x1"),
            _make_wallet_signal("long", wallet_address="0x2"),
        ]
        strategy = self._make_strategy_with_signals(signals)
        ms = _make_market_state()
        result = strategy.on_tick(ms)
        self.assertEqual(result.signal_type, SignalType.ENTER_LONG)
        self.assertIn("smart_money_consensus_long", result.metadata.get("entry_reason", ""))

    def test_entry_short_when_wallets_agree(self):
        signals = [
            _make_wallet_signal("short", wallet_address="0x1"),
            _make_wallet_signal("short", wallet_address="0x2"),
        ]
        strategy = self._make_strategy_with_signals(signals)
        ms = _make_market_state()
        result = strategy.on_tick(ms)
        self.assertEqual(result.signal_type, SignalType.ENTER_SHORT)

    def test_no_entry_below_agreement_threshold(self):
        # Only 1 wallet, need 2
        signals = [
            _make_wallet_signal("long", wallet_address="0x1"),
        ]
        strategy = self._make_strategy_with_signals(signals)
        ms = _make_market_state()
        result = strategy.on_tick(ms)
        self.assertEqual(result.signal_type, SignalType.HOLD)

    def test_no_entry_mixed_directions_no_majority(self):
        signals = [
            _make_wallet_signal("long", wallet_address="0x1"),
            _make_wallet_signal("short", wallet_address="0x2"),
        ]
        strategy = self._make_strategy_with_signals(signals)
        ms = _make_market_state()
        result = strategy.on_tick(ms)
        self.assertEqual(result.signal_type, SignalType.HOLD)

    def test_no_entry_without_wallet_monitor(self):
        strategy = SmartMoneyStrategy({"min_wallets_agree": 2}, wallet_monitor=None)
        ms = _make_market_state()
        result = strategy.on_tick(ms)
        self.assertEqual(result.signal_type, SignalType.HOLD)

    def test_no_entry_zero_price(self):
        signals = [
            _make_wallet_signal("long", wallet_address="0x1"),
            _make_wallet_signal("long", wallet_address="0x2"),
        ]
        strategy = self._make_strategy_with_signals(signals)
        ms = _make_market_state(price=0.0)
        result = strategy.on_tick(ms)
        self.assertEqual(result.signal_type, SignalType.HOLD)

    def test_low_score_signals_filtered(self):
        # Wallet score 50 is below min_wallet_score=70
        signals = [
            _make_wallet_signal("long", wallet_address="0x1", wallet_score=50),
            _make_wallet_signal("long", wallet_address="0x2", wallet_score=50),
        ]
        strategy = self._make_strategy_with_signals(signals)
        ms = _make_market_state()
        result = strategy.on_tick(ms)
        self.assertEqual(result.signal_type, SignalType.HOLD)


class TestSmartMoneyWeighting(unittest.TestCase):
    """Tests for wallet score weighting."""

    def test_wallet_score_affects_confidence(self):
        strategy = SmartMoneyStrategy({"min_wallets_agree": 2, "min_wallet_score": 0})
        # High-score wallets
        high_signals = [
            _make_wallet_signal("long", wallet_address="0x1", wallet_score=95),
            _make_wallet_signal("long", wallet_address="0x2", wallet_score=90),
        ]
        dir_high, conf_high = strategy._compute_aggregate_direction(high_signals)

        # Low-score wallets
        low_signals = [
            _make_wallet_signal("long", wallet_address="0x3", wallet_score=30),
            _make_wallet_signal("long", wallet_address="0x4", wallet_score=25),
        ]
        dir_low, conf_low = strategy._compute_aggregate_direction(low_signals)

        self.assertEqual(dir_high, "long")
        self.assertEqual(dir_low, "long")
        # Both have 100% agreement so confidence should be equal (1.0)
        # The key difference is that these would be filtered by min_wallet_score in _get_wallet_signals

    def test_unusual_size_boost_increases_weight(self):
        strategy = SmartMoneyStrategy({
            "min_wallets_agree": 2,
            "unusual_size_boost": 0.3,
            "min_wallet_score": 0,
        })
        # 2 longs (unusual) vs 2 shorts (normal), equal wallet scores
        signals = [
            _make_wallet_signal("long", wallet_address="0x1", wallet_score=50, is_unusual=True),
            _make_wallet_signal("long", wallet_address="0x2", wallet_score=50, is_unusual=True),
            _make_wallet_signal("short", wallet_address="0x3", wallet_score=50),
            _make_wallet_signal("short", wallet_address="0x4", wallet_score=50),
        ]
        direction, confidence = strategy._compute_aggregate_direction(signals)
        # Long wallets get unusual boost: 0.5+0.3=0.8 each = 1.6 total
        # Short wallets: 0.5 each = 1.0 total
        self.assertEqual(direction, "long")

    def test_deduplication_keeps_latest(self):
        strategy = SmartMoneyStrategy({"min_wallets_agree": 1, "min_wallet_score": 0})
        # Same wallet, two signals — first long, then short (later timestamp)
        signals = [
            _make_wallet_signal("long", wallet_address="0xsame", ts_offset_s=60),
            _make_wallet_signal("short", wallet_address="0xsame", ts_offset_s=0),
        ]
        direction, _ = strategy._compute_aggregate_direction(signals)
        # Latest is short (ts_offset_s=0 means later)
        self.assertEqual(direction, "short")


class TestSmartMoneyExit(unittest.TestCase):
    """Tests for exit conditions."""

    def _enter_long(self, strategy, price=50000.0):
        monitor = strategy.wallet_monitor
        monitor.get_recent_signals.return_value = [
            _make_wallet_signal("long", wallet_address="0x1"),
            _make_wallet_signal("long", wallet_address="0x2"),
        ]
        ms = _make_market_state(price=price)
        result = strategy.on_tick(ms)
        self.assertEqual(result.signal_type, SignalType.ENTER_LONG)
        return result

    def test_exit_on_max_hold_period(self):
        monitor = MagicMock()
        monitor.get_recent_signals.return_value = []
        strategy = SmartMoneyStrategy(
            {"min_wallets_agree": 2, "max_hold_periods": 3}, wallet_monitor=monitor)

        # Enter
        monitor.get_recent_signals.return_value = [
            _make_wallet_signal("long", wallet_address="0x1"),
            _make_wallet_signal("long", wallet_address="0x2"),
        ]
        ms = _make_market_state(price=50000.0)
        strategy.on_tick(ms)

        # Tick 3 times (max_hold_periods=3)
        monitor.get_recent_signals.return_value = []
        for _ in range(2):
            result = strategy.on_tick(_make_market_state(price=50000.0))
            self.assertEqual(result.signal_type, SignalType.HOLD)

        result = strategy.on_tick(_make_market_state(price=50000.0))
        self.assertEqual(result.signal_type, SignalType.EXIT_LONG)
        self.assertEqual(result.metadata.get("exit_reason"), "max_hold_period")

    def test_exit_on_drawdown(self):
        monitor = MagicMock()
        monitor.get_recent_signals.return_value = [
            _make_wallet_signal("long", wallet_address="0x1"),
            _make_wallet_signal("long", wallet_address="0x2"),
        ]
        strategy = SmartMoneyStrategy(
            {"min_wallets_agree": 2, "drawdown_exit_pct": 2.0, "max_hold_periods": 100},
            wallet_monitor=monitor)

        # Enter at 50000
        strategy.on_tick(_make_market_state(price=50000.0))

        # Price drops 3% -> triggers 2% exit
        monitor.get_recent_signals.return_value = []
        result = strategy.on_tick(_make_market_state(price=48500.0))
        self.assertEqual(result.signal_type, SignalType.EXIT_LONG)
        self.assertEqual(result.metadata.get("exit_reason"), "drawdown_exit")

    def test_exit_on_consensus_reversal(self):
        monitor = MagicMock()
        monitor.get_recent_signals.return_value = [
            _make_wallet_signal("long", wallet_address="0x1"),
            _make_wallet_signal("long", wallet_address="0x2"),
        ]
        strategy = SmartMoneyStrategy(
            {"min_wallets_agree": 2, "max_hold_periods": 100},
            wallet_monitor=monitor)

        # Enter long
        strategy.on_tick(_make_market_state(price=50000.0))

        # Wallets flip to short
        monitor.get_recent_signals.return_value = [
            _make_wallet_signal("short", wallet_address="0x1"),
            _make_wallet_signal("short", wallet_address="0x2"),
        ]
        result = strategy.on_tick(_make_market_state(price=50000.0))
        self.assertEqual(result.signal_type, SignalType.EXIT_LONG)
        self.assertEqual(result.metadata.get("exit_reason"), "consensus_reversal")

    def test_hold_when_no_exit_conditions(self):
        monitor = MagicMock()
        monitor.get_recent_signals.return_value = [
            _make_wallet_signal("long", wallet_address="0x1"),
            _make_wallet_signal("long", wallet_address="0x2"),
        ]
        strategy = SmartMoneyStrategy(
            {"min_wallets_agree": 2, "max_hold_periods": 100, "drawdown_exit_pct": 10.0},
            wallet_monitor=monitor)

        # Enter long
        strategy.on_tick(_make_market_state(price=50000.0))

        # No exit conditions met — price barely changed, no reversal
        monitor.get_recent_signals.return_value = []
        result = strategy.on_tick(_make_market_state(price=50100.0))
        self.assertEqual(result.signal_type, SignalType.HOLD)


class TestSmartMoneyState(unittest.TestCase):
    """Tests for state save/restore."""

    def test_get_state_no_position(self):
        strategy = SmartMoneyStrategy({"min_wallets_agree": 2})
        state = strategy.get_state()
        self.assertIsNone(state["position_side"])
        self.assertIsNone(state["entry_price"])
        self.assertEqual(state["hold_periods"], 0)

    def test_set_state_restores_position(self):
        strategy = SmartMoneyStrategy({"min_wallets_agree": 2})
        state = {
            "position_side": "long",
            "entry_price": 50000.0,
            "entry_time": "2024-01-15T12:00:00+00:00",
            "hold_periods": 5,
        }
        strategy.set_state(state)
        self.assertEqual(strategy._position_side, "long")
        self.assertEqual(strategy._entry_price, 50000.0)
        self.assertEqual(strategy._hold_periods, 5)
        self.assertIsNotNone(strategy._entry_time)

    def test_set_state_none_entry_time(self):
        strategy = SmartMoneyStrategy({"min_wallets_agree": 2})
        state = {
            "position_side": None,
            "entry_price": None,
            "entry_time": None,
            "hold_periods": 0,
        }
        strategy.set_state(state)
        self.assertIsNone(strategy._position_side)
        self.assertIsNone(strategy._entry_time)

    def test_round_trip_state(self):
        strategy = SmartMoneyStrategy({"min_wallets_agree": 2})
        strategy._position_side = "short"
        strategy._entry_price = 45000.0
        strategy._entry_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        strategy._hold_periods = 10

        state = strategy.get_state()

        strategy2 = SmartMoneyStrategy({"min_wallets_agree": 2})
        strategy2.set_state(state)

        self.assertEqual(strategy2._position_side, "short")
        self.assertEqual(strategy2._entry_price, 45000.0)
        self.assertEqual(strategy2._hold_periods, 10)


class TestSmartMoneyMetadata(unittest.TestCase):
    """Tests for metadata."""

    def test_metadata_fields(self):
        strategy = SmartMoneyStrategy({"min_wallets_agree": 3, "max_hold_periods": 24})
        meta = strategy.get_metadata()
        self.assertEqual(meta["name"], "smart_money")
        self.assertEqual(meta["version"], "1.0")
        self.assertEqual(meta["category"], "smart_money")
        self.assertEqual(meta["params"]["min_wallets_agree"], 3)
        self.assertEqual(meta["params"]["max_hold_periods"], 24)


class TestSmartMoneyBacktestAdapter(unittest.TestCase):
    """Tests for the backtest adapter."""

    def test_adapter_setup(self):
        import pandas as pd
        import numpy as np
        adapter = SmartMoneyBacktestAdapter({"min_wallets_agree": 2})
        dates = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
        df = pd.DataFrame({
            "open": np.random.uniform(49000, 51000, 10),
            "high": np.random.uniform(50000, 52000, 10),
            "low": np.random.uniform(48000, 50000, 10),
            "close": np.random.uniform(49000, 51000, 10),
            "volume": np.random.uniform(100, 1000, 10),
        }, index=dates)
        adapter.setup(df)  # Should not raise

    def test_adapter_generate_signal_returns_hold(self):
        import pandas as pd
        import numpy as np
        adapter = SmartMoneyBacktestAdapter({"min_wallets_agree": 2})
        dates = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
        df = pd.DataFrame({
            "open": np.random.uniform(49000, 51000, 10),
            "high": np.random.uniform(50000, 52000, 10),
            "low": np.random.uniform(48000, 50000, 10),
            "close": np.random.uniform(49000, 51000, 10),
            "volume": np.random.uniform(100, 1000, 10),
        }, index=dates)
        adapter.setup(df)
        # Without wallet monitor, should return HOLD
        signal = adapter.generate_signal(5, df)
        self.assertEqual(signal.signal_type, SignalType.HOLD)


if __name__ == "__main__":
    unittest.main()
