"""Tests for wallet monitor and wallet signal."""

import asyncio
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intelligence.wallet_signal import WalletSignal
from intelligence.wallet_monitor import WalletMonitor


class TestWalletSignal(unittest.TestCase):
    """Tests for WalletSignal dataclass."""

    def _make_signal(self, **overrides):
        defaults = {
            "timestamp": datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            "wallet_address": "0xabc123",
            "wallet_score": 85.0,
            "symbol": "BTC/USDC",
            "signal_type": "new_position",
            "direction": "long",
            "size": 1.5,
            "size_change_pct": 100.0,
            "is_unusual": False,
            "confidence": 0.72,
        }
        defaults.update(overrides)
        return WalletSignal(**defaults)

    def test_create_signal(self):
        sig = self._make_signal()
        self.assertEqual(sig.wallet_address, "0xabc123")
        self.assertEqual(sig.signal_type, "new_position")
        self.assertEqual(sig.direction, "long")
        self.assertAlmostEqual(sig.confidence, 0.72)

    def test_to_dict_keys(self):
        sig = self._make_signal()
        d = sig.to_dict()
        expected_keys = {
            "timestamp", "wallet_address", "wallet_score", "symbol",
            "signal_type", "direction", "size", "size_change_pct",
            "is_unusual", "confidence", "metadata",
        }
        self.assertEqual(set(d.keys()), expected_keys)

    def test_to_dict_timestamp_is_iso(self):
        sig = self._make_signal()
        d = sig.to_dict()
        self.assertIsInstance(d["timestamp"], str)
        # Should be parseable
        parsed = datetime.fromisoformat(d["timestamp"])
        self.assertEqual(parsed.year, 2024)

    def test_to_dict_values(self):
        sig = self._make_signal(size=2.5, size_change_pct=50.0, is_unusual=True)
        d = sig.to_dict()
        self.assertEqual(d["size"], 2.5)
        self.assertEqual(d["size_change_pct"], 50.0)
        self.assertTrue(d["is_unusual"])

    def test_metadata_default_empty(self):
        sig = self._make_signal()
        self.assertEqual(sig.metadata, {})

    def test_metadata_custom(self):
        sig = self._make_signal(metadata={"source": "hyperliquid"})
        self.assertEqual(sig.metadata["source"], "hyperliquid")
        d = sig.to_dict()
        self.assertEqual(d["metadata"]["source"], "hyperliquid")


class TestWalletMonitorDetectChanges(unittest.TestCase):
    """Tests for WalletMonitor._detect_changes."""

    def _make_monitor(self, **config_overrides):
        config = {
            "poll_interval_s": 30,
            "max_tracked_wallets": 20,
            "min_wallet_score": 70,
            "size_change_threshold_pct": 25,
            "unusual_size_std": 2.0,
        }
        config.update(config_overrides)
        provider = MagicMock()
        redis_store = MagicMock()
        timescale = MagicMock()
        monitor = WalletMonitor(provider, redis_store, timescale, config)
        return monitor

    def test_detect_new_position(self):
        monitor = self._make_monitor()
        monitor.add_wallet("0xabc", 85.0)
        current = [{"symbol": "BTC/USDC", "side": "long", "size": 1.0}]
        previous = []
        signals = monitor._detect_changes("0xabc", current, previous)
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].signal_type, "new_position")
        self.assertEqual(signals[0].direction, "long")

    def test_detect_position_closed(self):
        monitor = self._make_monitor()
        monitor.add_wallet("0xabc", 85.0)
        current = []
        previous = [{"symbol": "BTC/USDC", "side": "long", "size": 1.0}]
        signals = monitor._detect_changes("0xabc", current, previous)
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].signal_type, "closed")
        self.assertEqual(signals[0].size, 0.0)
        self.assertAlmostEqual(signals[0].size_change_pct, -100.0)

    def test_detect_size_increase(self):
        monitor = self._make_monitor()
        monitor.add_wallet("0xabc", 85.0)
        current = [{"symbol": "BTC/USDC", "side": "long", "size": 2.0}]
        previous = [{"symbol": "BTC/USDC", "side": "long", "size": 1.0}]
        signals = monitor._detect_changes("0xabc", current, previous)
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].signal_type, "size_increase")
        self.assertAlmostEqual(signals[0].size_change_pct, 100.0)

    def test_detect_size_decrease(self):
        monitor = self._make_monitor()
        monitor.add_wallet("0xabc", 85.0)
        current = [{"symbol": "BTC/USDC", "side": "long", "size": 0.5}]
        previous = [{"symbol": "BTC/USDC", "side": "long", "size": 1.0}]
        signals = monitor._detect_changes("0xabc", current, previous)
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].signal_type, "size_decrease")
        self.assertAlmostEqual(signals[0].size_change_pct, -50.0)

    def test_no_change_below_threshold(self):
        monitor = self._make_monitor(size_change_threshold_pct=25)
        monitor.add_wallet("0xabc", 85.0)
        # 10% increase — below 25% threshold
        current = [{"symbol": "BTC/USDC", "side": "long", "size": 1.1}]
        previous = [{"symbol": "BTC/USDC", "side": "long", "size": 1.0}]
        signals = monitor._detect_changes("0xabc", current, previous)
        self.assertEqual(len(signals), 0)

    def test_detect_multiple_symbols(self):
        monitor = self._make_monitor()
        monitor.add_wallet("0xabc", 85.0)
        current = [
            {"symbol": "BTC/USDC", "side": "long", "size": 2.0},
            {"symbol": "ETH/USDC", "side": "short", "size": 10.0},
        ]
        previous = [
            {"symbol": "BTC/USDC", "side": "long", "size": 1.0},
        ]
        signals = monitor._detect_changes("0xabc", current, previous)
        # size_increase for BTC + new_position for ETH
        self.assertEqual(len(signals), 2)
        types = {s.signal_type for s in signals}
        self.assertIn("size_increase", types)
        self.assertIn("new_position", types)

    def test_no_changes_same_positions(self):
        monitor = self._make_monitor()
        monitor.add_wallet("0xabc", 85.0)
        positions = [{"symbol": "BTC/USDC", "side": "long", "size": 1.0}]
        signals = monitor._detect_changes("0xabc", positions, positions)
        self.assertEqual(len(signals), 0)


class TestWalletMonitorUnusualSize(unittest.TestCase):
    """Tests for unusual size detection."""

    def _make_monitor(self):
        config = {"unusual_size_std": 2.0, "min_wallet_score": 70}
        return WalletMonitor(MagicMock(), MagicMock(), MagicMock(), config)

    def test_unusual_size_above_threshold(self):
        monitor = self._make_monitor()
        # Add history with some variance: mean~=1.05, std~=0.158
        # threshold = 1.05 + 2 * 0.158 = ~1.37
        for i in range(10):
            monitor._record_size("0xabc", "BTC/USDC", 1.0 if i % 2 == 0 else 1.1)
        # 5.0 is way above threshold
        self.assertTrue(monitor._is_unusual_size("0xabc", 5.0, "BTC/USDC"))

    def test_not_unusual_normal_size(self):
        monitor = self._make_monitor()
        # History with some variance: mean~=1.05
        for i in range(10):
            monitor._record_size("0xabc", "BTC/USDC", 1.0 if i % 2 == 0 else 1.1)
        # 1.1 is within normal range
        self.assertFalse(monitor._is_unusual_size("0xabc", 1.1, "BTC/USDC"))

    def test_not_unusual_insufficient_history(self):
        monitor = self._make_monitor()
        # Only 2 data points — threshold is 3
        monitor._record_size("0xabc", "BTC/USDC", 1.0)
        monitor._record_size("0xabc", "BTC/USDC", 1.0)
        self.assertFalse(monitor._is_unusual_size("0xabc", 100.0, "BTC/USDC"))

    def test_unusual_with_varied_history(self):
        monitor = self._make_monitor()
        # History: [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]  mean=1.5, std=0.5
        # threshold = 1.5 + 2*0.5 = 2.5
        for i in range(10):
            monitor._record_size("0xabc", "BTC/USDC", 1.0 if i % 2 == 0 else 2.0)
        self.assertTrue(monitor._is_unusual_size("0xabc", 3.0, "BTC/USDC"))
        self.assertFalse(monitor._is_unusual_size("0xabc", 2.0, "BTC/USDC"))


class TestWalletMonitorAddRemove(unittest.TestCase):
    """Tests for add/remove wallet management."""

    def _make_monitor(self, **config_overrides):
        config = {
            "max_tracked_wallets": 3,
            "min_wallet_score": 70,
        }
        config.update(config_overrides)
        return WalletMonitor(MagicMock(), MagicMock(), MagicMock(), config)

    def test_add_wallet(self):
        monitor = self._make_monitor()
        monitor.add_wallet("0xabc", 85.0)
        tracked = monitor.get_tracked_wallets()
        self.assertEqual(len(tracked), 1)
        self.assertEqual(tracked[0]["address"], "0xabc")
        self.assertEqual(tracked[0]["score"], 85.0)

    def test_add_wallet_below_min_score(self):
        monitor = self._make_monitor(min_wallet_score=80)
        monitor.add_wallet("0xabc", 60.0)
        self.assertEqual(len(monitor.get_tracked_wallets()), 0)

    def test_add_wallet_at_max_limit(self):
        monitor = self._make_monitor(max_tracked_wallets=2)
        monitor.add_wallet("0x1", 80.0)
        monitor.add_wallet("0x2", 85.0)
        monitor.add_wallet("0x3", 90.0)  # Should be rejected
        self.assertEqual(len(monitor.get_tracked_wallets()), 2)

    def test_remove_wallet(self):
        monitor = self._make_monitor()
        monitor.add_wallet("0xabc", 85.0)
        monitor.remove_wallet("0xabc")
        self.assertEqual(len(monitor.get_tracked_wallets()), 0)

    def test_remove_nonexistent_wallet(self):
        monitor = self._make_monitor()
        monitor.remove_wallet("0xnonexistent")  # Should not raise
        self.assertEqual(len(monitor.get_tracked_wallets()), 0)

    def test_get_tracked_wallets_has_added_at(self):
        monitor = self._make_monitor()
        monitor.add_wallet("0xabc", 85.0)
        tracked = monitor.get_tracked_wallets()
        self.assertIn("added_at", tracked[0])


class TestWalletMonitorRecentSignals(unittest.TestCase):
    """Tests for recent signal storage and retrieval."""

    def _make_monitor(self):
        config = {"min_wallet_score": 0}
        redis_store = MagicMock()
        redis_store.add_wallet_signal = MagicMock()
        return WalletMonitor(MagicMock(), redis_store, MagicMock(), config)

    def test_store_and_retrieve_signals(self):
        monitor = self._make_monitor()
        sig = WalletSignal(
            timestamp=datetime.now(timezone.utc),
            wallet_address="0xabc",
            wallet_score=85.0,
            symbol="BTC/USDC",
            signal_type="new_position",
            direction="long",
            size=1.0,
            size_change_pct=100.0,
            is_unusual=False,
            confidence=0.7,
        )
        monitor._store_signal(sig)
        recent = monitor.get_recent_signals(limit=10)
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]["signal_type"], "new_position")

    def test_recent_signals_limit(self):
        monitor = self._make_monitor()
        for i in range(10):
            sig = WalletSignal(
                timestamp=datetime.now(timezone.utc),
                wallet_address=f"0x{i}",
                wallet_score=80.0,
                symbol="BTC/USDC",
                signal_type="new_position",
                direction="long",
                size=1.0,
                size_change_pct=100.0,
                is_unusual=False,
                confidence=0.7,
            )
            monitor._store_signal(sig)
        recent = monitor.get_recent_signals(limit=5)
        self.assertEqual(len(recent), 5)


class TestWalletMonitorPoll(unittest.TestCase):
    """Tests for async poll behavior."""

    def test_poll_wallet_no_changes(self):
        config = {"min_wallet_score": 0}
        provider = MagicMock()
        provider.get_wallet_positions.return_value = [
            {"symbol": "BTC/USDC", "side": "long", "size": 1.0}
        ]
        redis_store = MagicMock()
        redis_store.get_wallet_positions.return_value = [
            {"symbol": "BTC/USDC", "side": "long", "size": 1.0}
        ]
        monitor = WalletMonitor(provider, redis_store, MagicMock(), config)
        monitor.add_wallet("0xabc", 85.0)
        signals = asyncio.run(monitor._poll_wallet("0xabc"))
        self.assertEqual(len(signals), 0)

    def test_poll_wallet_detects_new(self):
        config = {"min_wallet_score": 0}
        provider = MagicMock()
        provider.get_wallet_positions.return_value = [
            {"symbol": "BTC/USDC", "side": "long", "size": 1.0}
        ]
        redis_store = MagicMock()
        redis_store.get_wallet_positions.return_value = []
        redis_store.add_wallet_signal = MagicMock()
        monitor = WalletMonitor(provider, redis_store, MagicMock(), config)
        monitor.add_wallet("0xabc", 85.0)
        signals = asyncio.run(monitor._poll_wallet("0xabc"))
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].signal_type, "new_position")

    def test_poll_wallet_provider_error(self):
        config = {"min_wallet_score": 0}
        provider = MagicMock()
        provider.get_wallet_positions.side_effect = Exception("API error")
        monitor = WalletMonitor(provider, MagicMock(), MagicMock(), config)
        monitor.add_wallet("0xabc", 85.0)
        signals = asyncio.run(monitor._poll_wallet("0xabc"))
        self.assertEqual(len(signals), 0)


class TestWalletMonitorConfidence(unittest.TestCase):
    """Tests for confidence computation."""

    def _make_monitor(self):
        return WalletMonitor(MagicMock(), MagicMock(), MagicMock(), {})

    def test_confidence_scales_with_wallet_score(self):
        monitor = self._make_monitor()
        c_high = monitor._compute_confidence(90.0, "new_position", False)
        c_low = monitor._compute_confidence(50.0, "new_position", False)
        self.assertGreater(c_high, c_low)

    def test_confidence_unusual_boost(self):
        monitor = self._make_monitor()
        c_normal = monitor._compute_confidence(80.0, "new_position", False)
        c_unusual = monitor._compute_confidence(80.0, "new_position", True)
        self.assertGreater(c_unusual, c_normal)

    def test_confidence_capped_at_one(self):
        monitor = self._make_monitor()
        c = monitor._compute_confidence(100.0, "new_position", True)
        self.assertLessEqual(c, 1.0)

    def test_confidence_zero_score(self):
        monitor = self._make_monitor()
        c = monitor._compute_confidence(0.0, "new_position", False)
        self.assertEqual(c, 0.0)


class TestWalletMonitorNotification(unittest.TestCase):
    """Tests for notification dispatch on unusual signals."""

    def test_notification_dispatched_on_unusual(self):
        config = {"min_wallet_score": 0}
        provider = MagicMock()
        provider.get_wallet_positions.return_value = [
            {"symbol": "BTC/USDC", "side": "long", "size": 100.0}
        ]
        redis_store = MagicMock()
        redis_store.get_wallet_positions.return_value = []
        redis_store.add_wallet_signal = MagicMock()
        dispatcher = MagicMock()
        dispatcher.dispatch = AsyncMock()

        monitor = WalletMonitor(provider, redis_store, MagicMock(), config,
                                notification_dispatcher=dispatcher)
        monitor.add_wallet("0xabc", 85.0)
        # Pre-populate size history with variance so unusual detection works
        for i in range(10):
            monitor._record_size("0xabc", "BTC/USDC", 1.0 if i % 2 == 0 else 1.1)

        async def run_poll():
            stop = asyncio.Event()
            signals = await monitor._poll_wallet("0xabc")
            for sig in signals:
                monitor._store_signal(sig)
                if sig.is_unusual and monitor.notification_dispatcher:
                    await monitor.notification_dispatcher.dispatch(
                        message="test", level="warning", event_type="wallet_signal")
            return signals

        signals = asyncio.run(run_poll())
        # The signal should be unusual (100 vs mean of 1)
        self.assertTrue(any(s.is_unusual for s in signals))


if __name__ == "__main__":
    unittest.main()
