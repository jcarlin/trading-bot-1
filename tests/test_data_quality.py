"""Tests for DataQualityMonitor."""

import time
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from data.quality.monitor import DataQualityMonitor


class TestStalenessDetection(unittest.TestCase):
    """Test that stale feeds are detected correctly."""

    def setUp(self):
        self.redis = MagicMock()
        self.monitor = DataQualityMonitor(
            self.redis, staleness_threshold_s=0.1,
        )

    def test_staleness_detection(self):
        """A source that hasn't updated past the threshold is stale."""
        self.monitor.register_source("orderbook", "BTC/USDC", 0.05)

        now = datetime.now(timezone.utc)
        self.monitor.record_update(
            "orderbook", "BTC/USDC",
            exchange_ts=now, receipt_ts=now,
        )

        # Immediately should be ok
        status = self.monitor.get_status("orderbook", "BTC/USDC")
        self.assertEqual(status["status"], "ok")

        # Wait past the threshold
        time.sleep(0.15)
        status = self.monitor.get_status("orderbook", "BTC/USDC")
        self.assertEqual(status["status"], "stale")

    def test_fresh_data_is_ok(self):
        """A source updated within the threshold should be ok."""
        self.monitor.register_source("trades", "ETH/USDC", 1.0)

        now = datetime.now(timezone.utc)
        self.monitor.record_update(
            "trades", "ETH/USDC",
            exchange_ts=now, receipt_ts=now,
        )
        status = self.monitor.get_status("trades", "ETH/USDC")
        self.assertEqual(status["status"], "ok")


class TestGapDetection(unittest.TestCase):
    """Test sequence number gap detection."""

    def setUp(self):
        self.redis = MagicMock()
        self.monitor = DataQualityMonitor(self.redis)

    def test_gap_detection(self):
        """Missing sequence numbers should trigger gap status."""
        self.monitor.register_source("orderbook", "BTC/USDC", 1.0)
        now = datetime.now(timezone.utc)

        # Normal sequence
        self.monitor.record_update(
            "orderbook", "BTC/USDC", now, now, seq_num=1,
        )
        status = self.monitor.get_status("orderbook", "BTC/USDC")
        self.assertEqual(status["status"], "ok")

        # Gap: skip seq_num 2
        self.monitor.record_update(
            "orderbook", "BTC/USDC", now, now, seq_num=3,
        )
        status = self.monitor.get_status("orderbook", "BTC/USDC")
        self.assertEqual(status["status"], "gap")
        self.assertEqual(status["gaps_detected"], 1)

    def test_no_gap_sequential(self):
        """Sequential seq_nums should not trigger gap."""
        self.monitor.register_source("trades", "BTC/USDC", 1.0)
        now = datetime.now(timezone.utc)

        for i in range(1, 5):
            self.monitor.record_update(
                "trades", "BTC/USDC", now, now, seq_num=i,
            )
        status = self.monitor.get_status("trades", "BTC/USDC")
        self.assertEqual(status["gaps_detected"], 0)


class TestAnomalyDetection(unittest.TestCase):
    """Test price anomaly detection."""

    def setUp(self):
        self.redis = MagicMock()
        self.monitor = DataQualityMonitor(
            self.redis, anomaly_sigma=3.0,
        )

    def test_anomaly_detection(self):
        """An extreme price should trigger anomaly status."""
        self.monitor.register_source("trades", "BTC/USDC", 1.0)
        now = datetime.now(timezone.utc)

        # Feed stable prices to build a baseline
        for price in [50000, 50010, 49990, 50005, 49995,
                      50000, 50010, 49990, 50005, 49995]:
            self.monitor.record_update(
                "trades", "BTC/USDC", now, now, price=price,
            )

        # Check that baseline is ok
        status = self.monitor.get_status("trades", "BTC/USDC")
        self.assertEqual(status["anomalies_detected"], 0)

        # Extreme price (well beyond 3 sigma)
        self.monitor.record_update(
            "trades", "BTC/USDC", now, now, price=60000,
        )
        status = self.monitor.get_status("trades", "BTC/USDC")
        self.assertGreater(status["anomalies_detected"], 0)
        self.assertEqual(status["status"], "anomaly")

    def test_normal_price_no_anomaly(self):
        """Prices within normal range should not trigger anomaly."""
        self.monitor.register_source("trades", "ETH/USDC", 1.0)
        now = datetime.now(timezone.utc)

        for price in [3000, 3001, 2999, 3002, 2998]:
            self.monitor.record_update(
                "trades", "ETH/USDC", now, now, price=price,
            )
        status = self.monitor.get_status("trades", "ETH/USDC")
        self.assertEqual(status["anomalies_detected"], 0)


class TestMultipleSources(unittest.TestCase):
    """Test independent tracking of multiple sources."""

    def setUp(self):
        self.redis = MagicMock()
        self.monitor = DataQualityMonitor(
            self.redis, staleness_threshold_s=0.1,
        )

    def test_multiple_sources_independent(self):
        """Each source/symbol pair is tracked independently."""
        self.monitor.register_source("orderbook", "BTC/USDC", 0.05)
        self.monitor.register_source("trades", "ETH/USDC", 0.05)

        now = datetime.now(timezone.utc)

        # Update only one source
        self.monitor.record_update(
            "orderbook", "BTC/USDC", now, now,
        )

        # Both should exist
        all_status = self.monitor.get_all_status()
        self.assertIn("orderbook:BTC/USDC", all_status)
        self.assertIn("trades:ETH/USDC", all_status)

        # Updated one is ok, other is pending
        self.assertEqual(
            all_status["orderbook:BTC/USDC"]["status"], "ok"
        )
        self.assertEqual(
            all_status["trades:ETH/USDC"]["status"], "pending"
        )

    def test_unknown_source_returns_unknown(self):
        """Querying an unregistered source returns unknown."""
        status = self.monitor.get_status("nonexistent", "FOO/BAR")
        self.assertEqual(status["status"], "unknown")


class TestRedisIntegration(unittest.TestCase):
    """Verify Redis calls are made correctly."""

    def test_record_update_calls_redis(self):
        """Each record_update should write quality data to Redis."""
        redis = MagicMock()
        monitor = DataQualityMonitor(redis)
        monitor.register_source("trades", "BTC/USDC", 1.0)

        now = datetime.now(timezone.utc)
        monitor.record_update("trades", "BTC/USDC", now, now)

        redis.set_data_quality.assert_called_once()
        call_kw = redis.set_data_quality.call_args[1]
        self.assertEqual(call_kw["source"], "trades")
        self.assertEqual(call_kw["symbol"], "BTC/USDC")
        self.assertEqual(call_kw["status"], "ok")
        self.assertEqual(call_kw["ttl"], 300)


if __name__ == "__main__":
    unittest.main()
