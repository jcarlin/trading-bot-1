#!/usr/bin/env python3
"""Tests for signal aggregator."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from core.models import Signal
from core.types import SignalType
from strategy.signal_aggregator import SignalAggregator


class TestSignalAggregator(unittest.TestCase):

    def setUp(self):
        self.strategy_names = ["momentum", "mean_reversion", "funding_rate_arb"]
        self.agg = SignalAggregator(self.strategy_names)
        self.base_time = datetime(2025, 6, 1, tzinfo=timezone.utc)

    def _make_signal(self, signal_type, price=100.0):
        return Signal(
            signal_type=signal_type,
            price=price,
            timestamp=self.base_time,
            size=1.0,
        )

    def test_record_and_get_latest(self):
        """Record a signal and retrieve it."""
        sig = self._make_signal(SignalType.ENTER_LONG)
        self.agg.record_signal("momentum", sig)
        latest = self.agg.get_latest_signals()
        self.assertIn("momentum", latest)
        self.assertEqual(latest["momentum"]["signal_type"], "enter_long")
        self.assertEqual(latest["momentum"]["direction"], "bullish")

    def test_multiple_strategies(self):
        """Record signals from multiple strategies."""
        self.agg.record_signal("momentum", self._make_signal(SignalType.ENTER_LONG))
        self.agg.record_signal("mean_reversion", self._make_signal(SignalType.ENTER_SHORT))
        self.agg.record_signal("funding_rate_arb", self._make_signal(SignalType.HOLD))

        latest = self.agg.get_latest_signals()
        self.assertEqual(len(latest), 3)
        self.assertEqual(latest["momentum"]["direction"], "bullish")
        self.assertEqual(latest["mean_reversion"]["direction"], "bearish")
        self.assertEqual(latest["funding_rate_arb"]["direction"], "neutral")

    def test_latest_overwrites_previous(self):
        """Later signal overwrites earlier one for same strategy."""
        self.agg.record_signal("momentum", self._make_signal(SignalType.ENTER_LONG))
        self.agg.record_signal("momentum", self._make_signal(SignalType.ENTER_SHORT))
        latest = self.agg.get_latest_signals()
        self.assertEqual(latest["momentum"]["direction"], "bearish")

    def test_consensus_all_bullish(self):
        """Consensus when all strategies are bullish."""
        for name in self.strategy_names:
            self.agg.record_signal(name, self._make_signal(SignalType.ENTER_LONG))
        consensus = self.agg.get_consensus()
        self.assertEqual(consensus["bullish_count"], 3)
        self.assertEqual(consensus["bearish_count"], 0)
        self.assertEqual(consensus["neutral_count"], 0)
        self.assertGreater(consensus["weighted_direction"], 0)

    def test_consensus_all_bearish(self):
        """Consensus when all strategies are bearish."""
        for name in self.strategy_names:
            self.agg.record_signal(name, self._make_signal(SignalType.ENTER_SHORT))
        consensus = self.agg.get_consensus()
        self.assertEqual(consensus["bullish_count"], 0)
        self.assertEqual(consensus["bearish_count"], 3)
        self.assertLess(consensus["weighted_direction"], 0)

    def test_consensus_mixed(self):
        """Consensus with mixed signals."""
        self.agg.record_signal("momentum", self._make_signal(SignalType.ENTER_LONG))
        self.agg.record_signal("mean_reversion", self._make_signal(SignalType.ENTER_SHORT))
        self.agg.record_signal("funding_rate_arb", self._make_signal(SignalType.HOLD))
        consensus = self.agg.get_consensus()
        self.assertEqual(consensus["bullish_count"], 1)
        self.assertEqual(consensus["bearish_count"], 1)
        self.assertEqual(consensus["neutral_count"], 1)

    def test_consensus_all_neutral(self):
        """Consensus when all strategies hold."""
        for name in self.strategy_names:
            self.agg.record_signal(name, self._make_signal(SignalType.HOLD))
        consensus = self.agg.get_consensus()
        self.assertEqual(consensus["bullish_count"], 0)
        self.assertEqual(consensus["bearish_count"], 0)
        self.assertEqual(consensus["neutral_count"], 3)
        self.assertAlmostEqual(consensus["weighted_direction"], 0.0)

    def test_consensus_empty_aggregator(self):
        """Consensus on empty aggregator."""
        consensus = self.agg.get_consensus()
        self.assertEqual(consensus["bullish_count"], 0)
        self.assertEqual(consensus["bearish_count"], 0)
        self.assertEqual(consensus["neutral_count"], 0)
        self.assertEqual(consensus["weighted_direction"], 0.0)
        self.assertEqual(consensus["total_weight"], 0.0)

    def test_signal_history_basic(self):
        """Signal history records multiple signals per strategy."""
        self.agg.record_signal("momentum", self._make_signal(SignalType.ENTER_LONG))
        self.agg.record_signal("momentum", self._make_signal(SignalType.HOLD))
        self.agg.record_signal("momentum", self._make_signal(SignalType.EXIT_LONG))

        history = self.agg.get_signal_history(lookback_ticks=10)
        self.assertEqual(len(history["momentum"]), 3)
        self.assertEqual(history["momentum"][0]["signal_type"], "enter_long")
        self.assertEqual(history["momentum"][2]["signal_type"], "exit_long")

    def test_signal_history_lookback_limit(self):
        """History is trimmed to lookback_ticks."""
        for _ in range(10):
            self.agg.record_signal("momentum", self._make_signal(SignalType.HOLD))
        history = self.agg.get_signal_history(lookback_ticks=5)
        self.assertEqual(len(history["momentum"]), 5)

    def test_signal_history_empty_strategy(self):
        """History for strategy with no signals returns empty list."""
        history = self.agg.get_signal_history()
        self.assertEqual(len(history["momentum"]), 0)

    def test_ring_buffer_max_size(self):
        """Ring buffer respects max_history size."""
        agg = SignalAggregator(["test"], max_history=5)
        for i in range(10):
            agg.record_signal("test", self._make_signal(SignalType.HOLD, price=float(i)))
        history = agg.get_signal_history(lookback_ticks=100)
        self.assertEqual(len(history["test"]), 5)
        # Should contain last 5
        self.assertEqual(history["test"][0]["price"], 5.0)
        self.assertEqual(history["test"][4]["price"], 9.0)

    def test_health_score_stored(self):
        """Health score is stored in signal record."""
        self.agg.record_signal("momentum", self._make_signal(SignalType.HOLD),
                               health_score=85.0)
        latest = self.agg.get_latest_signals()
        self.assertEqual(latest["momentum"]["health_score"], 85.0)

    def test_health_score_affects_consensus_weight(self):
        """Higher health score gives more weight in consensus."""
        self.agg.record_signal("momentum", self._make_signal(SignalType.ENTER_LONG),
                               health_score=90.0)
        self.agg.record_signal("mean_reversion", self._make_signal(SignalType.ENTER_SHORT),
                               health_score=10.0)
        consensus = self.agg.get_consensus()
        # Bullish should win because momentum has much higher health
        self.assertGreater(consensus["weighted_direction"], 0)

    def test_unknown_strategy_creates_history(self):
        """Recording a signal for unknown strategy creates its history."""
        self.agg.record_signal("new_strategy", self._make_signal(SignalType.ENTER_LONG))
        latest = self.agg.get_latest_signals()
        self.assertIn("new_strategy", latest)

    def test_exit_long_is_bearish(self):
        """EXIT_LONG direction maps to bearish."""
        self.agg.record_signal("momentum", self._make_signal(SignalType.EXIT_LONG))
        latest = self.agg.get_latest_signals()
        self.assertEqual(latest["momentum"]["direction"], "bearish")

    def test_exit_short_is_bullish(self):
        """EXIT_SHORT direction maps to bullish."""
        self.agg.record_signal("momentum", self._make_signal(SignalType.EXIT_SHORT))
        latest = self.agg.get_latest_signals()
        self.assertEqual(latest["momentum"]["direction"], "bullish")

    def test_redis_persistence(self):
        """Signal is persisted to Redis when store is available."""
        mock_redis = MagicMock()
        mock_redis._r = MagicMock()
        agg = SignalAggregator(["test"], redis_store=mock_redis)
        agg.record_signal("test", self._make_signal(SignalType.HOLD))
        mock_redis._r.set.assert_called_once()

    def test_redis_failure_does_not_raise(self):
        """Redis failure is handled gracefully."""
        mock_redis = MagicMock()
        mock_redis._r = MagicMock()
        mock_redis._r.set.side_effect = Exception("connection refused")
        agg = SignalAggregator(["test"], redis_store=mock_redis)
        # Should not raise
        agg.record_signal("test", self._make_signal(SignalType.HOLD))
        latest = agg.get_latest_signals()
        self.assertIn("test", latest)

    def test_metadata_preserved(self):
        """Signal metadata is preserved in record."""
        sig = Signal(
            signal_type=SignalType.ENTER_LONG,
            price=100.0,
            timestamp=self.base_time,
            metadata={"strategy": "test", "custom_key": 42},
        )
        self.agg.record_signal("momentum", sig)
        latest = self.agg.get_latest_signals()
        self.assertEqual(latest["momentum"]["metadata"]["custom_key"], 42)


if __name__ == "__main__":
    unittest.main()
