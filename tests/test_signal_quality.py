"""Tests for metrics.signal_quality.SignalQualityAssessor."""

import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from metrics.signal_quality import SignalQualityAssessor


class TestSignalQualityAssessor(unittest.TestCase):
    """Tests for SignalQualityAssessor."""

    def setUp(self):
        self.mock_timescale = MagicMock()
        self.assessor = SignalQualityAssessor(
            self.mock_timescale, "funding_rate_arb", "BTC/USDC")

    def _make_candles(self, start: datetime, hours: int,
                      base_price: float = 100.0,
                      trend: float = 0.0) -> list[dict]:
        """Create mock candle data."""
        candles = []
        for i in range(hours):
            t = start + timedelta(hours=i)
            price = base_price + trend * i
            candles.append({
                "time": t,
                "open": price - 0.1,
                "high": price + 0.5,
                "low": price - 0.5,
                "close": price,
                "volume": 1000.0,
            })
        return candles

    def test_assess_with_accurate_long_signals(self):
        """Long entries followed by price increases should show high accuracy."""
        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=24)

        # Entry fill (long buy)
        fills = [{
            "time": start + timedelta(hours=2),
            "side": "buy",
            "price": 100.0,
            "quantity": 1.0,
            "closed_pnl": 0.0,  # entry fill
        }]

        # Price goes up after entry
        candles = self._make_candles(start, 30, base_price=100.0, trend=0.5)

        self.mock_timescale.query_fills_by_strategy.return_value = fills
        self.mock_timescale.query_candles.return_value = candles

        result = self.assessor.assess(window_hours=24)

        self.assertGreater(result["signal_accuracy_pct"], 0)
        self.assertGreater(result["avg_favorable_move_pct"], 0)
        self.assertEqual(result["sample_size"], 1)

    def test_assess_with_no_fills(self):
        """No fills should return empty assessment."""
        self.mock_timescale.query_fills_by_strategy.return_value = []

        result = self.assessor.assess()
        self.assertEqual(result["signal_accuracy_pct"], 0.0)
        self.assertEqual(result["sample_size"], 0)

    def test_assess_with_no_candles(self):
        """No candles should return empty assessment."""
        now = datetime.now(timezone.utc)
        fills = [{
            "time": now - timedelta(hours=2),
            "side": "buy",
            "price": 100.0,
            "quantity": 1.0,
            "closed_pnl": 0.0,
        }]
        self.mock_timescale.query_fills_by_strategy.return_value = fills
        self.mock_timescale.query_candles.return_value = []

        result = self.assessor.assess()
        self.assertEqual(result["sample_size"], 0)

    def test_assess_skips_exit_fills(self):
        """Fills with non-zero closed_pnl should be skipped as exits."""
        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=24)

        fills = [{
            "time": start + timedelta(hours=2),
            "side": "sell",
            "price": 105.0,
            "quantity": 1.0,
            "closed_pnl": 5.0,  # exit fill — should be skipped
        }]
        candles = self._make_candles(start, 30)

        self.mock_timescale.query_fills_by_strategy.return_value = fills
        self.mock_timescale.query_candles.return_value = candles

        result = self.assessor.assess()
        self.assertEqual(result["sample_size"], 0)

    def test_assess_short_entry(self):
        """Short entries followed by price decreases should be accurate."""
        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=24)

        fills = [{
            "time": start + timedelta(hours=2),
            "side": "sell",
            "price": 100.0,
            "quantity": 1.0,
            "closed_pnl": 0.0,
        }]

        # Price goes down after entry
        candles = self._make_candles(start, 30, base_price=100.0, trend=-0.5)

        self.mock_timescale.query_fills_by_strategy.return_value = fills
        self.mock_timescale.query_candles.return_value = candles

        result = self.assessor.assess()
        self.assertGreater(result["signal_accuracy_pct"], 0)

    def test_false_positive_rate(self):
        """False positive rate should be 100 - accuracy."""
        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=24)

        fills = [{
            "time": start + timedelta(hours=2),
            "side": "buy",
            "price": 100.0,
            "quantity": 1.0,
            "closed_pnl": 0.0,
        }]
        candles = self._make_candles(start, 30, base_price=100.0, trend=0.5)

        self.mock_timescale.query_fills_by_strategy.return_value = fills
        self.mock_timescale.query_candles.return_value = candles

        result = self.assessor.assess()
        self.assertAlmostEqual(
            result["signal_accuracy_pct"] + result["false_positive_rate"],
            100.0, places=2)

    def test_result_structure(self):
        """Result should have all expected keys."""
        self.mock_timescale.query_fills_by_strategy.return_value = []
        result = self.assessor.assess()

        expected_keys = [
            "signal_accuracy_pct", "avg_favorable_move_pct",
            "avg_adverse_move_pct", "timing_score",
            "false_positive_rate", "sample_size", "total_entries",
        ]
        for key in expected_keys:
            self.assertIn(key, result)

    def test_exception_handling(self):
        """Exceptions from timescale should be handled gracefully."""
        self.mock_timescale.query_fills_by_strategy.side_effect = Exception("DB error")
        result = self.assessor.assess()
        self.assertEqual(result["sample_size"], 0)


if __name__ == "__main__":
    unittest.main()
