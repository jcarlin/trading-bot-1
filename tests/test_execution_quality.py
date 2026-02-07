"""Tests for metrics.execution_quality.ExecutionQualityTracker."""

import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from metrics.execution_quality import ExecutionQualityTracker


class TestExecutionQualityTracker(unittest.TestCase):
    """Tests for ExecutionQualityTracker."""

    def setUp(self):
        self.mock_timescale = MagicMock()
        self.tracker = ExecutionQualityTracker(
            self.mock_timescale, "funding_rate_arb")

    def test_compute_with_fills_and_orders(self):
        """Should compute slippage from matched orders and fills."""
        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=24)

        orders = [
            {"order_id": "ord-1", "symbol": "BTC-USD", "side": "buy",
             "type": "limit", "quantity": 1.0, "price": 100.0,
             "status": "filled", "strategy_name": "funding_rate_arb"},
            {"order_id": "ord-2", "symbol": "BTC-USD", "side": "sell",
             "type": "limit", "quantity": 1.0, "price": 105.0,
             "status": "filled", "strategy_name": "funding_rate_arb"},
        ]

        fills = [
            {"time": start + timedelta(hours=1), "fill_id": "f-1",
             "order_id": "ord-1", "symbol": "BTC-USD", "side": "buy",
             "quantity": 1.0, "price": 100.05},  # 5 bps slippage
            {"time": start + timedelta(hours=2), "fill_id": "f-2",
             "order_id": "ord-2", "symbol": "BTC-USD", "side": "sell",
             "quantity": 1.0, "price": 104.90},  # ~9.5 bps slippage
        ]

        self.mock_timescale.query_orders_by_strategy.return_value = orders
        self.mock_timescale.query_fills_by_strategy.return_value = fills

        result = self.tracker.compute(window_hours=24)

        self.assertEqual(result["total_orders"], 2)
        self.assertEqual(result["total_fills"], 2)
        self.assertEqual(result["fill_rate_pct"], 100.0)
        self.assertGreater(result["avg_slippage_bps"], 0)
        self.assertEqual(result["sample_size"], 2)

    def test_compute_with_no_data(self):
        """No data should return empty metrics."""
        self.mock_timescale.query_fills_by_strategy.return_value = []
        self.mock_timescale.query_orders_by_strategy.return_value = []

        result = self.tracker.compute()
        self.assertEqual(result["total_orders"], 0)
        self.assertEqual(result["total_fills"], 0)
        self.assertEqual(result["avg_slippage_bps"], 0.0)

    def test_compute_with_rejected_orders(self):
        """Should count rejected orders."""
        orders = [
            {"order_id": "ord-1", "status": "filled", "price": 100.0},
            {"order_id": "ord-2", "status": "rejected", "price": 101.0},
            {"order_id": "ord-3", "status": "rejected", "price": 102.0},
        ]
        fills = [
            {"time": datetime.now(timezone.utc), "fill_id": "f-1",
             "order_id": "ord-1", "price": 100.1},
        ]

        self.mock_timescale.query_orders_by_strategy.return_value = orders
        self.mock_timescale.query_fills_by_strategy.return_value = fills

        result = self.tracker.compute()
        self.assertEqual(result["rejected_count"], 2)
        self.assertEqual(result["total_orders"], 3)

    def test_fill_rate_calculation(self):
        """Fill rate should be fills/orders * 100."""
        orders = [
            {"order_id": "ord-1", "status": "filled", "price": 100.0},
            {"order_id": "ord-2", "status": "cancelled", "price": 101.0},
        ]
        fills = [
            {"time": datetime.now(timezone.utc), "fill_id": "f-1",
             "order_id": "ord-1", "price": 100.0},
        ]

        self.mock_timescale.query_orders_by_strategy.return_value = orders
        self.mock_timescale.query_fills_by_strategy.return_value = fills

        result = self.tracker.compute()
        self.assertEqual(result["fill_rate_pct"], 50.0)

    def test_result_structure(self):
        """Result should have all expected keys."""
        self.mock_timescale.query_fills_by_strategy.return_value = []
        self.mock_timescale.query_orders_by_strategy.return_value = []

        result = self.tracker.compute()
        expected_keys = [
            "avg_slippage_bps", "max_slippage_bps", "fill_rate_pct",
            "avg_latency_ms", "p95_latency_ms", "p99_latency_ms",
            "rejected_count", "total_orders", "total_fills", "sample_size",
        ]
        for key in expected_keys:
            self.assertIn(key, result)

    def test_exception_handling(self):
        """Exceptions should be handled gracefully."""
        self.mock_timescale.query_fills_by_strategy.side_effect = Exception("DB error")
        self.mock_timescale.query_orders_by_strategy.side_effect = Exception("DB error")

        result = self.tracker.compute()
        self.assertEqual(result["total_orders"], 0)

    def test_max_slippage(self):
        """Max slippage should be the worst case."""
        orders = [
            {"order_id": "ord-1", "price": 100.0, "status": "filled"},
            {"order_id": "ord-2", "price": 100.0, "status": "filled"},
        ]
        fills = [
            {"time": datetime.now(timezone.utc), "order_id": "ord-1",
             "price": 100.05, "fill_id": "f-1"},  # 5 bps
            {"time": datetime.now(timezone.utc), "order_id": "ord-2",
             "price": 100.20, "fill_id": "f-2"},  # 20 bps
        ]

        self.mock_timescale.query_orders_by_strategy.return_value = orders
        self.mock_timescale.query_fills_by_strategy.return_value = fills

        result = self.tracker.compute()
        self.assertGreater(result["max_slippage_bps"], result["avg_slippage_bps"])


if __name__ == "__main__":
    unittest.main()
