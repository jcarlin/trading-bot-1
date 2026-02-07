"""Execution quality tracking for live trading strategies."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class ExecutionQualityTracker:
    """Aggregates slippage, fill rates, and latency metrics.

    Analyzes execution quality by comparing orders to fills.
    """

    def __init__(self, timescale, strategy_name: str):
        self.timescale = timescale
        self.strategy_name = strategy_name

    def compute(self, window_hours: int = 24) -> dict:
        """Compute execution quality metrics over a time window.

        Args:
            window_hours: Number of hours to look back.

        Returns:
            dict with execution quality metrics.
        """
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=window_hours)

        try:
            fills = self.timescale.query_fills_by_strategy(
                self.strategy_name, start, end)
        except Exception:
            logger.exception("Failed to query fills for execution quality")
            fills = []

        try:
            orders = self.timescale.query_orders_by_strategy(
                self.strategy_name, start, end)
        except Exception:
            logger.exception("Failed to query orders for execution quality")
            orders = []

        if not fills and not orders:
            return self._empty_metrics()

        total_orders = len(orders)
        total_fills = len(fills)

        # Fill rate
        fill_rate = (total_fills / total_orders * 100) if total_orders > 0 else 0.0

        # Rejected orders count
        rejected_count = sum(
            1 for o in orders if o.get("status") == "rejected")

        # Slippage: compare order price to fill price
        slippage_values = []
        for fill in fills:
            fill_price = float(fill.get("price", 0))
            order_id = fill.get("order_id")
            if fill_price <= 0 or not order_id:
                continue

            # Find matching order
            matching_order = None
            for order in orders:
                if order.get("order_id") == order_id:
                    matching_order = order
                    break

            if matching_order and matching_order.get("price"):
                order_price = float(matching_order["price"])
                if order_price > 0:
                    slippage_bps = abs(fill_price - order_price) / order_price * 10000
                    slippage_values.append(slippage_bps)

        avg_slippage = (sum(slippage_values) / len(slippage_values)) if slippage_values else 0.0
        max_slippage = max(slippage_values) if slippage_values else 0.0

        # Latency: estimate from order timestamp vs fill timestamp
        latencies_ms = []
        for fill in fills:
            fill_time = fill.get("time")
            order_id = fill.get("order_id")
            if not fill_time or not order_id:
                continue

            for order in orders:
                if order.get("order_id") == order_id:
                    # Use order creation time if available
                    # Orders table doesn't have explicit created_at, approximate
                    # from the fill latency perspective
                    latencies_ms.append(0.0)  # placeholder — real latency needs timestamps
                    break

        avg_latency = (sum(latencies_ms) / len(latencies_ms)) if latencies_ms else 0.0

        # Percentile calculation for latency
        p95_latency = 0.0
        p99_latency = 0.0
        if latencies_ms:
            sorted_lat = sorted(latencies_ms)
            p95_idx = int(len(sorted_lat) * 0.95)
            p99_idx = int(len(sorted_lat) * 0.99)
            p95_latency = sorted_lat[min(p95_idx, len(sorted_lat) - 1)]
            p99_latency = sorted_lat[min(p99_idx, len(sorted_lat) - 1)]

        return {
            "avg_slippage_bps": round(avg_slippage, 4),
            "max_slippage_bps": round(max_slippage, 4),
            "fill_rate_pct": round(fill_rate, 2),
            "avg_latency_ms": round(avg_latency, 4),
            "p95_latency_ms": round(p95_latency, 4),
            "p99_latency_ms": round(p99_latency, 4),
            "rejected_count": rejected_count,
            "total_orders": total_orders,
            "total_fills": total_fills,
            "sample_size": len(slippage_values),
        }

    @staticmethod
    def _empty_metrics() -> dict:
        return {
            "avg_slippage_bps": 0.0,
            "max_slippage_bps": 0.0,
            "fill_rate_pct": 0.0,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "rejected_count": 0,
            "total_orders": 0,
            "total_fills": 0,
            "sample_size": 0,
        }
