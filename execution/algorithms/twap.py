"""Time-Weighted Average Price (TWAP) execution algorithm."""

import asyncio
import logging
import time
from datetime import datetime, timezone

from .base import ExecutionAlgorithm

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "duration_seconds": 300,
    "num_slices": 5,
    "spread_buffer_bps": 2,
    "child_timeout_s": 30,
}


class TWAPAlgorithm(ExecutionAlgorithm):
    """Time-Weighted Average Price execution.

    Splits a parent order into N equal child orders spread evenly
    over a total duration.  Each child is submitted as a limit order
    at mid +/- buffer.  If a child is not filled within
    ``child_timeout_s``, it is cancelled and re-submitted as a
    market order.

    Config
    ------
    duration_seconds : int
        Total execution window (default 300).
    num_slices : int
        Number of child orders (default 5).
    spread_buffer_bps : float
        Offset from mid for limit price in basis points (default 2).
    child_timeout_s : int
        Seconds to wait before falling back to market (default 30).
    """

    async def execute(self, order, exchange, config: dict) -> list:
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        num_slices = max(1, int(cfg["num_slices"]))
        duration = max(0, float(cfg["duration_seconds"]))
        child_timeout = float(cfg["child_timeout_s"])
        buffer_bps = float(cfg["spread_buffer_bps"])

        slices = self._compute_slices(order.quantity, num_slices)
        interval = duration / num_slices if num_slices > 1 else 0

        fills = []
        for i, slice_qty in enumerate(slices):
            if slice_qty <= 0:
                continue

            fill = await self._execute_slice(
                slice_qty=slice_qty,
                symbol=order.symbol,
                side=order.side,
                exchange=exchange,
                config={
                    "buffer_bps": buffer_bps,
                    "child_timeout_s": child_timeout,
                },
            )
            if fill:
                fills.append(fill)

            # Wait between slices (skip after last)
            if i < len(slices) - 1 and interval > 0:
                await asyncio.sleep(interval)

        return fills

    def _compute_slices(self, total_qty: float, num_slices: int) -> list[float]:
        """Split total_qty into num_slices roughly equal parts."""
        if total_qty <= 0 or num_slices <= 0:
            return []
        base = total_qty / num_slices
        slices = [base] * num_slices
        # Distribute any floating-point remainder to the last slice
        remainder = total_qty - sum(slices)
        if remainder != 0:
            slices[-1] += remainder
        return slices

    async def _execute_slice(
        self, slice_qty, symbol, side, exchange, config
    ) -> dict:
        """Execute a single TWAP slice.

        Places a limit order at mid +/- buffer.  If not filled within
        timeout, cancels and places a market order.
        """
        buffer_bps = config.get("buffer_bps", 2)
        timeout = config.get("child_timeout_s", 30)

        try:
            ticker = exchange.get_ticker(symbol)
            mid = ticker.get("mid", ticker.get("last", 0))
        except Exception:
            mid = 0

        if mid <= 0:
            # Fall back to market
            return await self._market_fallback(slice_qty, symbol, side, exchange)

        # Compute limit price
        from core.types import Side
        offset = mid * buffer_bps / 10000.0
        if side == Side.BUY:
            limit_price = mid + offset
        else:
            limit_price = mid - offset

        # Place limit order
        try:
            order_id = await exchange.place_limit_order(
                symbol=symbol,
                side=side,
                quantity=slice_qty,
                price=limit_price,
            )
        except (NotImplementedError, AttributeError):
            return await self._market_fallback(slice_qty, symbol, side, exchange)
        except Exception:
            logger.exception("TWAP slice limit order failed")
            return await self._market_fallback(slice_qty, symbol, side, exchange)

        # Wait for fill
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            try:
                status = await exchange.get_order_status(order_id)
                if status and status.get("status") == "filled":
                    return {
                        "price": status.get("fill_price", limit_price),
                        "quantity": slice_qty,
                        "timestamp": datetime.now(timezone.utc),
                        "fee": status.get("fee", 0.0),
                        "order_id": order_id,
                        "algo": "twap",
                        "child_type": "limit",
                    }
            except (NotImplementedError, AttributeError):
                break
            except Exception:
                break
            await asyncio.sleep(1)

        # Timeout — cancel and market fallback
        try:
            exchange.cancel_order(str(order_id), symbol)
        except Exception:
            pass

        return await self._market_fallback(slice_qty, symbol, side, exchange)

    async def _market_fallback(self, qty, symbol, side, exchange) -> dict:
        """Place a market order as fallback."""
        from core.models import Order
        from core.types import OrderType

        order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=qty,
        )
        try:
            fill = exchange.place_order(order)
            return {
                "price": fill.fill_price,
                "quantity": fill.quantity,
                "timestamp": fill.timestamp,
                "fee": fill.commission,
                "order_id": fill.order_id,
                "algo": "twap",
                "child_type": "market_fallback",
            }
        except Exception:
            logger.exception("TWAP market fallback failed")
            return None

    def get_metadata(self) -> dict:
        return {
            "name": "twap",
            "description": (
                "Time-Weighted Average Price: splits order into equal "
                "slices over a duration with limit-then-market fallback"
            ),
            "default_config": DEFAULT_CONFIG.copy(),
        }
