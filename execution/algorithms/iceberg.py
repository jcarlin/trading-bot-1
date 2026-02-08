"""Iceberg execution algorithm."""

import asyncio
import logging
from datetime import datetime, timezone

from .base import ExecutionAlgorithm

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "visible_pct": 0.20,
    "min_visible_qty": 0.001,
    "price_offset_bps": 3,
    "refill_delay_s": 2,
}


class IcebergAlgorithm(ExecutionAlgorithm):
    """Iceberg order execution.

    Displays only a fraction of the total order to the market.
    Submits a ``visible_qty`` limit order, then refills after each
    fill until the total quantity is executed.

    Config
    ------
    visible_pct : float
        Fraction of total to show per slice (default 0.20).
    min_visible_qty : float
        Minimum visible size (default 0.001).
    price_offset_bps : float
        Offset from mid for limit price in basis points (default 3).
    refill_delay_s : float
        Seconds to wait between refills (default 2).
    """

    async def execute(self, order, exchange, config: dict) -> list:
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        visible_pct = float(cfg["visible_pct"])
        min_qty = float(cfg["min_visible_qty"])
        offset_bps = float(cfg["price_offset_bps"])
        refill_delay = float(cfg["refill_delay_s"])

        visible_qty = self._compute_visible_qty(
            order.quantity, visible_pct, min_qty
        )

        remaining = order.quantity
        fills = []

        while remaining > 0:
            slice_qty = min(visible_qty, remaining)
            if slice_qty <= 0:
                break

            fill = await self._execute_visible(
                slice_qty=slice_qty,
                symbol=order.symbol,
                side=order.side,
                exchange=exchange,
                offset_bps=offset_bps,
            )

            if fill:
                fills.append(fill)
                remaining -= fill.get("quantity", slice_qty)
            else:
                # Failed to fill this slice — break to avoid infinite loop
                logger.warning(
                    "Iceberg slice failed, %s remaining of %s",
                    remaining, order.quantity,
                )
                break

            if remaining > 0 and refill_delay > 0:
                await asyncio.sleep(refill_delay)

        return fills

    def _compute_visible_qty(
        self, total_qty: float, visible_pct: float, min_qty: float
    ) -> float:
        """Compute the visible order quantity."""
        visible = total_qty * visible_pct
        return max(visible, min_qty)

    async def _execute_visible(
        self, slice_qty, symbol, side, exchange, offset_bps
    ) -> dict:
        """Execute a single visible slice."""
        from core.types import Side

        try:
            ticker = exchange.get_ticker(symbol)
            mid = ticker.get("mid", ticker.get("last", 0))
        except Exception:
            mid = 0

        if mid <= 0:
            return await self._market_fallback(slice_qty, symbol, side, exchange)

        offset = mid * offset_bps / 10000.0
        if side == Side.BUY:
            limit_price = mid + offset
        else:
            limit_price = mid - offset

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
            return await self._market_fallback(slice_qty, symbol, side, exchange)

        # Wait briefly for fill
        await asyncio.sleep(1)
        try:
            status = await exchange.get_order_status(order_id)
            if status and status.get("status") == "filled":
                return {
                    "price": status.get("fill_price", limit_price),
                    "quantity": slice_qty,
                    "timestamp": datetime.now(timezone.utc),
                    "fee": status.get("fee", 0.0),
                    "order_id": order_id,
                    "algo": "iceberg",
                    "child_type": "limit",
                }
        except Exception:
            pass

        # Fall back to market for this slice
        try:
            exchange.cancel_order(str(order_id), symbol)
        except Exception:
            pass

        return await self._market_fallback(slice_qty, symbol, side, exchange)

    async def _market_fallback(self, qty, symbol, side, exchange) -> dict:
        """Place a market order as fallback for a single slice."""
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
                "algo": "iceberg",
                "child_type": "market_fallback",
            }
        except Exception:
            logger.exception("Iceberg market fallback failed")
            return None

    def get_metadata(self) -> dict:
        return {
            "name": "iceberg",
            "description": (
                "Iceberg order: shows only a fraction of total order, "
                "refilling after each fill to hide true size"
            ),
            "default_config": DEFAULT_CONFIG.copy(),
        }
