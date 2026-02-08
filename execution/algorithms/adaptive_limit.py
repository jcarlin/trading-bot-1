"""Adaptive limit order execution algorithm."""

import asyncio
import logging
import time
from datetime import datetime, timezone

from .base import ExecutionAlgorithm

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "initial_offset_bps": 5,
    "step_bps": 1,
    "adjust_interval_s": 5,
    "max_wait_s": 60,
}


class AdaptiveLimitAlgorithm(ExecutionAlgorithm):
    """Adaptive limit order that walks price toward the market.

    Places an initial limit order at a favourable offset from mid,
    then periodically adjusts (walks) the price closer to the market
    in increments of ``step_bps`` until the order is filled or
    ``max_wait_s`` is exceeded, at which point it falls back to a
    market order.

    Config
    ------
    initial_offset_bps : float
        Starting offset from mid price in basis points (default 5).
    step_bps : float
        Price adjustment step in basis points (default 1).
    adjust_interval_s : float
        Seconds between price adjustments (default 5).
    max_wait_s : float
        Maximum wait before market fallback (default 60).
    """

    async def execute(self, order, exchange, config: dict) -> list:
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        initial_offset = float(cfg["initial_offset_bps"])
        max_wait = float(cfg["max_wait_s"])

        try:
            ticker = exchange.get_ticker(order.symbol)
            mid_price = ticker.get("mid", ticker.get("last", 0))
        except Exception:
            mid_price = 0

        if mid_price <= 0:
            fill = await self._market_fallback(order, exchange)
            return [fill] if fill else []

        limit_price = self._compute_limit_price(
            mid_price, order.side, initial_offset
        )

        # Place initial limit
        try:
            order_id = await exchange.place_limit_order(
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=limit_price,
            )
        except (NotImplementedError, AttributeError):
            fill = await self._market_fallback(order, exchange)
            return [fill] if fill else []
        except Exception:
            logger.exception("Adaptive limit initial order failed")
            fill = await self._market_fallback(order, exchange)
            return [fill] if fill else []

        fill = await self._adjust_loop(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            mid_price=mid_price,
            quantity=order.quantity,
            exchange=exchange,
            config=cfg,
        )

        if fill:
            return [fill]

        # Max wait exceeded — cancel and market fallback
        try:
            exchange.cancel_order(str(order_id), order.symbol)
        except Exception:
            pass

        mkt_fill = await self._market_fallback(order, exchange)
        return [mkt_fill] if mkt_fill else []

    def _compute_limit_price(
        self, mid_price: float, side, offset_bps: float
    ) -> float:
        """Compute a limit price offset from mid.

        For buys, place below mid (more favourable).
        For sells, place above mid (more favourable).
        """
        from core.types import Side

        offset = mid_price * offset_bps / 10000.0
        if side == Side.BUY:
            return mid_price - offset
        else:
            return mid_price + offset

    async def _adjust_loop(
        self, order_id, symbol, side, mid_price, quantity, exchange, config
    ) -> dict:
        """Periodically walk the limit price toward mid."""
        from core.types import Side

        step_bps = float(config.get("step_bps", 1))
        interval = float(config.get("adjust_interval_s", 5))
        max_wait = float(config.get("max_wait_s", 60))
        current_offset = float(config.get("initial_offset_bps", 5))

        start = time.monotonic()

        while time.monotonic() - start < max_wait:
            # Check if filled
            try:
                status = await exchange.get_order_status(order_id)
                if status and status.get("status") == "filled":
                    return {
                        "price": status.get("fill_price", mid_price),
                        "quantity": quantity,
                        "timestamp": datetime.now(timezone.utc),
                        "fee": status.get("fee", 0.0),
                        "order_id": order_id,
                        "algo": "adaptive_limit",
                        "child_type": "limit",
                    }
            except (NotImplementedError, AttributeError):
                return None
            except Exception:
                return None

            await asyncio.sleep(interval)

            # Walk price closer
            current_offset = max(0, current_offset - step_bps)
            new_price = self._compute_limit_price(mid_price, side, current_offset)

            # Cancel old, place new
            try:
                exchange.cancel_order(str(order_id), symbol)
                order_id = await exchange.place_limit_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=new_price,
                )
            except Exception:
                logger.exception("Adaptive limit adjustment failed")
                return None

            if current_offset <= 0:
                # Already at mid — wait for fill or timeout
                break

        # Final check
        try:
            status = await exchange.get_order_status(order_id)
            if status and status.get("status") == "filled":
                return {
                    "price": status.get("fill_price", mid_price),
                    "quantity": quantity,
                    "timestamp": datetime.now(timezone.utc),
                    "fee": status.get("fee", 0.0),
                    "order_id": order_id,
                    "algo": "adaptive_limit",
                    "child_type": "limit",
                }
        except Exception:
            pass

        return None

    async def _market_fallback(self, order, exchange) -> dict:
        """Fall back to a market order."""
        from core.models import Order
        from core.types import OrderType

        mkt_order = Order(
            symbol=order.symbol,
            side=order.side,
            order_type=OrderType.MARKET,
            quantity=order.quantity,
        )
        try:
            fill = exchange.place_order(mkt_order)
            return {
                "price": fill.fill_price,
                "quantity": fill.quantity,
                "timestamp": fill.timestamp,
                "fee": fill.commission,
                "order_id": fill.order_id,
                "algo": "adaptive_limit",
                "child_type": "market_fallback",
            }
        except Exception:
            logger.exception("Adaptive limit market fallback failed")
            return None

    def get_metadata(self) -> dict:
        return {
            "name": "adaptive_limit",
            "description": (
                "Adaptive limit order: starts at favourable offset, "
                "walks price toward market in steps until filled or timeout"
            ),
            "default_config": DEFAULT_CONFIG.copy(),
        }
