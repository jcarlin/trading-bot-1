"""Emergency kill switch: flatten all positions and halt trading."""

import logging
from datetime import datetime, timezone

from core.models import Order
from core.types import OrderType, Side
from monitoring.metrics import circuit_breaker_status, kill_switch_activations_total

logger = logging.getLogger(__name__)


class KillSwitch:
    """Emergency mechanism to cancel orders, flatten positions, and halt the system."""

    def __init__(self, exchange, redis_store, timescale_store, config):
        self.exchange = exchange
        self.redis = redis_store
        self.timescale = timescale_store

        # Symbols to scan when cancelling orders / flattening
        symbols = (
            config.get("symbols", [])
            if hasattr(config, "get")
            else []
        )
        if isinstance(symbols, str):
            symbols = [symbols]
        # Also pick up the legacy single-symbol key
        single = (
            config.get("symbol")
            if hasattr(config, "get")
            else None
        )
        if single and single not in symbols:
            symbols.append(single)
        self.symbols: list[str] = symbols

    # ------------------------------------------------------------------
    # Activate
    # ------------------------------------------------------------------

    async def activate(self, reason: str) -> None:
        """Activate the kill switch: cancel orders, flatten positions, halt.

        Steps:
            1. Log activation to system_events.
            2. Cancel all open orders for each configured symbol.
            3. Flatten all open positions with market close orders.
            4. Set Redis circuit_breaker:state to active.
            5. Update Prometheus metrics.
        """
        now = datetime.now(timezone.utc)
        logger.critical("KILL SWITCH ACTIVATED: %s", reason)

        # 1. Log to system_events
        self.timescale.insert_system_event({
            "time": now,
            "event_type": "kill_switch_activated",
            "severity": "critical",
            "component": "kill_switch",
            "message": f"Kill switch activated: {reason}",
            "details": {"reason": reason},
        })

        # 2. Cancel all open orders
        for symbol in self.symbols:
            try:
                open_orders = self.exchange.get_open_orders(symbol)
                for oo in open_orders:
                    oid = oo.get("order_id") or oo.get("id", "")
                    try:
                        self.exchange.cancel_order(oid, symbol)
                        logger.info("Cancelled order %s on %s", oid, symbol)
                    except Exception:
                        logger.exception("Failed to cancel order %s on %s", oid, symbol)
            except Exception:
                logger.exception("Failed to fetch open orders for %s", symbol)

        # 3. Flatten all positions
        try:
            positions = self.exchange.get_positions()
            for pos in positions:
                close_side = Side.SELL if pos.side == Side.BUY else Side.BUY
                close_order = Order(
                    symbol=pos.symbol,
                    side=close_side,
                    order_type=OrderType.MARKET,
                    quantity=pos.quantity,
                    timestamp=now,
                )
                try:
                    self.exchange.place_order(close_order)
                    logger.info(
                        "Flattened %s position on %s (qty=%s)",
                        pos.side.value, pos.symbol, pos.quantity,
                    )
                except Exception:
                    logger.exception(
                        "Failed to flatten %s position on %s",
                        pos.side.value, pos.symbol,
                    )
        except Exception:
            logger.exception("Failed to fetch positions for flattening")

        # 4. Update Redis circuit breaker state
        self.redis.set_circuit_breaker(
            active=True,
            reason=reason,
            activated_at=now.isoformat(),
        )

        # 5. Update Prometheus metrics
        circuit_breaker_status.set(1)
        kill_switch_activations_total.inc()

    # ------------------------------------------------------------------
    # Deactivate
    # ------------------------------------------------------------------

    async def deactivate(self) -> None:
        """Deactivate the kill switch and allow trading to resume."""
        now = datetime.now(timezone.utc)
        logger.info("Kill switch deactivated")

        self.redis.set_circuit_breaker(
            active=False,
            reason="",
            activated_at=now.isoformat(),
        )
        circuit_breaker_status.set(0)

        self.timescale.insert_system_event({
            "time": now,
            "event_type": "kill_switch_deactivated",
            "severity": "info",
            "component": "kill_switch",
            "message": "Kill switch deactivated — trading may resume",
            "details": {},
        })

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def is_active(self) -> bool:
        """Check whether the kill switch is currently engaged."""
        state = self.redis.get_circuit_breaker()
        if state is None:
            return False
        return state.get("active", False)
