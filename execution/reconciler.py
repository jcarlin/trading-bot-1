"""Position reconciler: detects drift between local and exchange state."""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Discrepancy:
    """A mismatch between local and exchange position state."""

    symbol: str
    field: str
    local_value: object
    exchange_value: object
    severity: str  # "warning" or "critical"


class PositionReconciler:
    """Compares local (Redis) position state against the exchange and reports diffs."""

    def __init__(self, exchange, redis_store, timescale_store, config):
        self.exchange = exchange
        self.redis = redis_store
        self.timescale = timescale_store
        self.auto_correct: bool = (
            config.get("reconciler.auto_correct", False)
            if hasattr(config, "get")
            else False
        )

    async def reconcile(self) -> list[Discrepancy]:
        """Run a full reconciliation cycle.

        Returns:
            List of Discrepancy objects for any detected mismatches.
        """
        discrepancies: list[Discrepancy] = []

        exchange_positions = self.exchange.get_positions()
        exchange_by_symbol: dict = {p.symbol: p for p in exchange_positions}

        # Collect all symbols we know about (local + exchange)
        local_symbols: set[str] = set()
        for pos in exchange_positions:
            local_symbols.add(pos.symbol)

        # Also check locally-tracked positions that may not be on the exchange
        # We rely on the exchange positions list for the authoritative symbol set,
        # but we'll also check any Redis keys we can discover.

        for symbol in local_symbols | self._known_local_symbols():
            exchange_pos = exchange_by_symbol.get(symbol)
            local_data = self.redis.get_position(symbol)

            if exchange_pos is None and local_data is None:
                continue

            if exchange_pos is not None and local_data is None:
                disc = Discrepancy(
                    symbol=symbol,
                    field="existence",
                    local_value=None,
                    exchange_value="open",
                    severity="critical",
                )
                discrepancies.append(disc)
                if self.auto_correct:
                    self._sync_from_exchange(symbol, exchange_pos)
                continue

            if exchange_pos is None and local_data is not None:
                disc = Discrepancy(
                    symbol=symbol,
                    field="existence",
                    local_value="open",
                    exchange_value=None,
                    severity="critical",
                )
                discrepancies.append(disc)
                if self.auto_correct:
                    self.redis._r.delete(f"position:{symbol}")
                continue

            # Both exist — compare fields
            checks = [
                ("side", local_data.get("side"), exchange_pos.side.value, "critical"),
                ("quantity", float(local_data.get("quantity", 0)), exchange_pos.quantity, "critical"),
                ("entry_price", float(local_data.get("entry_price", 0)), exchange_pos.entry_price, "warning"),
            ]

            for field, local_val, exchange_val, severity in checks:
                if field == "entry_price":
                    # Allow small floating-point tolerance for price
                    if abs(local_val - exchange_val) > 0.01 * exchange_val:
                        discrepancies.append(Discrepancy(symbol, field, local_val, exchange_val, severity))
                elif local_val != exchange_val:
                    discrepancies.append(Discrepancy(symbol, field, local_val, exchange_val, severity))

            if self.auto_correct and discrepancies:
                self._sync_from_exchange(symbol, exchange_pos)

        # Log discrepancies to system_events
        for disc in discrepancies:
            self._log_discrepancy(disc)

        if discrepancies:
            logger.warning("Reconciliation found %d discrepancies", len(discrepancies))
        else:
            logger.debug("Reconciliation passed — no discrepancies")

        return discrepancies

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _known_local_symbols(self) -> set[str]:
        """Best-effort discovery of locally-tracked position symbols."""
        try:
            keys = self.redis._r.keys("position:*")
            return {k.split(":", 1)[1] for k in keys if ":" in k}
        except Exception:
            return set()

    def _sync_from_exchange(self, symbol: str, pos) -> None:
        """Overwrite the local Redis position with the exchange state."""
        self.redis.set_position(
            symbol=symbol,
            entry_price=pos.entry_price,
            quantity=pos.quantity,
            side=pos.side.value,
            entry_time=pos.entry_time.isoformat() if pos.entry_time else "",
            unrealized_pnl=pos.unrealized_pnl,
        )
        logger.info("Auto-corrected local position for %s", symbol)

    def _log_discrepancy(self, disc: Discrepancy) -> None:
        """Persist a discrepancy as a system event."""
        try:
            self.timescale.insert_system_event({
                "time": datetime.now(timezone.utc),
                "event_type": "reconciliation_discrepancy",
                "severity": disc.severity,
                "component": "reconciler",
                "message": (
                    f"{disc.symbol} {disc.field}: "
                    f"local={disc.local_value} exchange={disc.exchange_value}"
                ),
                "details": {
                    "symbol": disc.symbol,
                    "field": disc.field,
                    "local_value": str(disc.local_value),
                    "exchange_value": str(disc.exchange_value),
                },
            })
        except Exception:
            logger.exception("Failed to log discrepancy to system_events")
