"""Base class for execution algorithms."""

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ExecutionAlgorithm(ABC):
    """Base class for all execution algorithms.

    Execution algorithms split a parent order into child orders
    to minimise market impact, improve fill quality, or hide
    order size from other participants.
    """

    @abstractmethod
    async def execute(self, order, exchange, config: dict) -> list:
        """Execute an order using this algorithm's logic.

        Parameters
        ----------
        order:
            The parent Order to execute.
        exchange:
            A BaseExchange instance for placing orders.
        config:
            Algorithm-specific configuration dict.

        Returns
        -------
        list
            List of Fill-like dicts: [{price, quantity, timestamp, fee}, ...]
        """

    @abstractmethod
    def get_metadata(self) -> dict:
        """Return algorithm metadata.

        Returns
        -------
        dict
            {name, description, default_config}
        """

    def estimate_impact(self, order_qty: float, orderbook: dict) -> float:
        """Estimate market impact in basis points.

        Walks through orderbook levels to estimate slippage for
        the given order quantity.

        Parameters
        ----------
        order_qty:
            Quantity to execute.
        orderbook:
            Dict with 'bids' and 'asks' keys, each a list of
            [price, size] pairs sorted best-first.

        Returns
        -------
        float
            Estimated slippage in basis points.
        """
        if not orderbook or order_qty <= 0:
            return 0.0

        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])

        if not bids or not asks:
            return 0.0

        mid_price = (bids[0][0] + asks[0][0]) / 2.0
        if mid_price <= 0:
            return 0.0

        # Walk through asks (buy side impact)
        remaining = order_qty
        total_cost = 0.0
        for price, size in asks:
            fill_qty = min(remaining, size)
            total_cost += fill_qty * price
            remaining -= fill_qty
            if remaining <= 0:
                break

        if remaining > 0:
            # Not enough liquidity — return high impact
            return 10000.0

        vwap = total_cost / order_qty
        impact_bps = abs(vwap - mid_price) / mid_price * 10000.0
        return impact_bps
