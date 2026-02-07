"""Unified exchange wrapper for order execution via ccxt."""

import logging
import os
from datetime import datetime, timezone

import ccxt

from core.models import Fill, Order, Position
from core.types import OrderType, Side
from exchange.base import BaseExchange

logger = logging.getLogger(__name__)


class CcxtExchange(BaseExchange):
    """Thin wrapper around a ccxt exchange for placing and managing orders."""

    def __init__(
        self,
        exchange_id: str,
        sandbox: bool = True,
        api_key: str = "",
        api_secret: str = "",
    ):
        api_key = api_key or os.environ.get("EXCHANGE_API_KEY", "")
        api_secret = api_secret or os.environ.get("EXCHANGE_API_SECRET", "")

        exchange_class = getattr(ccxt, exchange_id)
        self.exchange: ccxt.Exchange = exchange_class(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
            }
        )

        if sandbox:
            self.exchange.set_sandbox_mode(True)

        logger.info(
            "CcxtExchange initialised: exchange=%s sandbox=%s",
            exchange_id,
            sandbox,
        )

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_balance(self, currency: str = "USDT") -> float:
        """Return the free (available) balance for *currency*.

        Args:
            currency: The asset to query, e.g. "USDT", "BTC".

        Returns:
            Available balance as a float.
        """
        try:
            balance = self.exchange.fetch_balance()
            free = balance.get("free", {}).get(currency, 0.0)
            logger.info("Balance for %s: %s", currency, free)
            return float(free)
        except ccxt.BaseError as exc:
            logger.error("Error fetching balance: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def place_order(self, order: Order) -> Fill:
        """Submit an order to the exchange and return the resulting Fill.

        Supports market and limit orders. For limit orders, *order.price*
        must be set.

        Args:
            order: An Order dataclass from core.models.

        Returns:
            A Fill dataclass with execution details.
        """
        side = order.side.value  # "buy" or "sell"
        order_type = order.order_type.value  # "market" or "limit"

        try:
            result = self.exchange.create_order(
                symbol=order.symbol,
                type=order_type,
                side=side,
                amount=order.quantity,
                price=order.price,  # None is fine for market orders
            )
        except ccxt.BaseError as exc:
            logger.error("Error placing %s %s order for %s: %s", side, order_type, order.symbol, exc)
            raise

        fill_price = result.get("average") or result.get("price") or 0.0
        fee_cost = 0.0
        if result.get("fee"):
            fee_cost = result["fee"].get("cost", 0.0) or 0.0

        filled_qty = result.get("filled", order.quantity)
        ts = result.get("datetime")
        fill_ts = (
            datetime.fromisoformat(ts) if ts else datetime.now(timezone.utc)
        )

        fill = Fill(
            order_id=result["id"],
            symbol=order.symbol,
            side=order.side,
            quantity=float(filled_qty),
            fill_price=float(fill_price),
            timestamp=fill_ts,
            commission=float(fee_cost),
        )

        logger.info(
            "Order filled: %s %s %.6f %s @ %.2f (fee=%.4f)",
            fill.side.value,
            fill.symbol,
            fill.quantity,
            order_type,
            fill.fill_price,
            fill.commission,
        )
        return fill

    def get_ticker(self, symbol: str) -> dict:
        """Fetch the current ticker for *symbol*.

        Returns:
            Dict with at least 'last', 'bid', and 'ask' keys.
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                "last": ticker.get("last"),
                "bid": ticker.get("bid"),
                "ask": ticker.get("ask"),
                "symbol": symbol,
            }
        except ccxt.BaseError as exc:
            logger.error("Error fetching ticker for %s: %s", symbol, exc)
            raise

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order.

        Args:
            order_id: The exchange order ID.
            symbol: The trading pair (required by most exchanges).

        Returns:
            True if the cancellation request succeeded.
        """
        try:
            self.exchange.cancel_order(order_id, symbol)
            logger.info("Cancelled order %s on %s", order_id, symbol)
            return True
        except ccxt.OrderNotFound:
            logger.warning("Order %s not found (may already be filled/cancelled)", order_id)
            return False
        except ccxt.BaseError as exc:
            logger.error("Error cancelling order %s: %s", order_id, exc)
            raise

    def get_open_orders(self, symbol: str) -> list:
        """Return a list of open orders for *symbol*.

        Each item is the raw ccxt order dict.
        """
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            logger.info("Found %d open orders for %s", len(orders), symbol)
            return orders
        except ccxt.BaseError as exc:
            logger.error("Error fetching open orders for %s: %s", symbol, exc)
            raise

    def get_positions(self) -> list[Position]:
        """Return open positions from the exchange.

        Not all CCXT exchanges support position fetching. Returns an empty
        list if the exchange does not support it.
        """
        if not self.exchange.has.get('fetchPositions', False):
            logger.debug("Exchange does not support fetchPositions")
            return []
        try:
            raw_positions = self.exchange.fetch_positions()
            positions = []
            for pos in raw_positions:
                contracts = float(pos.get('contracts', 0) or 0)
                if contracts == 0:
                    continue
                side = Side.BUY if pos.get('side') == 'long' else Side.SELL
                entry_price = float(pos.get('entryPrice', 0) or 0)
                ts = pos.get('datetime')
                entry_time = (
                    datetime.fromisoformat(ts) if ts
                    else datetime.now(timezone.utc)
                )
                positions.append(Position(
                    symbol=pos.get('symbol', ''),
                    side=side,
                    entry_price=entry_price,
                    quantity=contracts,
                    entry_time=entry_time,
                    unrealized_pnl=float(pos.get('unrealizedPnl', 0) or 0),
                ))
            return positions
        except ccxt.BaseError as exc:
            logger.error("Error fetching positions: %s", exc)
            return []

    def get_account_state(self) -> dict:
        """Return account state including equity and balances."""
        try:
            balance = self.exchange.fetch_balance()
            total = balance.get('total', {})
            free = balance.get('free', {})
            used = balance.get('used', {})
            return {
                'total': total,
                'free': free,
                'used': used,
                'timestamp': datetime.now(timezone.utc).isoformat(),
            }
        except ccxt.BaseError as exc:
            logger.error("Error fetching account state: %s", exc)
            raise
