"""Hyperliquid REST client implementing the BaseExchange interface.

Uses the ``hyperliquid-python-sdk`` (``hyperliquid`` package) under the hood
for Info queries and Exchange (order) actions.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from core.models import Fill, Order, Position
from core.types import OrderType, Side
from exchange.base import BaseExchange
from exchange.hyperliquid.auth import HyperliquidAuth
from exchange.hyperliquid.normalizer import HyperliquidNormalizer
from exchange.hyperliquid.types import (
    MAINNET_API_URL,
    TESTNET_API_URL,
)

logger = logging.getLogger(__name__)


class HyperliquidClient(BaseExchange):
    """Hyperliquid exchange adapter.

    Wraps the ``hyperliquid`` SDK's :class:`Info` and :class:`Exchange`
    clients behind the common :class:`BaseExchange` interface.

    Parameters
    ----------
    account_address:
        The Ethereum address of the trading account.
    private_key:
        Hex-encoded private key used to sign orders.
    testnet:
        If ``True`` (default) connect to the Hyperliquid testnet.
    """

    def __init__(
        self,
        account_address: str,
        private_key: str,
        testnet: bool = True,
    ):
        from hyperliquid.info import Info
        from hyperliquid.exchange import Exchange

        self._auth = HyperliquidAuth(private_key, account_address)
        self._testnet = testnet
        self._base_url = TESTNET_API_URL if testnet else MAINNET_API_URL
        self._normalizer = HyperliquidNormalizer()

        self._info = Info(base_url=self._base_url, skip_ws=True)
        self._exchange = Exchange(
            self._auth.wallet,
            base_url=self._base_url,
        )

        logger.info(
            "HyperliquidClient initialised: address=%s testnet=%s",
            self._auth.address,
            testnet,
        )

    # ------------------------------------------------------------------
    # BaseExchange – place_order
    # ------------------------------------------------------------------

    def place_order(self, order: Order) -> Fill:
        """Submit an order to Hyperliquid.

        For MARKET orders HL does not have a native market type; we use an
        IOC (Immediate-or-Cancel) limit at an aggressive price obtained from
        the current mid.  For LIMIT orders we use GTC (Good-Till-Cancelled).
        """
        coin = HyperliquidNormalizer.symbol_to_coin(order.symbol)
        is_buy = order.side == Side.BUY

        if order.order_type == OrderType.MARKET:
            # Use IOC at a very aggressive price to simulate market order
            limit_px = self._get_aggressive_price(coin, is_buy)
            order_type_spec = {"limit": {"tif": "Ioc"}}
        else:
            limit_px = order.price or 0.0
            order_type_spec = {"limit": {"tif": "Gtc"}}

        try:
            result = self._exchange.order(
                coin,
                is_buy,
                order.quantity,
                limit_px,
                order_type_spec,
            )
            logger.info(
                "Order submitted: coin=%s side=%s qty=%.6f px=%.2f type=%s",
                coin,
                order.side.value,
                order.quantity,
                limit_px,
                order.order_type.value,
            )
        except Exception as exc:
            logger.error("Error placing order on Hyperliquid: %s", exc)
            raise

        return self._normalizer.normalize_fill(result, order)

    # ------------------------------------------------------------------
    # BaseExchange – cancel_order
    # ------------------------------------------------------------------

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        coin = HyperliquidNormalizer.symbol_to_coin(symbol)
        try:
            result = self._exchange.cancel(coin, int(order_id))
            logger.info("Cancelled order %s on %s", order_id, symbol)
            return result.get("status", "") == "ok"
        except Exception as exc:
            logger.error("Error cancelling order %s: %s", order_id, exc)
            return False

    # ------------------------------------------------------------------
    # BaseExchange – get_open_orders
    # ------------------------------------------------------------------

    def get_open_orders(self, symbol: str) -> list[dict]:
        try:
            orders = self._info.open_orders(self._auth.address)
            coin = HyperliquidNormalizer.symbol_to_coin(symbol)
            return [o for o in orders if o.get("coin") == coin]
        except Exception as exc:
            logger.error("Error fetching open orders: %s", exc)
            return []

    # ------------------------------------------------------------------
    # BaseExchange – get_balance
    # ------------------------------------------------------------------

    def get_balance(self, currency: str = "USDT") -> float:
        """Return account value from HL margin summary.

        Hyperliquid settles in USDC so ``currency`` is informational only.
        """
        try:
            user_state = self._info.user_state(self._auth.address)
            margin = user_state.get("marginSummary", {})
            account_value = float(margin.get("accountValue", 0))
            logger.info("Balance (accountValue): %.2f", account_value)
            return account_value
        except Exception as exc:
            logger.error("Error fetching balance: %s", exc)
            raise

    # ------------------------------------------------------------------
    # BaseExchange – get_ticker
    # ------------------------------------------------------------------

    def get_ticker(self, symbol: str) -> dict:
        coin = HyperliquidNormalizer.symbol_to_coin(symbol)
        try:
            all_mids = self._info.all_mids()
            mid = float(all_mids.get(coin, 0))

            # Grab best bid/ask from L2 snapshot
            l2 = self._info.l2_snapshot(coin)
            bid = ask = mid
            levels = l2.get("levels", [[], []])
            if levels[0]:
                bid = float(levels[0][0].get("px", mid))
            if levels[1]:
                ask = float(levels[1][0].get("px", mid))

            return {
                "symbol": symbol,
                "last": mid,
                "bid": bid,
                "ask": ask,
                "mid": mid,
            }
        except Exception as exc:
            logger.error("Error fetching ticker for %s: %s", symbol, exc)
            raise

    # ------------------------------------------------------------------
    # BaseExchange – get_positions
    # ------------------------------------------------------------------

    def get_positions(self) -> list[Position]:
        try:
            user_state = self._info.user_state(self._auth.address)
            return self._normalizer.normalize_positions(user_state)
        except Exception as exc:
            logger.error("Error fetching positions: %s", exc)
            return []

    # ------------------------------------------------------------------
    # BaseExchange – get_account_state
    # ------------------------------------------------------------------

    def get_account_state(self) -> dict:
        try:
            user_state = self._info.user_state(self._auth.address)
            margin = user_state.get("marginSummary", {})
            return {
                "account_value": float(margin.get("accountValue", 0)),
                "total_margin_used": float(margin.get("totalMarginUsed", 0)),
                "total_ntl_pos": float(margin.get("totalNtlPos", 0)),
                "total_raw_usd": float(margin.get("totalRawUsd", 0)),
                "withdrawable": float(user_state.get("withdrawable", 0)),
                "positions": len(user_state.get("assetPositions", [])),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            logger.error("Error fetching account state: %s", exc)
            raise

    # ==================================================================
    # Hyperliquid-specific methods
    # ==================================================================

    def get_funding_history(
        self, symbol: str, start_time: int = 0
    ) -> list[dict]:
        """Retrieve funding rate history for a symbol."""
        coin = HyperliquidNormalizer.symbol_to_coin(symbol)
        try:
            history = self._info.funding_history(coin, start_time)
            return history
        except Exception as exc:
            logger.error("Error fetching funding history: %s", exc)
            return []

    def get_user_fills(self, start_time_ms: int = 0) -> list[dict]:
        """Retrieve the authenticated user's fills."""
        try:
            fills = self._info.user_fills(self._auth.address)
            if start_time_ms:
                fills = [f for f in fills if f.get("time", 0) >= start_time_ms]
            return fills
        except Exception as exc:
            logger.error("Error fetching user fills: %s", exc)
            return []

    def get_meta(self) -> dict:
        """Retrieve exchange metadata (asset universe, fees, etc.)."""
        try:
            return self._info.meta()
        except Exception as exc:
            logger.error("Error fetching meta: %s", exc)
            return {}

    def get_l2_snapshot(self, symbol: str) -> dict:
        """Retrieve an L2 order book snapshot."""
        coin = HyperliquidNormalizer.symbol_to_coin(symbol)
        try:
            raw = self._info.l2_snapshot(coin)
            return self._normalizer.normalize_orderbook(raw)
        except Exception as exc:
            logger.error("Error fetching L2 snapshot: %s", exc)
            return {}

    def get_candles(
        self,
        symbol: str,
        interval: str,
        start_time: int,
        end_time: int,
    ) -> list[dict]:
        """Retrieve historical candles.

        Parameters
        ----------
        symbol:
            Canonical symbol, e.g. ``"BTC/USDC"``.
        interval:
            Candle interval, e.g. ``"1m"``, ``"1h"``.
        start_time / end_time:
            Unix timestamps in milliseconds.
        """
        coin = HyperliquidNormalizer.symbol_to_coin(symbol)
        try:
            raw_candles = self._info.candles_snapshot(
                coin, interval, start_time, end_time
            )
            return [
                self._normalizer.normalize_candle(c) for c in raw_candles
            ]
        except Exception as exc:
            logger.error("Error fetching candles: %s", exc)
            return []

    def update_leverage(
        self, symbol: str, leverage: int, is_cross: bool = True
    ) -> None:
        """Update the leverage setting for a symbol."""
        coin = HyperliquidNormalizer.symbol_to_coin(symbol)
        try:
            self._exchange.update_leverage(leverage, coin, is_cross=is_cross)
            logger.info(
                "Updated leverage: coin=%s leverage=%d cross=%s",
                coin,
                leverage,
                is_cross,
            )
        except Exception as exc:
            logger.error("Error updating leverage: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_aggressive_price(self, coin: str, is_buy: bool) -> float:
        """Return a price far enough to guarantee a fill for IOC market-like
        orders.  Uses mid price with a 5 % buffer.
        """
        try:
            all_mids = self._info.all_mids()
            mid = float(all_mids.get(coin, 0))
        except Exception:
            mid = 0.0

        if mid == 0:
            raise ValueError(
                f"Cannot determine market price for {coin}; "
                "unable to submit market order."
            )

        buffer = 0.05
        return round(mid * (1 + buffer), 1) if is_buy else round(mid * (1 - buffer), 1)
