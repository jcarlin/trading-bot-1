"""Async WebSocket client for Hyperliquid real-time data feeds."""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Callable, Optional

from exchange.hyperliquid.types import MAINNET_WS_URL, TESTNET_WS_URL

logger = logging.getLogger(__name__)


class HyperliquidWebSocket:
    """Async WebSocket client for Hyperliquid streaming data.

    Supports the following channels: ``l2Book``, ``trades``, ``allMids``,
    ``candle``, ``userFills``, ``orderUpdates``, ``userFundings``.

    Parameters
    ----------
    testnet:
        If ``True`` (default) connect to the testnet WebSocket endpoint.
    """

    # Channels that require a ``coin`` parameter
    _COIN_CHANNELS = {"l2Book", "trades", "candle"}
    # Channels that require a ``user`` parameter
    _USER_CHANNELS = {"userFills", "orderUpdates", "userFundings"}

    def __init__(self, testnet: bool = True):
        self._ws_url = TESTNET_WS_URL if testnet else MAINNET_WS_URL
        self._testnet = testnet
        self._ws: Any = None  # websockets connection
        self._subscriptions: dict[str, dict] = {}
        self._seq_num: int = 0
        self._running = False
        self._recv_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        # Reconnection parameters
        self._reconnect_base = 1.0
        self._reconnect_max = 60.0
        self._max_reconnect_attempts = 10

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Establish WebSocket connection and start background loops."""
        import websockets

        logger.info("Connecting to Hyperliquid WS: %s", self._ws_url)
        self._ws = await websockets.connect(self._ws_url)
        self._running = True
        self._recv_task = asyncio.create_task(self._recv_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("WebSocket connected")

    async def disconnect(self) -> None:
        """Close the WebSocket connection and cancel background tasks."""
        self._running = False

        for task in (self._recv_task, self._heartbeat_task, self._reconnect_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        if self._ws is not None:
            await self._ws.close()
            self._ws = None

        self._subscriptions.clear()
        logger.info("WebSocket disconnected")

    async def subscribe(
        self,
        channel: str,
        params: dict,
        callback: Callable,
    ) -> str:
        """Subscribe to a Hyperliquid WebSocket channel.

        Parameters
        ----------
        channel:
            Channel name (e.g. ``"l2Book"``, ``"trades"``).
        params:
            Channel-specific parameters (e.g. ``{"coin": "BTC"}``).
        callback:
            Async or sync callable invoked with each message payload.

        Returns
        -------
        str
            A unique subscription id that can be used to unsubscribe.
        """
        sub_id = str(uuid.uuid4())
        self._subscriptions[sub_id] = {
            "channel": channel,
            "params": params,
            "callback": callback,
        }
        await self._send_subscribe(channel, params)
        logger.info("Subscribed to %s (sub_id=%s, params=%s)", channel, sub_id, params)
        return sub_id

    async def unsubscribe(self, subscription_id: str) -> None:
        """Remove a subscription by its id."""
        sub = self._subscriptions.pop(subscription_id, None)
        if sub is not None:
            await self._send_unsubscribe(sub["channel"], sub["params"])
            logger.info("Unsubscribed %s (channel=%s)", subscription_id, sub["channel"])

    # ------------------------------------------------------------------
    # Internal – message sending
    # ------------------------------------------------------------------

    async def _send_subscribe(self, channel: str, params: dict) -> None:
        """Send a subscribe message in HL format."""
        subscription: dict[str, Any] = {"type": channel}

        if channel in self._COIN_CHANNELS:
            subscription["coin"] = params.get("coin", "")
            if channel == "candle":
                subscription["interval"] = params.get("interval", "1m")
        elif channel in self._USER_CHANNELS:
            subscription["user"] = params.get("user", "")
        # allMids has no extra params

        msg = {"method": "subscribe", "subscription": subscription}
        await self._ws_send(msg)

    async def _send_unsubscribe(self, channel: str, params: dict) -> None:
        """Send an unsubscribe message."""
        subscription: dict[str, Any] = {"type": channel}

        if channel in self._COIN_CHANNELS:
            subscription["coin"] = params.get("coin", "")
            if channel == "candle":
                subscription["interval"] = params.get("interval", "1m")
        elif channel in self._USER_CHANNELS:
            subscription["user"] = params.get("user", "")

        msg = {"method": "unsubscribe", "subscription": subscription}
        await self._ws_send(msg)

    async def _ws_send(self, msg: dict) -> None:
        if self._ws is not None:
            await self._ws.send(json.dumps(msg))

    # ------------------------------------------------------------------
    # Internal – receive loop
    # ------------------------------------------------------------------

    async def _recv_loop(self) -> None:
        """Read messages from the WebSocket and route them to callbacks."""
        import websockets

        while self._running:
            try:
                raw_msg = await self._ws.recv()
                await self._handle_message(raw_msg)
            except websockets.ConnectionClosed:
                logger.warning("WebSocket connection closed")
                if self._running:
                    await self._reconnect_loop()
                break
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in WS recv loop")

    async def _handle_message(self, raw_msg: str) -> None:
        """Parse incoming JSON, inject metadata, and route to callback."""
        try:
            data = json.loads(raw_msg)
        except json.JSONDecodeError:
            logger.warning("Non-JSON WS message: %s", raw_msg[:200])
            return

        self._seq_num += 1
        receipt_ts = time.time()

        # Inject metadata
        data["_receipt_ts"] = receipt_ts
        data["_seq_num"] = self._seq_num

        channel = data.get("channel", "")

        # Route to matching subscriptions
        for sub in self._subscriptions.values():
            if sub["channel"] == channel:
                try:
                    result = sub["callback"](data)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    logger.exception(
                        "Error in callback for channel %s", channel
                    )

    # ------------------------------------------------------------------
    # Internal – heartbeat
    # ------------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        """Send a ping every 30 seconds to keep the connection alive."""
        while self._running:
            try:
                await asyncio.sleep(30)
                if self._ws is not None:
                    msg = {"method": "ping"}
                    await self._ws_send(msg)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in heartbeat loop")

    # ------------------------------------------------------------------
    # Internal – reconnection
    # ------------------------------------------------------------------

    async def _reconnect_loop(self) -> None:
        """Attempt to reconnect with exponential backoff."""
        import websockets

        delay = self._reconnect_base
        for attempt in range(1, self._max_reconnect_attempts + 1):
            if not self._running:
                return
            logger.info(
                "Reconnect attempt %d/%d in %.1fs",
                attempt,
                self._max_reconnect_attempts,
                delay,
            )
            await asyncio.sleep(delay)
            try:
                self._ws = await websockets.connect(self._ws_url)
                logger.info("Reconnected to WebSocket")
                # Re-subscribe all active subscriptions
                for sub in self._subscriptions.values():
                    await self._send_subscribe(sub["channel"], sub["params"])
                # Restart the recv loop
                self._recv_task = asyncio.create_task(self._recv_loop())
                return
            except Exception:
                logger.exception("Reconnect attempt %d failed", attempt)
                delay = min(delay * 2, self._reconnect_max)

        logger.error(
            "Failed to reconnect after %d attempts", self._max_reconnect_attempts
        )
        self._running = False
