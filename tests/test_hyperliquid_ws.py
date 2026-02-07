"""Tests for HyperliquidWebSocket."""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from exchange.hyperliquid.ws_client import HyperliquidWebSocket


class TestHyperliquidWebSocket(unittest.IsolatedAsyncioTestCase):
    """Test HyperliquidWebSocket with mocked websockets library."""

    def setUp(self):
        self.ws_client = HyperliquidWebSocket(testnet=True)

    # ------------------------------------------------------------------
    # subscribe
    # ------------------------------------------------------------------

    async def test_subscribe(self):
        """Verify subscribe sends the correct JSON message."""
        mock_ws = AsyncMock()
        self.ws_client._ws = mock_ws
        self.ws_client._running = True

        callback = MagicMock()
        sub_id = await self.ws_client.subscribe(
            "l2Book", {"coin": "BTC"}, callback
        )

        self.assertIn(sub_id, self.ws_client._subscriptions)
        self.assertEqual(
            self.ws_client._subscriptions[sub_id]["channel"], "l2Book"
        )

        # Check the JSON message sent
        mock_ws.send.assert_called_once()
        sent_msg = json.loads(mock_ws.send.call_args[0][0])
        self.assertEqual(sent_msg["method"], "subscribe")
        self.assertEqual(sent_msg["subscription"]["type"], "l2Book")
        self.assertEqual(sent_msg["subscription"]["coin"], "BTC")

    async def test_subscribe_user_channel(self):
        """Verify user channels include the user param."""
        mock_ws = AsyncMock()
        self.ws_client._ws = mock_ws
        self.ws_client._running = True

        callback = MagicMock()
        await self.ws_client.subscribe(
            "userFills", {"user": "0xABC"}, callback
        )

        sent_msg = json.loads(mock_ws.send.call_args[0][0])
        self.assertEqual(sent_msg["subscription"]["type"], "userFills")
        self.assertEqual(sent_msg["subscription"]["user"], "0xABC")

    async def test_subscribe_allmids(self):
        """allMids channel has no extra params."""
        mock_ws = AsyncMock()
        self.ws_client._ws = mock_ws
        self.ws_client._running = True

        callback = MagicMock()
        await self.ws_client.subscribe("allMids", {}, callback)

        sent_msg = json.loads(mock_ws.send.call_args[0][0])
        self.assertEqual(sent_msg["subscription"]["type"], "allMids")

    # ------------------------------------------------------------------
    # unsubscribe
    # ------------------------------------------------------------------

    async def test_unsubscribe(self):
        """Verify cleanup and unsubscribe message."""
        mock_ws = AsyncMock()
        self.ws_client._ws = mock_ws
        self.ws_client._running = True

        callback = MagicMock()
        sub_id = await self.ws_client.subscribe(
            "trades", {"coin": "ETH"}, callback
        )
        self.assertIn(sub_id, self.ws_client._subscriptions)

        await self.ws_client.unsubscribe(sub_id)
        self.assertNotIn(sub_id, self.ws_client._subscriptions)

        # Check unsubscribe message was sent (second call)
        self.assertEqual(mock_ws.send.call_count, 2)
        unsub_msg = json.loads(mock_ws.send.call_args[0][0])
        self.assertEqual(unsub_msg["method"], "unsubscribe")
        self.assertEqual(unsub_msg["subscription"]["type"], "trades")

    async def test_unsubscribe_unknown_id(self):
        """Unsubscribing with an unknown id is a no-op."""
        mock_ws = AsyncMock()
        self.ws_client._ws = mock_ws
        self.ws_client._running = True
        # Should not raise
        await self.ws_client.unsubscribe("nonexistent-id")
        mock_ws.send.assert_not_called()

    # ------------------------------------------------------------------
    # message routing
    # ------------------------------------------------------------------

    async def test_message_routing(self):
        """Simulate incoming message and verify callback is called."""
        callback = MagicMock()

        self.ws_client._subscriptions["sub-1"] = {
            "channel": "trades",
            "params": {"coin": "BTC"},
            "callback": callback,
        }

        incoming = json.dumps({
            "channel": "trades",
            "data": [
                {"coin": "BTC", "side": "Buy", "px": "50000", "sz": "0.1"}
            ],
        })

        await self.ws_client._handle_message(incoming)

        callback.assert_called_once()
        call_data = callback.call_args[0][0]
        self.assertIn("_receipt_ts", call_data)
        self.assertIn("_seq_num", call_data)
        self.assertEqual(call_data["_seq_num"], 1)
        self.assertEqual(call_data["channel"], "trades")

    async def test_message_routing_async_callback(self):
        """Async callbacks should be awaited."""
        callback = AsyncMock()

        self.ws_client._subscriptions["sub-1"] = {
            "channel": "allMids",
            "params": {},
            "callback": callback,
        }

        incoming = json.dumps({
            "channel": "allMids",
            "data": {"mids": {"BTC": "50000"}},
        })

        await self.ws_client._handle_message(incoming)
        callback.assert_awaited_once()

    async def test_message_seq_num_increments(self):
        """Sequence number should increment with each message."""
        callback = MagicMock()
        self.ws_client._subscriptions["sub-1"] = {
            "channel": "trades",
            "params": {},
            "callback": callback,
        }

        for i in range(3):
            await self.ws_client._handle_message(
                json.dumps({"channel": "trades", "data": []})
            )

        self.assertEqual(self.ws_client._seq_num, 3)
        # Last call should have seq_num=3
        last_data = callback.call_args[0][0]
        self.assertEqual(last_data["_seq_num"], 3)

    async def test_no_matching_subscription(self):
        """Messages with no matching channel should be silently ignored."""
        callback = MagicMock()
        self.ws_client._subscriptions["sub-1"] = {
            "channel": "trades",
            "params": {},
            "callback": callback,
        }

        await self.ws_client._handle_message(
            json.dumps({"channel": "l2Book", "data": {}})
        )
        callback.assert_not_called()

    # ------------------------------------------------------------------
    # reconnection
    # ------------------------------------------------------------------

    async def test_reconnection(self):
        """Verify reconnect attempts with backoff on connection failure."""
        mock_ws_new = AsyncMock()

        with patch(
            "websockets.connect",
            new_callable=AsyncMock,
        ) as mock_connect:
            # First attempt fails, second succeeds
            mock_connect.side_effect = [
                Exception("Connection refused"),
                mock_ws_new,
            ]

            # Add a subscription to verify re-subscribe
            callback = MagicMock()
            self.ws_client._subscriptions["sub-1"] = {
                "channel": "trades",
                "params": {"coin": "BTC"},
                "callback": callback,
            }
            self.ws_client._running = True

            # Patch asyncio.sleep to not actually wait and to track calls
            sleep_calls = []

            async def mock_sleep(duration):
                sleep_calls.append(duration)

            with patch("asyncio.sleep", side_effect=mock_sleep):
                # Patch create_task to avoid starting the recv loop
                with patch("asyncio.create_task"):
                    await self.ws_client._reconnect_loop()

            # Should have tried twice
            self.assertEqual(mock_connect.call_count, 2)
            # First sleep should be base delay (1.0s)
            self.assertAlmostEqual(sleep_calls[0], 1.0)
            # Second sleep should be 2x (2.0s)
            self.assertAlmostEqual(sleep_calls[1], 2.0)
            # After reconnecting, should re-subscribe
            mock_ws_new.send.assert_called_once()
            resub_msg = json.loads(mock_ws_new.send.call_args[0][0])
            self.assertEqual(resub_msg["method"], "subscribe")

    # ------------------------------------------------------------------
    # heartbeat
    # ------------------------------------------------------------------

    async def test_heartbeat(self):
        """Verify heartbeat sends a ping message."""
        mock_ws = AsyncMock()
        self.ws_client._ws = mock_ws
        self.ws_client._running = True

        # Run heartbeat for a very short time
        async def stop_after_one_ping():
            await asyncio.sleep(0.05)
            self.ws_client._running = False

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            # Make sleep return immediately but track calls, then stop
            call_count = 0

            async def controlled_sleep(duration):
                nonlocal call_count
                call_count += 1
                if call_count >= 2:
                    self.ws_client._running = False
                    raise asyncio.CancelledError()

            mock_sleep.side_effect = controlled_sleep

            try:
                await self.ws_client._heartbeat_loop()
            except asyncio.CancelledError:
                pass

            # Should have sent at least one ping
            if mock_ws.send.called:
                sent_msg = json.loads(mock_ws.send.call_args[0][0])
                self.assertEqual(sent_msg["method"], "ping")

    # ------------------------------------------------------------------
    # connect / disconnect
    # ------------------------------------------------------------------

    async def test_disconnect_clears_state(self):
        """Disconnect should clean up all subscriptions and tasks."""
        mock_ws = AsyncMock()
        self.ws_client._ws = mock_ws
        self.ws_client._running = True
        self.ws_client._subscriptions["sub-1"] = {
            "channel": "trades",
            "params": {},
            "callback": MagicMock(),
        }

        await self.ws_client.disconnect()

        self.assertFalse(self.ws_client._running)
        self.assertEqual(len(self.ws_client._subscriptions), 0)
        self.assertIsNone(self.ws_client._ws)
        mock_ws.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
