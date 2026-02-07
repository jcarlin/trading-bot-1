"""Tests for the data ingestion pipeline."""

import asyncio
import json
import time
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from data.ingestion.orderbook import OrderbookIngestor
from data.ingestion.trades import TradeIngestor
from data.ingestion.funding import FundingIngestor
from data.ingestion.candles import CandleIngestor
from data.ingestion.manager import IngestionManager


def _make_config(overrides=None):
    """Build a mock Config object."""
    data = {
        "ingestion": {
            "symbols": ["BTC/USDC"],
            "orderbook": {"snapshot_interval_s": 5.0},
            "trades": {"flush_interval_s": 1.0, "batch_size": 100},
            "funding": {"poll_interval_s": 60.0},
            "candles": {
                "timeframes": ["1m"],
                "backfill_on_start": False,
                "backfill_hours": 24,
            },
        },
        "exchange": {"account_address": "0xTestAddr"},
    }
    if overrides:
        data.update(overrides)

    config = MagicMock()
    config.get = lambda key, default=None: _dot_get(data, key, default)
    return config


def _dot_get(data, key, default=None):
    keys = key.split(".")
    value = data
    for k in keys:
        if isinstance(value, dict):
            value = value.get(k)
        else:
            return default
        if value is None:
            return default
    return value


class TestOrderbookIngestor(unittest.IsolatedAsyncioTestCase):
    """Test OrderbookIngestor WS callback and DB snapshot."""

    def setUp(self):
        self.ws = AsyncMock()
        self.ws.subscribe = AsyncMock(return_value="sub-ob-1")
        self.ws.unsubscribe = AsyncMock()
        self.redis = MagicMock()
        self.ts = MagicMock()
        self.qm = MagicMock()
        self.config = _make_config()

        self.ingestor = OrderbookIngestor(
            "BTC/USDC", self.ws, self.redis, self.ts, self.qm, self.config,
        )

    async def test_ws_callback_triggers_redis_write(self):
        """Simulate an L2 book update and verify Redis is updated."""
        msg = {
            "_receipt_ts": time.time(),
            "_seq_num": 1,
            "channel": "l2Book",
            "data": {
                "coin": "BTC",
                "levels": [
                    [{"px": "50000", "sz": "1.5", "n": 3}],
                    [{"px": "50001", "sz": "0.8", "n": 2}],
                ],
            },
        }
        self.ingestor._on_message(msg)
        self.redis.set_orderbook.assert_called_once()
        call_args = self.redis.set_orderbook.call_args
        self.assertEqual(call_args[0][0], "BTC/USDC")  # symbol
        # bids
        self.assertEqual(call_args[0][1], [[50000.0, 1.5]])
        # asks
        self.assertEqual(call_args[0][2], [[50001.0, 0.8]])

    async def test_snapshot_persists_to_timescale(self):
        """Simulate a book update then trigger the snapshot loop."""
        msg = {
            "_receipt_ts": time.time(),
            "_seq_num": 1,
            "data": {
                "coin": "BTC",
                "levels": [
                    [{"px": "50000", "sz": "1.0", "n": 1}],
                    [{"px": "50001", "sz": "0.5", "n": 1}],
                ],
            },
        }
        self.ingestor._on_message(msg)

        # Override snapshot interval for fast test
        self.ingestor._snapshot_interval_s = 0.01
        self.ingestor._running = True
        task = asyncio.create_task(self.ingestor._snapshot_loop())
        await asyncio.sleep(0.05)
        self.ingestor._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        self.ts.insert_orderbook_snapshot.assert_called()
        snapshot = self.ts.insert_orderbook_snapshot.call_args[0][0]
        self.assertEqual(snapshot["symbol"], "BTC/USDC")
        self.assertIn("mid_price", snapshot)

    async def test_start_stop(self):
        """Start subscribes, stop unsubscribes."""
        with patch("asyncio.create_task") as mock_ct:
            mock_ct.return_value = MagicMock(done=MagicMock(return_value=True))
            await self.ingestor.start()
            self.ws.subscribe.assert_called_once()
            await self.ingestor.stop()
            self.ws.unsubscribe.assert_called_once_with("sub-ob-1")


class TestTradeIngestor(unittest.IsolatedAsyncioTestCase):
    """Test TradeIngestor WS callback, batching, and Redis update."""

    def setUp(self):
        self.ws = AsyncMock()
        self.ws.subscribe = AsyncMock(return_value="sub-tr-1")
        self.ws.unsubscribe = AsyncMock()
        self.redis = MagicMock()
        self.ts = MagicMock()
        self.qm = MagicMock()
        self.config = _make_config()

        self.ingestor = TradeIngestor(
            "BTC/USDC", self.ws, self.redis, self.ts, self.qm, self.config,
        )

    async def test_ws_callback_buffers_trade(self):
        """A trade message should be buffered and Redis price set."""
        msg = {
            "_receipt_ts": time.time(),
            "_seq_num": 1,
            "channel": "trades",
            "data": [
                {
                    "coin": "BTC",
                    "side": "Buy",
                    "px": "50100",
                    "sz": "0.1",
                    "time": int(time.time() * 1000),
                    "hash": "0xabc",
                },
            ],
        }
        self.ingestor._on_message(msg)
        self.assertEqual(len(self.ingestor._buffer), 1)
        self.redis.set_price.assert_called_once()

    async def test_batch_flush_writes_to_db(self):
        """Flushing the buffer should call insert_trades_batch."""
        # Add a trade to the buffer
        self.ingestor._buffer.append({
            "time": datetime.now(timezone.utc),
            "symbol": "BTC/USDC",
            "price": 50000.0,
            "size": 0.1,
            "side": "buy",
            "trade_id": "0x123",
            "exchange_ts": datetime.now(timezone.utc),
            "receipt_ts": datetime.now(timezone.utc),
            "seq_num": 1,
        })
        self.ingestor._flush_buffer()
        self.ts.insert_trades_batch.assert_called_once()
        batch = self.ts.insert_trades_batch.call_args[0][0]
        self.assertEqual(len(batch), 1)
        self.assertEqual(batch[0]["symbol"], "BTC/USDC")

    async def test_auto_flush_on_batch_size(self):
        """When buffer hits batch_size, it should auto-flush."""
        self.ingestor._batch_size = 2
        trade_msg = {
            "_receipt_ts": time.time(),
            "_seq_num": 1,
            "data": [
                {"coin": "BTC", "side": "Buy", "px": "50000",
                 "sz": "0.1", "time": int(time.time() * 1000), "hash": "0x1"},
                {"coin": "BTC", "side": "Sell", "px": "50001",
                 "sz": "0.2", "time": int(time.time() * 1000), "hash": "0x2"},
            ],
        }
        self.ingestor._on_message(trade_msg)
        self.ts.insert_trades_batch.assert_called_once()
        # Buffer should be empty after flush
        self.assertEqual(len(self.ingestor._buffer), 0)


class TestFundingIngestor(unittest.IsolatedAsyncioTestCase):
    """Test FundingIngestor REST poll."""

    def setUp(self):
        self.ws = AsyncMock()
        self.ws.subscribe = AsyncMock(return_value="sub-fund-1")
        self.ws.unsubscribe = AsyncMock()
        self.rest = MagicMock()
        self.redis = MagicMock()
        self.ts = MagicMock()
        self.qm = MagicMock()
        self.config = _make_config()

        self.ingestor = FundingIngestor(
            "BTC/USDC", self.ws, self.rest, self.redis, self.ts,
            self.qm, self.config,
        )

    async def test_rest_poll_stores_funding(self):
        """Simulate a successful REST poll and verify Redis + DB writes."""
        self.rest.get_meta.return_value = {"universe": []}
        self.rest._info = MagicMock()
        self.rest._info.meta_and_asset_ctxs.return_value = [
            {"universe": [{"name": "BTC"}, {"name": "ETH"}]},
            [
                {
                    "funding": "0.0001",
                    "openInterest": "1234.5",
                    "premium": "0.00005",
                    "oraclePx": "50050",
                    "markPx": "50045",
                    "dayNtlVlm": "999999",
                    "prevDayPx": "49900",
                },
                {
                    "funding": "0.0002",
                    "openInterest": "5678.0",
                    "premium": "0.0001",
                    "oraclePx": "3000",
                    "markPx": "3001",
                    "dayNtlVlm": "555555",
                    "prevDayPx": "2950",
                },
            ],
        ]

        await self.ingestor._fetch_and_store()

        self.redis.set_funding.assert_called_once()
        call_kw = self.redis.set_funding.call_args
        self.assertEqual(call_kw[1]["rate"], 0.0001)

        self.ts.insert_funding_rate.assert_called_once()
        rate_record = self.ts.insert_funding_rate.call_args[0][0]
        self.assertEqual(rate_record["symbol"], "BTC/USDC")
        self.assertAlmostEqual(rate_record["rate"], 0.0001)


class TestCandleIngestor(unittest.IsolatedAsyncioTestCase):
    """Test CandleIngestor WS callback."""

    def setUp(self):
        self.ws = AsyncMock()
        self.ws.subscribe = AsyncMock(return_value="sub-candle-1")
        self.ws.unsubscribe = AsyncMock()
        self.rest = MagicMock()
        self.ts = MagicMock()
        self.qm = MagicMock()
        self.config = _make_config()

        self.ingestor = CandleIngestor(
            "BTC/USDC", self.ws, self.rest, self.ts, self.qm, self.config,
        )

    async def test_ws_callback_persists_candle(self):
        """A candle WS message should be written to TimescaleDB."""
        msg = {
            "_receipt_ts": time.time(),
            "_seq_num": 1,
            "channel": "candle",
            "data": {
                "t": int(time.time() * 1000),
                "T": int(time.time() * 1000) + 60000,
                "s": "BTC",
                "i": "1m",
                "o": "50000",
                "c": "50100",
                "h": "50150",
                "l": "49950",
                "v": "123.45",
            },
        }
        self.ingestor._on_message(msg)
        self.ts.insert_candle.assert_called_once()
        candle = self.ts.insert_candle.call_args[0][0]
        self.assertEqual(candle["symbol"], "BTC/USDC")
        self.assertEqual(candle["timeframe"], "1m")
        self.assertAlmostEqual(candle["open"], 50000.0)

    async def test_start_subscribes_all_timeframes(self):
        """Start should subscribe to each configured timeframe."""
        await self.ingestor.start()
        # Config has ["1m"] so should subscribe once
        self.assertEqual(self.ws.subscribe.call_count, 1)
        call_args = self.ws.subscribe.call_args
        self.assertEqual(call_args[0][0], "candle")

    async def test_stop_unsubscribes(self):
        """Stop should unsubscribe all subscriptions."""
        await self.ingestor.start()
        await self.ingestor.stop()
        self.ws.unsubscribe.assert_called_once_with("sub-candle-1")


class TestIngestionManager(unittest.IsolatedAsyncioTestCase):
    """Test IngestionManager lifecycle and status."""

    def setUp(self):
        self.ws = AsyncMock()
        self.ws.subscribe = AsyncMock(return_value="sub-mgr-1")
        self.ws.unsubscribe = AsyncMock()
        self.rest = MagicMock()
        self.redis = MagicMock()
        self.ts = MagicMock()
        self.qm = MagicMock()
        self.config = _make_config()

    async def test_start_creates_ingestors(self):
        """Start should create ingestors for each configured symbol."""
        with patch("asyncio.create_task") as mock_ct:
            mock_ct.return_value = MagicMock(done=MagicMock(return_value=True))
            mgr = IngestionManager(
                self.config, self.ws, self.rest, self.ts,
                self.redis, self.qm,
            )
            await mgr.start()

            self.assertTrue(mgr._running)
            self.assertIn("BTC/USDC", mgr._ingestors)
            # 4 ingestor types per symbol
            self.assertEqual(len(mgr._ingestors["BTC/USDC"]), 4)

    async def test_stop_clears_ingestors(self):
        """Stop should clear all ingestors."""
        with patch("asyncio.create_task") as mock_ct:
            mock_ct.return_value = MagicMock(done=MagicMock(return_value=True))
            mgr = IngestionManager(
                self.config, self.ws, self.rest, self.ts,
                self.redis, self.qm,
            )
            await mgr.start()
            await mgr.stop()

            self.assertFalse(mgr._running)
            self.assertEqual(len(mgr._ingestors), 0)

    async def test_get_status(self):
        """get_status should report on all ingestors."""
        with patch("asyncio.create_task") as mock_ct:
            mock_ct.return_value = MagicMock(done=MagicMock(return_value=True))
            mgr = IngestionManager(
                self.config, self.ws, self.rest, self.ts,
                self.redis, self.qm,
            )
            await mgr.start()
            status = mgr.get_status()

            self.assertTrue(status["running"])
            self.assertIn("BTC/USDC", status["ingestors"])
            ingestor_list = status["ingestors"]["BTC/USDC"]
            types = {i["type"] for i in ingestor_list}
            self.assertIn("OrderbookIngestor", types)
            self.assertIn("TradeIngestor", types)
            self.assertIn("FundingIngestor", types)
            self.assertIn("CandleIngestor", types)


if __name__ == "__main__":
    unittest.main()
