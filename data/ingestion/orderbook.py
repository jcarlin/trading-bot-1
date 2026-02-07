"""Order book ingestor — streams L2 data via WebSocket."""

import asyncio
import logging
import time
from datetime import datetime, timezone

from exchange.hyperliquid.normalizer import HyperliquidNormalizer
from monitoring.metrics import data_latency_seconds, orderbook_updates_total

logger = logging.getLogger(__name__)


class OrderbookIngestor:
    """Subscribe to Hyperliquid L2 book updates for a single symbol.

    Every update is normalised and written to Redis.  Snapshots are persisted
    to TimescaleDB at a configurable interval.
    """

    def __init__(self, symbol, ws_client, redis_store, timescale_store,
                 quality_monitor, config):
        self._symbol = symbol
        self._coin = HyperliquidNormalizer.symbol_to_coin(symbol)
        self._ws = ws_client
        self._redis = redis_store
        self._ts = timescale_store
        self._qm = quality_monitor
        self._snapshot_interval_s = config.get(
            "ingestion.orderbook.snapshot_interval_s", 5.0
        )
        self._sub_id: str | None = None
        self._running = False
        self._snapshot_task: asyncio.Task | None = None
        self._last_book: dict | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._running = True
        self._sub_id = await self._ws.subscribe(
            "l2Book", {"coin": self._coin}, self._on_message,
        )
        self._snapshot_task = asyncio.create_task(self._snapshot_loop())
        logger.info("OrderbookIngestor started for %s", self._symbol)

    async def stop(self) -> None:
        self._running = False
        if self._sub_id:
            await self._ws.unsubscribe(self._sub_id)
            self._sub_id = None
        if self._snapshot_task and not self._snapshot_task.done():
            self._snapshot_task.cancel()
            try:
                await self._snapshot_task
            except asyncio.CancelledError:
                pass
        logger.info("OrderbookIngestor stopped for %s", self._symbol)

    # ------------------------------------------------------------------
    # WS callback
    # ------------------------------------------------------------------

    def _on_message(self, msg: dict) -> None:
        receipt_ts = msg.get("_receipt_ts", time.time())
        seq_num = msg.get("_seq_num")

        raw_data = msg.get("data", {})
        book = HyperliquidNormalizer.normalize_orderbook(raw_data)

        bids = book.get("bids", [])
        asks = book.get("asks", [])
        mid_price = 0.0
        spread = 0.0
        if bids and asks:
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            mid_price = (best_bid + best_ask) / 2
            spread = best_ask - best_bid

        # Derived: depth at 5 levels
        bid_depth_5 = sum(b[1] for b in bids[:5])
        ask_depth_5 = sum(a[1] for a in asks[:5])

        self._last_book = {
            "symbol": self._symbol,
            "bids": bids,
            "asks": asks,
            "mid_price": mid_price,
            "spread": spread,
            "bid_depth_5": bid_depth_5,
            "ask_depth_5": ask_depth_5,
            "receipt_ts": receipt_ts,
            "seq_num": seq_num,
        }

        # Redis hot state
        now_iso = datetime.now(timezone.utc).isoformat()
        self._redis.set_orderbook(
            self._symbol, bids, asks, mid_price, spread, now_iso,
        )

        # Metrics
        orderbook_updates_total.labels(symbol=self._symbol).inc()
        exchange_ts_dt = datetime.now(timezone.utc)
        receipt_ts_dt = datetime.fromtimestamp(receipt_ts, tz=timezone.utc)
        latency = receipt_ts_dt.timestamp() - exchange_ts_dt.timestamp()
        data_latency_seconds.labels(source="orderbook", symbol=self._symbol).observe(
            abs(latency)
        )

        # Quality monitor
        if self._qm:
            self._qm.record_update(
                "orderbook", self._symbol,
                exchange_ts=exchange_ts_dt,
                receipt_ts=receipt_ts_dt,
                seq_num=seq_num,
                price=mid_price if mid_price > 0 else None,
            )

    # ------------------------------------------------------------------
    # Periodic DB snapshot
    # ------------------------------------------------------------------

    async def _snapshot_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self._snapshot_interval_s)
            if self._last_book is not None:
                try:
                    self._ts.insert_orderbook_snapshot({
                        "time": datetime.now(timezone.utc),
                        "symbol": self._symbol,
                        "bids": self._last_book["bids"],
                        "asks": self._last_book["asks"],
                        "spread": self._last_book["spread"],
                        "mid_price": self._last_book["mid_price"],
                    })
                except Exception:
                    logger.exception("Error persisting OB snapshot for %s",
                                     self._symbol)
