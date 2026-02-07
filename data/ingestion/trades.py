"""Trade ingestor — streams trade tape via WebSocket with batched DB writes."""

import asyncio
import logging
import time
from datetime import datetime, timezone

from exchange.hyperliquid.normalizer import HyperliquidNormalizer
from monitoring.metrics import data_latency_seconds, trades_ingested_total

logger = logging.getLogger(__name__)


class TradeIngestor:
    """Subscribe to Hyperliquid trade feed and persist with batching.

    Trades are accumulated in a buffer and flushed to TimescaleDB either
    every ``flush_interval_s`` (default 1 s) or when the buffer reaches
    ``batch_size`` (default 100), whichever comes first.
    """

    def __init__(self, symbol, ws_client, redis_store, timescale_store,
                 quality_monitor, config):
        self._symbol = symbol
        self._coin = HyperliquidNormalizer.symbol_to_coin(symbol)
        self._ws = ws_client
        self._redis = redis_store
        self._ts = timescale_store
        self._qm = quality_monitor
        self._flush_interval_s = config.get(
            "ingestion.trades.flush_interval_s", 1.0
        )
        self._batch_size = config.get("ingestion.trades.batch_size", 100)
        self._sub_id: str | None = None
        self._running = False
        self._buffer: list[dict] = []
        self._flush_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._running = True
        self._sub_id = await self._ws.subscribe(
            "trades", {"coin": self._coin}, self._on_message,
        )
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info("TradeIngestor started for %s", self._symbol)

    async def stop(self) -> None:
        self._running = False
        if self._sub_id:
            await self._ws.unsubscribe(self._sub_id)
            self._sub_id = None
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        # Flush remaining buffer
        self._flush_buffer()
        logger.info("TradeIngestor stopped for %s", self._symbol)

    # ------------------------------------------------------------------
    # WS callback
    # ------------------------------------------------------------------

    def _on_message(self, msg: dict) -> None:
        receipt_ts = msg.get("_receipt_ts", time.time())
        seq_num = msg.get("_seq_num")

        raw_trades = msg.get("data", [])
        if not isinstance(raw_trades, list):
            raw_trades = [raw_trades]

        receipt_ts_dt = datetime.fromtimestamp(receipt_ts, tz=timezone.utc)

        for raw in raw_trades:
            trade = HyperliquidNormalizer.normalize_trade(raw)

            # DB record
            trade_ts = trade.get("timestamp", 0)
            if isinstance(trade_ts, (int, float)) and trade_ts > 0:
                trade_time = datetime.fromtimestamp(trade_ts / 1000, tz=timezone.utc)
            else:
                trade_time = receipt_ts_dt

            self._buffer.append({
                "time": trade_time,
                "symbol": trade["symbol"],
                "price": trade["price"],
                "size": trade["size"],
                "side": trade["side"],
                "trade_id": trade.get("trade_id", ""),
                "exchange_ts": trade_time,
                "receipt_ts": receipt_ts_dt,
                "seq_num": seq_num,
            })

            # Redis latest price
            self._redis.set_price(
                self._symbol,
                last=trade["price"],
                bid=trade["price"],
                ask=trade["price"],
                timestamp=receipt_ts_dt.isoformat(),
            )

            # Metrics
            trades_ingested_total.labels(symbol=self._symbol).inc()
            latency = receipt_ts_dt.timestamp() - trade_time.timestamp()
            data_latency_seconds.labels(
                source="trades", symbol=self._symbol,
            ).observe(abs(latency))

            # Quality
            if self._qm:
                self._qm.record_update(
                    "trades", self._symbol,
                    exchange_ts=trade_time,
                    receipt_ts=receipt_ts_dt,
                    seq_num=seq_num,
                    price=trade["price"],
                )

        # Flush if batch is full
        if len(self._buffer) >= self._batch_size:
            self._flush_buffer()

    # ------------------------------------------------------------------
    # Batching
    # ------------------------------------------------------------------

    async def _flush_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self._flush_interval_s)
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        if not self._buffer:
            return
        batch = self._buffer[:]
        self._buffer.clear()
        try:
            self._ts.insert_trades_batch(batch)
        except Exception:
            logger.exception("Error flushing trade batch for %s", self._symbol)
