"""Candle ingestor — streams candle updates via WS with optional REST backfill."""

import asyncio
import logging
import time
from datetime import datetime, timezone

from exchange.hyperliquid.normalizer import HyperliquidNormalizer
from monitoring.metrics import data_latency_seconds

logger = logging.getLogger(__name__)

DEFAULT_TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h"]


class CandleIngestor:
    """Subscribe to Hyperliquid candle updates for a symbol across timeframes.

    Optionally backfills gaps from REST on startup when
    ``ingestion.candles.backfill_on_start`` is ``True``.
    """

    def __init__(self, symbol, ws_client, rest_client, timescale_store,
                 quality_monitor, config):
        self._symbol = symbol
        self._coin = HyperliquidNormalizer.symbol_to_coin(symbol)
        self._ws = ws_client
        self._rest = rest_client
        self._ts = timescale_store
        self._qm = quality_monitor
        self._timeframes: list[str] = config.get(
            "ingestion.candles.timeframes", DEFAULT_TIMEFRAMES
        )
        self._backfill_on_start: bool = config.get(
            "ingestion.candles.backfill_on_start", False
        )
        self._backfill_hours: int = config.get(
            "ingestion.candles.backfill_hours", 24
        )
        self._running = False
        self._sub_ids: list[str] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._running = True

        # Optional REST backfill
        if self._backfill_on_start:
            await self._backfill()

        # Subscribe to WS candle channels per timeframe
        for tf in self._timeframes:
            sub_id = await self._ws.subscribe(
                "candle",
                {"coin": self._coin, "interval": tf},
                self._on_message,
            )
            self._sub_ids.append(sub_id)

        logger.info(
            "CandleIngestor started for %s (timeframes=%s)",
            self._symbol, self._timeframes,
        )

    async def stop(self) -> None:
        self._running = False
        for sid in self._sub_ids:
            await self._ws.unsubscribe(sid)
        self._sub_ids.clear()
        logger.info("CandleIngestor stopped for %s", self._symbol)

    # ------------------------------------------------------------------
    # WS callback
    # ------------------------------------------------------------------

    def _on_message(self, msg: dict) -> None:
        receipt_ts = msg.get("_receipt_ts", time.time())
        seq_num = msg.get("_seq_num")

        raw_data = msg.get("data", {})
        candle = HyperliquidNormalizer.normalize_candle(raw_data)

        ts_ms = candle.get("timestamp", 0)
        if ts_ms > 0:
            candle_time = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        else:
            candle_time = datetime.now(timezone.utc)

        receipt_ts_dt = datetime.fromtimestamp(receipt_ts, tz=timezone.utc)

        # Persist
        try:
            self._ts.insert_candle({
                "time": candle_time,
                "symbol": candle["symbol"],
                "timeframe": candle["interval"],
                "open": candle["open"],
                "high": candle["high"],
                "low": candle["low"],
                "close": candle["close"],
                "volume": candle["volume"],
                "exchange_ts": candle_time,
                "receipt_ts": receipt_ts_dt,
                "seq_num": seq_num,
            })
        except Exception:
            logger.exception("Error persisting candle for %s", self._symbol)

        # Metrics
        latency = receipt_ts_dt.timestamp() - candle_time.timestamp()
        data_latency_seconds.labels(
            source="candle", symbol=self._symbol,
        ).observe(abs(latency))

        # Quality
        if self._qm:
            self._qm.record_update(
                "candle", self._symbol,
                exchange_ts=candle_time,
                receipt_ts=receipt_ts_dt,
                seq_num=seq_num,
            )

    # ------------------------------------------------------------------
    # Backfill
    # ------------------------------------------------------------------

    async def _backfill(self) -> None:
        """Fetch recent candle history from REST to fill gaps."""
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - self._backfill_hours * 3600 * 1000

        for tf in self._timeframes:
            try:
                candles = self._rest.get_candles(
                    self._symbol, tf, start_ms, now_ms
                )
                for c in candles:
                    ts_ms = c.get("timestamp", 0)
                    candle_time = datetime.fromtimestamp(
                        ts_ms / 1000, tz=timezone.utc
                    ) if ts_ms > 0 else datetime.now(timezone.utc)

                    self._ts.insert_candle({
                        "time": candle_time,
                        "symbol": c["symbol"],
                        "timeframe": c["interval"],
                        "open": c["open"],
                        "high": c["high"],
                        "low": c["low"],
                        "close": c["close"],
                        "volume": c["volume"],
                    })
                logger.info(
                    "Backfilled %d candles for %s/%s",
                    len(candles), self._symbol, tf,
                )
            except Exception:
                logger.exception(
                    "Error backfilling candles for %s/%s", self._symbol, tf
                )
