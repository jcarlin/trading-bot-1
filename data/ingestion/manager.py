"""Ingestion manager — orchestrates all data ingestors for configured symbols."""

import asyncio
import logging

from data.ingestion.orderbook import OrderbookIngestor
from data.ingestion.trades import TradeIngestor
from data.ingestion.funding import FundingIngestor
from data.ingestion.candles import CandleIngestor

logger = logging.getLogger(__name__)


class IngestionManager:
    """Top-level orchestrator that creates and manages per-symbol ingestors.

    Parameters
    ----------
    config:
        Application ``Config`` instance.  Expects:
        - ``ingestion.symbols``: list of symbols to ingest (e.g. ["BTC/USDC"])
    ws_client:
        A connected ``HyperliquidWebSocket`` instance.
    rest_client:
        A ``HyperliquidClient`` instance for REST calls.
    timescale_store:
        ``TimescaleStore`` for persistent writes.
    redis_store:
        ``RedisStore`` for hot-state writes.
    quality_monitor:
        ``DataQualityMonitor`` for health tracking.
    """

    def __init__(self, config, ws_client, rest_client, timescale_store,
                 redis_store, quality_monitor):
        self._config = config
        self._ws = ws_client
        self._rest = rest_client
        self._ts = timescale_store
        self._redis = redis_store
        self._qm = quality_monitor

        self._symbols: list[str] = config.get("ingestion.symbols", [])
        self._ingestors: dict[str, list] = {}  # symbol -> list of ingestors
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Create and start all ingestors for every configured symbol."""
        self._running = True

        for symbol in self._symbols:
            workers = self._create_ingestors(symbol)
            self._ingestors[symbol] = workers
            for w in workers:
                await w.start()

        logger.info(
            "IngestionManager started for %d symbols: %s",
            len(self._symbols), self._symbols,
        )

    async def stop(self) -> None:
        """Gracefully stop all ingestors."""
        self._running = False
        for symbol, workers in self._ingestors.items():
            for w in workers:
                await w.stop()
        self._ingestors.clear()
        logger.info("IngestionManager stopped")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return the status of every ingestor grouped by symbol."""
        status: dict[str, list[dict]] = {}
        for symbol, workers in self._ingestors.items():
            status[symbol] = [
                {
                    "type": type(w).__name__,
                    "running": w._running,
                }
                for w in workers
            ]
        return {
            "running": self._running,
            "symbols": self._symbols,
            "ingestors": status,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _create_ingestors(self, symbol: str) -> list:
        """Instantiate the four ingestor types for a symbol."""
        return [
            OrderbookIngestor(
                symbol, self._ws, self._redis, self._ts, self._qm,
                self._config,
            ),
            TradeIngestor(
                symbol, self._ws, self._redis, self._ts, self._qm,
                self._config,
            ),
            FundingIngestor(
                symbol, self._ws, self._rest, self._redis, self._ts,
                self._qm, self._config,
            ),
            CandleIngestor(
                symbol, self._ws, self._rest, self._ts, self._qm,
                self._config,
            ),
        ]
