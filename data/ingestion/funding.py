"""Funding rate ingestor — polls REST and streams WS funding events."""

import asyncio
import logging
from datetime import datetime, timezone

from exchange.hyperliquid.normalizer import HyperliquidNormalizer
from monitoring.metrics import data_latency_seconds

logger = logging.getLogger(__name__)


class FundingIngestor:
    """Ingest funding rate data from Hyperliquid.

    Uses two sources:
    - **REST poll**: periodically fetches current funding rates for all assets
      via ``rest_client.get_meta()`` + asset context data.
    - **WS**: subscribes to ``userFundings`` for real-time funding payment
      notifications (requires an authenticated address).

    Rates are stored in both Redis (hot) and TimescaleDB (persistent).
    """

    def __init__(self, symbol, ws_client, rest_client, redis_store,
                 timescale_store, quality_monitor, config):
        self._symbol = symbol
        self._coin = HyperliquidNormalizer.symbol_to_coin(symbol)
        self._ws = ws_client
        self._rest = rest_client
        self._redis = redis_store
        self._ts = timescale_store
        self._qm = quality_monitor
        self._poll_interval_s = config.get(
            "ingestion.funding.poll_interval_s", 60.0
        )
        self._user_address = config.get("exchange.account_address", "")
        self._running = False
        self._poll_task: asyncio.Task | None = None
        self._sub_id: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        # Subscribe to user funding payments if address is available
        if self._user_address:
            self._sub_id = await self._ws.subscribe(
                "userFundings",
                {"user": self._user_address},
                self._on_funding_event,
            )
        logger.info("FundingIngestor started for %s", self._symbol)

    async def stop(self) -> None:
        self._running = False
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        if self._sub_id:
            await self._ws.unsubscribe(self._sub_id)
            self._sub_id = None
        logger.info("FundingIngestor stopped for %s", self._symbol)

    # ------------------------------------------------------------------
    # REST polling
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                await self._fetch_and_store()
            except Exception:
                logger.exception("Funding poll error for %s", self._symbol)
            await asyncio.sleep(self._poll_interval_s)

    async def _fetch_and_store(self) -> None:
        """Fetch current funding from REST and persist."""
        meta = self._rest.get_meta()
        if not meta:
            return

        # meta() returns {"universe": [...], ...}
        # We need metaAndAssetCtxs which returns [meta, [ctx_per_asset...]]
        # But the client exposes get_meta() only. We'll use info directly.
        try:
            meta_and_ctxs = self._rest._info.meta_and_asset_ctxs()
        except Exception:
            logger.debug("meta_and_asset_ctxs not available, skipping")
            return

        if not isinstance(meta_and_ctxs, list) or len(meta_and_ctxs) < 2:
            return

        universe = meta_and_ctxs[0].get("universe", [])
        asset_ctxs = meta_and_ctxs[1]

        # Find the index for our coin
        coin_idx = None
        for i, asset in enumerate(universe):
            if asset.get("name") == self._coin:
                coin_idx = i
                break

        if coin_idx is None or coin_idx >= len(asset_ctxs):
            return

        ctx = asset_ctxs[coin_idx]
        funding = HyperliquidNormalizer.normalize_funding(ctx)
        now = datetime.now(timezone.utc)

        # Redis
        self._redis.set_funding(
            self._symbol,
            rate=funding["funding_rate"],
            premium=funding["premium"],
            mark_price=funding["mark_price"],
            timestamp=now.isoformat(),
        )

        # TimescaleDB
        self._ts.insert_funding_rate({
            "time": now,
            "symbol": self._symbol,
            "rate": funding["funding_rate"],
            "premium": funding["premium"],
            "mark_price": funding["mark_price"],
            "oracle_price": funding["oracle_price"],
            "open_interest": funding["open_interest"],
        })

        # Quality
        if self._qm:
            self._qm.record_update(
                "funding", self._symbol,
                exchange_ts=now,
                receipt_ts=now,
            )

        data_latency_seconds.labels(
            source="funding", symbol=self._symbol,
        ).observe(0)

    # ------------------------------------------------------------------
    # WS callback
    # ------------------------------------------------------------------

    def _on_funding_event(self, msg: dict) -> None:
        """Handle real-time funding payment notifications."""
        data = msg.get("data", {})
        if not data:
            return
        logger.info(
            "Funding event for %s: %s", self._symbol, data,
        )
