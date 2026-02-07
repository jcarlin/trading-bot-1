"""Health checker for all trading system components."""

import asyncio
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 5.0


class HealthChecker:
    """Runs health checks against every system component and reports status."""

    def __init__(
        self,
        config,
        timescale_store=None,
        redis_store=None,
        exchange_client=None,
        ws_client=None,
    ):
        self.config = config
        self.timescale_store = timescale_store
        self.redis_store = redis_store
        self.exchange_client = exchange_client
        self.ws_client = ws_client
        self.timeout = config.get("health.timeout_s", DEFAULT_TIMEOUT_S) if config else DEFAULT_TIMEOUT_S

    async def check_all(self) -> dict[str, Any]:
        """Run all health checks and return an aggregate status dict."""
        results: dict[str, Any] = {}

        checks = [
            ("timescaledb", self._check_timescaledb),
            ("redis", self._check_redis),
            ("exchange_api", self._check_exchange_api),
            ("websocket", self._check_websocket),
            ("data_freshness", self._check_data_freshness),
        ]

        for name, check_fn in checks:
            try:
                results[name] = await asyncio.wait_for(
                    check_fn(), timeout=self.timeout
                )
            except asyncio.TimeoutError:
                results[name] = {"status": "error", "error": "timeout"}
            except Exception as exc:
                results[name] = {"status": "error", "error": str(exc)}

        results["overall"] = self._compute_overall(results)
        return results

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    async def _check_timescaledb(self) -> dict:
        if self.timescale_store is None:
            return {"status": "skipped", "reason": "not configured"}

        start = time.monotonic()
        try:
            # Use a lightweight query to verify connectivity
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, self.timescale_store._query, "SELECT 1"
            )
            latency_ms = (time.monotonic() - start) * 1000
            return {"status": "ok", "latency_ms": round(latency_ms, 2)}
        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000
            return {"status": "error", "latency_ms": round(latency_ms, 2), "error": str(exc)}

    async def _check_redis(self) -> dict:
        if self.redis_store is None:
            return {"status": "skipped", "reason": "not configured"}

        start = time.monotonic()
        try:
            loop = asyncio.get_running_loop()
            pong = await loop.run_in_executor(
                None, self.redis_store._redis.ping
            )
            latency_ms = (time.monotonic() - start) * 1000
            if pong:
                return {"status": "ok", "latency_ms": round(latency_ms, 2)}
            return {"status": "error", "latency_ms": round(latency_ms, 2), "error": "ping returned False"}
        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000
            return {"status": "error", "latency_ms": round(latency_ms, 2), "error": str(exc)}

    async def _check_exchange_api(self) -> dict:
        if self.exchange_client is None:
            return {"status": "skipped", "reason": "not configured"}

        start = time.monotonic()
        try:
            loop = asyncio.get_running_loop()
            # Try get_account_state as a lightweight check; fall back to get_ticker
            if hasattr(self.exchange_client, "get_account_state"):
                await loop.run_in_executor(None, self.exchange_client.get_account_state)
            elif hasattr(self.exchange_client, "get_ticker"):
                symbol = self.config.get("symbol", "BTC/USDT") if self.config else "BTC/USDT"
                await loop.run_in_executor(None, self.exchange_client.get_ticker, symbol)
            else:
                return {"status": "error", "error": "no callable check method on exchange client"}
            latency_ms = (time.monotonic() - start) * 1000
            return {"status": "ok", "latency_ms": round(latency_ms, 2)}
        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000
            return {"status": "error", "latency_ms": round(latency_ms, 2), "error": str(exc)}

    async def _check_websocket(self) -> dict:
        if self.ws_client is None:
            return {"status": "skipped", "reason": "not configured"}

        try:
            if hasattr(self.ws_client, "connected"):
                connected = self.ws_client.connected
            elif hasattr(self.ws_client, "is_connected"):
                connected = self.ws_client.is_connected
            else:
                return {"status": "error", "error": "cannot determine ws connection status"}

            return {"status": "connected" if connected else "disconnected"}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    async def _check_data_freshness(self) -> dict:
        if self.redis_store is None:
            return {"status": "skipped", "reason": "redis not configured"}

        try:
            loop = asyncio.get_running_loop()
            # Look for keys matching our data quality pattern
            keys = await loop.run_in_executor(
                None, self.redis_store._redis.keys, "data_quality:*"
            )

            stale_feeds: list[str] = []
            ok_feeds: list[str] = []

            for key in keys:
                val = await loop.run_in_executor(
                    None, self.redis_store._redis.get, key
                )
                feed_name = key.decode() if isinstance(key, bytes) else key
                val_str = val.decode() if isinstance(val, bytes) else str(val) if val else "0"
                if val_str == "1":
                    ok_feeds.append(feed_name)
                else:
                    stale_feeds.append(feed_name)

            status = "ok" if not stale_feeds else "degraded"
            return {
                "status": status,
                "ok_feeds": len(ok_feeds),
                "stale_feeds": stale_feeds,
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    # ------------------------------------------------------------------
    # Overall status
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_overall(results: dict) -> str:
        """Derive an aggregate status from individual check results.

        Critical components: timescaledb, exchange_api
        Non-critical: redis, websocket, data_freshness
        """
        critical = ["timescaledb", "exchange_api"]
        non_critical = ["redis", "websocket", "data_freshness"]

        critical_ok = all(
            results.get(c, {}).get("status") in ("ok", "connected", "skipped")
            for c in critical
        )
        non_critical_ok = all(
            results.get(c, {}).get("status") in ("ok", "connected", "skipped")
            for c in non_critical
        )

        if critical_ok and non_critical_ok:
            return "healthy"
        if critical_ok:
            return "degraded"
        return "unhealthy"
