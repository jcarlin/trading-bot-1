#!/usr/bin/env python3
"""Entry point for the data ingestion service.

Starts the Prometheus metrics server and runs the ingestion manager,
which subscribes to exchange WebSocket feeds and persists market data.
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prometheus_client import start_http_server

from core.config import Config
from core.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the data ingestion service")
    parser.add_argument(
        "--config",
        default="config/phase0.yaml",
        help="Path to YAML config file (default: config/phase0.yaml)",
    )
    return parser.parse_args()


async def run(config: Config) -> None:
    """Initialise all components and run the ingestion loop."""
    from storage.timescale import TimescaleStore
    from exchange.hyperliquid import HyperliquidClient, HyperliquidWebSocket

    # Optional Redis import – gracefully skip if not configured
    redis_store = None
    try:
        from storage.redis_store import RedisStore
        redis_cfg = config.get_section("redis")
        if redis_cfg:
            redis_store = RedisStore(redis_cfg)
            logger.info("Redis store initialised")
    except Exception as exc:
        logger.warning("Redis store not available: %s", exc)

    # TimescaleDB
    ts_cfg = config.get_section("timescaledb")
    timescale_store = TimescaleStore(ts_cfg)
    logger.info("TimescaleDB store initialised")

    # Exchange clients
    exchange_cfg = config.get_section("exchange")
    hl_client = HyperliquidClient(exchange_cfg)
    hl_ws = HyperliquidWebSocket(exchange_cfg)
    logger.info("Hyperliquid clients initialised")

    # Ingestion manager
    from data.ingestion.manager import IngestionManager

    manager = IngestionManager(
        config=config,
        timescale_store=timescale_store,
        redis_store=redis_store,
        exchange_client=hl_client,
        ws_client=hl_ws,
    )

    # Handle graceful shutdown
    stop_event = asyncio.Event()

    def _shutdown():
        logger.info("Shutdown signal received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown)

    logger.info("Starting ingestion manager")
    try:
        await manager.start()
        await stop_event.wait()
    finally:
        logger.info("Stopping ingestion manager")
        await manager.stop()
        timescale_store.close()
        logger.info("Ingestion service shut down cleanly")


def main() -> None:
    args = parse_args()

    config = Config.from_yaml(args.config)
    setup_logging(config)

    # Start Prometheus metrics server
    metrics_port = int(config.get("metrics.ingestion_port", 8001))
    start_http_server(metrics_port)
    logger.info("Prometheus metrics server started on port %d", metrics_port)

    asyncio.run(run(config))


if __name__ == "__main__":
    main()
