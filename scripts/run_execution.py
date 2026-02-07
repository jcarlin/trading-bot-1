#!/usr/bin/env python3
"""Entry point for the execution engine service.

Starts the Prometheus metrics server, initialises the execution engine,
and runs periodic reconciliation and equity-update loops.
"""

import argparse
import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prometheus_client import start_http_server

from core.config import Config
from core.logging_setup import setup_logging
from monitoring.metrics import portfolio_equity_usd, portfolio_drawdown_pct, portfolio_unrealized_pnl_usd

logger = logging.getLogger(__name__)

RECONCILE_INTERVAL_S = 30
EQUITY_UPDATE_INTERVAL_S = 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the execution engine service")
    parser.add_argument(
        "--config",
        default="config/phase0.yaml",
        help="Path to YAML config file (default: config/phase0.yaml)",
    )
    return parser.parse_args()


async def reconciliation_loop(exchange_client, timescale_store, stop_event: asyncio.Event) -> None:
    """Periodically reconcile local position state with the exchange."""
    while not stop_event.is_set():
        try:
            positions = exchange_client.get_positions()
            logger.debug("Reconciliation: %d open positions", len(positions))
        except Exception:
            logger.exception("Position reconciliation failed")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=RECONCILE_INTERVAL_S)
        except asyncio.TimeoutError:
            pass


async def equity_update_loop(exchange_client, timescale_store, stop_event: asyncio.Event) -> None:
    """Periodically fetch account state and update equity metrics."""
    peak_equity = 0.0

    while not stop_event.is_set():
        try:
            state = exchange_client.get_account_state()
            equity = float(state.get("equity", 0.0))
            unrealized_pnl = float(state.get("unrealized_pnl", 0.0))

            if equity > peak_equity:
                peak_equity = equity

            drawdown_pct = ((peak_equity - equity) / peak_equity * 100) if peak_equity > 0 else 0.0

            # Update Prometheus gauges
            portfolio_equity_usd.set(equity)
            portfolio_drawdown_pct.set(drawdown_pct)
            portfolio_unrealized_pnl_usd.set(unrealized_pnl)

            # Persist snapshot
            timescale_store.insert_equity_snapshot({
                "time": datetime.now(timezone.utc),
                "total_equity": equity,
                "cash": float(state.get("cash", 0.0)),
                "position_value": float(state.get("position_value", 0.0)),
                "unrealized_pnl": unrealized_pnl,
                "realized_pnl": float(state.get("realized_pnl", 0.0)),
                "peak_equity": peak_equity,
                "drawdown_pct": drawdown_pct,
            })

            logger.debug("Equity update: $%.2f  DD=%.2f%%", equity, drawdown_pct)
        except Exception:
            logger.exception("Equity update failed")

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=EQUITY_UPDATE_INTERVAL_S)
        except asyncio.TimeoutError:
            pass


async def run(config: Config) -> None:
    """Initialise all components and run the execution service."""
    from storage.timescale import TimescaleStore
    from exchange.hyperliquid import HyperliquidClient

    # TimescaleDB
    ts_cfg = config.get_section("timescaledb")
    timescale_store = TimescaleStore(ts_cfg)
    logger.info("TimescaleDB store initialised")

    # Exchange client
    exchange_cfg = config.get_section("exchange")
    exchange_client = HyperliquidClient(exchange_cfg)
    logger.info("Hyperliquid client initialised")

    # Execution engine (import when available)
    try:
        from execution.engine import ExecutionEngine
        engine = ExecutionEngine(
            config=config,
            exchange_client=exchange_client,
            timescale_store=timescale_store,
        )
        logger.info("Execution engine initialised")
    except ImportError:
        engine = None
        logger.warning("execution.engine not yet available — running reconciliation only")

    # Handle graceful shutdown
    stop_event = asyncio.Event()

    def _shutdown():
        logger.info("Shutdown signal received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown)

    # Start background loops
    tasks = [
        asyncio.create_task(reconciliation_loop(exchange_client, timescale_store, stop_event)),
        asyncio.create_task(equity_update_loop(exchange_client, timescale_store, stop_event)),
    ]

    logger.info("Execution service running")

    try:
        await stop_event.wait()
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        timescale_store.close()
        logger.info("Execution service shut down cleanly")


def main() -> None:
    args = parse_args()

    config = Config.from_yaml(args.config)
    setup_logging(config)

    # Start Prometheus metrics server
    metrics_port = int(config.get("metrics.execution_port", 8002))
    start_http_server(metrics_port)
    logger.info("Prometheus metrics server started on port %d", metrics_port)

    asyncio.run(run(config))


if __name__ == "__main__":
    main()
