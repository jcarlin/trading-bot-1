#!/usr/bin/env python3
"""Health check utility for the trading system.

One-shot mode (default):
    python check_health.py
    Runs all checks, prints results, exits with code 0 (healthy) or 1 (unhealthy).

Serve mode:
    python check_health.py --serve --port 8003
    Starts an HTTP server exposing /health (JSON) and Prometheus metrics.
"""

import argparse
import asyncio
import json
import logging
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prometheus_client import start_http_server as start_prom_server, Gauge

from core.config import Config
from core.logging_setup import setup_logging
from monitoring.health import HealthChecker

logger = logging.getLogger(__name__)

# Prometheus gauges exposed by the health monitor
health_overall = Gauge("health_overall_status", "Overall health (1=healthy, 0.5=degraded, 0=unhealthy)")
health_component = Gauge("health_component_status", "Per-component health (1=ok, 0=error)", ["component"])

STATUS_MAP = {"healthy": 1.0, "degraded": 0.5, "unhealthy": 0.0}
COMPONENT_OK = {"ok", "connected", "skipped"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Health check utility")
    parser.add_argument(
        "--config",
        default="config/phase0.yaml",
        help="Path to YAML config file (default: config/phase0.yaml)",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run as a persistent HTTP health-check server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8003,
        help="Port for the health-check HTTP server (default: 8003)",
    )
    return parser.parse_args()


def build_checker(config: Config) -> HealthChecker:
    """Construct a HealthChecker wired to real backends (best-effort)."""
    timescale_store = None
    redis_store = None
    exchange_client = None
    ws_client = None

    # TimescaleDB
    try:
        from storage.timescale import TimescaleStore
        ts_cfg = config.get_section("timescaledb")
        if ts_cfg:
            timescale_store = TimescaleStore(ts_cfg)
    except Exception as exc:
        logger.warning("Could not init TimescaleStore: %s", exc)

    # Redis
    try:
        from storage.redis_store import RedisStore
        redis_cfg = config.get_section("redis")
        if redis_cfg:
            redis_store = RedisStore(redis_cfg)
    except Exception as exc:
        logger.warning("Could not init RedisStore: %s", exc)

    # Exchange
    try:
        from exchange.hyperliquid import HyperliquidClient
        exchange_cfg = config.get_section("exchange")
        if exchange_cfg:
            exchange_client = HyperliquidClient(exchange_cfg)
    except Exception as exc:
        logger.warning("Could not init HyperliquidClient: %s", exc)

    # WebSocket
    try:
        from exchange.hyperliquid import HyperliquidWebSocket
        exchange_cfg = config.get_section("exchange")
        if exchange_cfg:
            ws_client = HyperliquidWebSocket(exchange_cfg)
    except Exception as exc:
        logger.warning("Could not init HyperliquidWebSocket: %s", exc)

    return HealthChecker(
        config=config,
        timescale_store=timescale_store,
        redis_store=redis_store,
        exchange_client=exchange_client,
        ws_client=ws_client,
    )


def update_prometheus_gauges(results: dict) -> None:
    """Push health check results into Prometheus gauges."""
    overall = results.get("overall", "unhealthy")
    health_overall.set(STATUS_MAP.get(overall, 0.0))

    for component, detail in results.items():
        if component == "overall":
            continue
        status = detail.get("status", "error") if isinstance(detail, dict) else "error"
        health_component.labels(component=component).set(1.0 if status in COMPONENT_OK else 0.0)


# ------------------------------------------------------------------
# One-shot mode
# ------------------------------------------------------------------

def run_oneshot(config: Config) -> int:
    """Run all checks, print results, return exit code."""
    checker = build_checker(config)
    results = asyncio.run(checker.check_all())
    update_prometheus_gauges(results)

    print(json.dumps(results, indent=2, default=str))

    return 0 if results.get("overall") == "healthy" else 1


# ------------------------------------------------------------------
# Serve mode
# ------------------------------------------------------------------

class _HealthHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for /health endpoint."""

    checker: HealthChecker = None  # set by serve()

    def do_GET(self):
        if self.path == "/health":
            results = asyncio.run(self.checker.check_all())
            update_prometheus_gauges(results)

            overall = results.get("overall", "unhealthy")
            status_code = 200 if overall in ("healthy", "degraded") else 503

            body = json.dumps(results, indent=2, default=str).encode()
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        logger.debug(format, *args)


def serve(config: Config, port: int) -> None:
    """Run a persistent HTTP server with /health and Prometheus metrics."""
    checker = build_checker(config)
    _HealthHandler.checker = checker

    # Start Prometheus metrics on the same port (separate internal server)
    # We use port+1 for Prometheus to avoid conflicts, or the same port
    # if the user wants a single port. Here we expose prom on port itself
    # since /metrics is handled by prom and /health by our handler.
    # Actually, start prom on a separate thread.
    prom_port = port  # prometheus_client will bind here
    start_prom_server(prom_port)
    logger.info("Prometheus metrics server started on port %d", prom_port)

    # Health HTTP on port+1 to avoid conflict with prom server on same port
    health_port = port + 1
    server = HTTPServer(("0.0.0.0", health_port), _HealthHandler)
    logger.info("Health check server listening on port %d", health_port)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down health check server")
        server.shutdown()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    config = Config.from_yaml(args.config)
    setup_logging(config)

    if args.serve:
        serve(config, args.port)
    else:
        exit_code = run_oneshot(config)
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
