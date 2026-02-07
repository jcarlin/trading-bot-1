#!/usr/bin/env python3
"""Entry point for the strategy runner service.

Starts the Prometheus metrics server, initialises all components,
and runs the strategy tick loop alongside equity updates and reconciliation.
"""

import argparse
import asyncio
import importlib
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

logger = logging.getLogger(__name__)

RECONCILE_INTERVAL_S = 30
EQUITY_UPDATE_INTERVAL_S = 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the strategy runner service")
    parser.add_argument(
        "--config",
        default="config/phase1.yaml",
        help="Path to YAML config file (default: config/phase1.yaml)",
    )
    return parser.parse_args()


def snake_to_pascal(name: str) -> str:
    """Convert snake_case to PascalCase."""
    return "".join(part.title() for part in name.split("_"))


def load_strategy(strategy_name: str, params: dict):
    """Dynamically load a LiveStrategy by name.

    Follows the convention: strategy/{name}.py -> {Name}Strategy class.
    """
    module_name = f"strategy.{strategy_name}"
    class_name = f"{snake_to_pascal(strategy_name)}Strategy"

    try:
        module = importlib.import_module(module_name)
        strategy_cls = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(
            f"Could not load strategy '{strategy_name}': "
            f"expected {module_name}.{class_name}"
        ) from e

    return strategy_cls(params)


async def reconciliation_loop(engine, stop_event: asyncio.Event) -> None:
    """Periodically reconcile positions."""
    while not stop_event.is_set():
        try:
            await engine.reconcile_positions()
        except Exception:
            logger.exception("Position reconciliation failed")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=RECONCILE_INTERVAL_S)
        except asyncio.TimeoutError:
            pass


async def equity_update_loop(engine, stop_event: asyncio.Event) -> None:
    """Periodically update equity metrics."""
    while not stop_event.is_set():
        try:
            await engine.update_equity()
        except Exception:
            logger.exception("Equity update failed")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=EQUITY_UPDATE_INTERVAL_S)
        except asyncio.TimeoutError:
            pass


async def performance_update_loop(tracker, config, stop_event: asyncio.Event) -> None:
    """Periodically compute and publish strategy performance metrics."""
    from monitoring.metrics import (
        strategy_max_drawdown,
        strategy_pnl_total,
        strategy_profit_factor,
        strategy_sharpe_ratio,
        strategy_win_rate,
    )

    update_interval = int(config.get("performance.update_interval_s", 300))
    windows = config.get("performance.windows", [1, 4, 24, 168, 720])

    while not stop_event.is_set():
        try:
            all_metrics = tracker.compute_all_windows(windows)
            for window, metrics in all_metrics.items():
                strategy_sharpe_ratio.labels(
                    strategy_name=tracker.strategy_name,
                    window=str(window),
                ).set(metrics["sharpe_ratio"])
                strategy_max_drawdown.labels(
                    strategy_name=tracker.strategy_name,
                    window=str(window),
                ).set(metrics["max_drawdown"])
                strategy_win_rate.labels(
                    strategy_name=tracker.strategy_name,
                    window=str(window),
                ).set(metrics["win_rate"])
                strategy_profit_factor.labels(
                    strategy_name=tracker.strategy_name,
                    window=str(window),
                ).set(metrics["profit_factor"])

            # Total PnL from largest window
            if all_metrics:
                largest_window = max(all_metrics.keys())
                strategy_pnl_total.labels(
                    strategy_name=tracker.strategy_name,
                ).set(all_metrics[largest_window]["total_pnl"])

        except Exception:
            logger.exception("Performance update failed")

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=update_interval)
        except asyncio.TimeoutError:
            pass


async def run(config: Config) -> None:
    """Initialise all components and run the strategy service."""
    from storage.timescale import TimescaleStore
    from storage.redis_store import RedisStore
    from execution.engine import ExecutionEngine
    from execution.risk_controls import ExecutionRiskControls
    from risk.manager import RiskManager
    from strategy.runner import StrategyRunner
    from strategy.decision_logger import StrategyDecisionLogger
    from metrics.strategy_performance import StrategyPerformanceTracker

    # Storage
    ts_cfg = config.get_section("storage.timescaledb")
    timescale_store = TimescaleStore(ts_cfg)
    logger.info("TimescaleDB store initialised")

    redis_cfg = config.get_section("storage.redis")
    redis_store = RedisStore(redis_cfg)
    logger.info("Redis store initialised")

    # Exchange client
    exchange_cfg = config.get_section("exchange")
    try:
        from exchange.hyperliquid import HyperliquidClient
        exchange_client = HyperliquidClient(exchange_cfg)
    except ImportError:
        from exchange.hyperliquid.client import HyperliquidClient
        exchange_client = HyperliquidClient(exchange_cfg)
    logger.info("Exchange client initialised")

    # Risk controls
    risk_cfg = config.get_section("risk")
    risk_manager = RiskManager(risk_cfg)
    risk_controls = ExecutionRiskControls(config)
    logger.info("Risk controls initialised")

    # Execution engine (correct 6-arg constructor)
    engine = ExecutionEngine(
        exchange=exchange_client,
        risk_manager=risk_manager,
        risk_controls=risk_controls,
        timescale=timescale_store,
        redis=redis_store,
        config=config,
    )
    logger.info("Execution engine initialised")

    # Load strategy
    strategy_name = config.get("strategy_runner.strategy", "funding_rate_arb")
    strategy_params = config.get_section(f"strategies.{strategy_name}")
    strategy = load_strategy(strategy_name, strategy_params)
    logger.info("Strategy loaded: %s", strategy.get_metadata().get("name"))

    # Decision logger
    decision_logger = StrategyDecisionLogger(timescale_store, strategy_name)

    # Strategy runner
    runner = StrategyRunner(
        strategy=strategy,
        execution_engine=engine,
        timescale=timescale_store,
        redis=redis_store,
        config=config,
        decision_logger=decision_logger,
    )

    # Performance tracker
    tracker = StrategyPerformanceTracker(timescale_store, strategy_name)

    # Graceful shutdown
    stop_event = asyncio.Event()

    def _shutdown():
        logger.info("Shutdown signal received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown)

    # Start concurrent loops
    tasks = [
        asyncio.create_task(runner.run(stop_event)),
        asyncio.create_task(equity_update_loop(engine, stop_event)),
        asyncio.create_task(reconciliation_loop(engine, stop_event)),
        asyncio.create_task(performance_update_loop(tracker, config, stop_event)),
    ]

    logger.info("Strategy service running: %s on %s",
                 strategy_name, config.get("strategy_runner.symbol"))

    try:
        await stop_event.wait()
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        timescale_store.close()
        logger.info("Strategy service shut down cleanly")


def main() -> None:
    args = parse_args()

    config = Config.from_yaml(args.config)
    setup_logging(config)

    # Start Prometheus metrics server
    metrics_port = int(config.get("monitoring.strategy_metrics_port", 8004))
    start_http_server(metrics_port)
    logger.info("Prometheus metrics server started on port %d", metrics_port)

    asyncio.run(run(config))


if __name__ == "__main__":
    main()
