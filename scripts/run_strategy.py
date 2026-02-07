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

# Phase 2 imports (optional — graceful if not available)
try:
    from analysis.market_regime import MarketRegimeClassifier
    from metrics.health_scorer import StrategyHealthScorer
    from metrics.signal_quality import SignalQualityAssessor
    from metrics.execution_quality import ExecutionQualityTracker
    from strategy.manager import StrategyManager
    from reporting.report_generator import ReportGenerator
    from orchestration.ooda import OODAOrchestrator
    from orchestration.scheduler import EvaluationScheduler
    PHASE2_AVAILABLE = True
except ImportError:
    PHASE2_AVAILABLE = False

# Phase 3 imports (optional — graceful if not available)
try:
    from metrics.correlation import CorrelationAnalyzer
    from metrics.portfolio_performance import PortfolioPerformanceTracker
    PHASE3_AVAILABLE = True
except ImportError:
    PHASE3_AVAILABLE = False

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

    # Determine run mode
    run_mode = config.get("strategy_runner.mode", "single")
    multi_strategies_cfg = config.get("strategy_runner.strategies", [])

    # Multi-strategy mode
    if run_mode == "multi" and multi_strategies_cfg:
        logger.info("Multi-strategy mode: loading %d strategies", len(multi_strategies_cfg))
        strategies = {}
        runners = {}
        trackers = {}
        strategy_names = []

        for strat_cfg in multi_strategies_cfg:
            sname = strat_cfg.get("name") if isinstance(strat_cfg, dict) else strat_cfg
            sparams = config.get_section(f"strategies.{sname}")
            strat = load_strategy(sname, sparams)
            slogger = StrategyDecisionLogger(timescale_store, sname)
            srunner = StrategyRunner(
                strategy=strat,
                execution_engine=engine,
                timescale=timescale_store,
                redis=redis_store,
                config=config,
                decision_logger=slogger,
            )
            stracker = StrategyPerformanceTracker(timescale_store, sname)
            strategies[sname] = strat
            runners[sname] = srunner
            trackers[sname] = stracker
            strategy_names.append(sname)
            logger.info("Loaded strategy: %s", sname)

        # Use first strategy as primary for single-strategy fallback references
        strategy_name = strategy_names[0]
        strategy = strategies[strategy_name]
        runner = runners[strategy_name]
        tracker = trackers[strategy_name]
        decision_logger = StrategyDecisionLogger(timescale_store, strategy_name)
    else:
        # Single strategy mode (backward compatible)
        strategy_name = config.get("strategy_runner.strategy", "funding_rate_arb")
        strategy_params = config.get_section(f"strategies.{strategy_name}")
        strategy = load_strategy(strategy_name, strategy_params)
        logger.info("Strategy loaded: %s", strategy.get_metadata().get("name"))
        decision_logger = StrategyDecisionLogger(timescale_store, strategy_name)
        runner = StrategyRunner(
            strategy=strategy,
            execution_engine=engine,
            timescale=timescale_store,
            redis=redis_store,
            config=config,
            decision_logger=decision_logger,
        )
        tracker = StrategyPerformanceTracker(timescale_store, strategy_name)
        strategies = {strategy_name: strategy}
        runners = {strategy_name: runner}
        trackers = {strategy_name: tracker}
        strategy_names = [strategy_name]

    # Phase 2: Self-Evaluation Loop (optional)
    scheduler = None
    if PHASE2_AVAILABLE and config.get("evaluation.enabled", False):
        logger.info("Phase 2 evaluation loop enabled")

        regime_classifier = MarketRegimeClassifier(
            config.get_section("regime_classification"))

        health_scorer = StrategyHealthScorer(
            tracker, config.get_section("health_scoring.weights"))

        signal_assessor = SignalQualityAssessor(
            timescale_store, strategy_name,
            config.get("strategy_runner.symbol", "BTC/USDC"))

        execution_tracker = ExecutionQualityTracker(
            timescale_store, strategy_name)

        strategy_manager_instance = StrategyManager(
            execution_engine=engine,
            timescale=timescale_store,
            redis_store=redis_store,
            config=config,
            decision_logger=decision_logger,
        )

        # Register all strategies with manager
        for sname in strategy_names:
            strategy_manager_instance._strategies[sname] = {
                "strategy": strategies[sname],
                "runner": runners[sname],
                "status": "active",
                "started_at": datetime.now(timezone.utc).isoformat(),
            }

        # Phase 3: Correlation and portfolio tracking
        correlation_analyzer = None
        portfolio_tracker_inst = None
        if PHASE3_AVAILABLE and len(strategy_names) > 1:
            correlation_analyzer = CorrelationAnalyzer(
                timescale_store, strategy_names)
            portfolio_tracker_inst = PortfolioPerformanceTracker(
                timescale_store, strategy_names)
            logger.info("Phase 3 correlation + portfolio tracking enabled")

        report_gen = ReportGenerator(
            tracker, health_scorer, timescale_store, strategy_name,
            correlation_analyzer=correlation_analyzer,
            portfolio_tracker=portfolio_tracker_inst,
        )

        # Backtest comparator (optional)
        backtest_comp = None
        if config.get("backtest_comparison.enabled", False):
            from analysis.backtest_comparator import BacktestLiveComparator
            backtest_comp = BacktestLiveComparator(
                strategy, timescale_store, redis_store,
                strategy_name,
                config.get("strategy_runner.symbol", "BTC/USDC"))

        orchestrator = OODAOrchestrator(
            strategy_manager=strategy_manager_instance,
            health_scorer=health_scorer,
            regime_classifier=regime_classifier,
            timescale=timescale_store,
            redis_store=redis_store,
            config=config.get_section("orchestrator"),
            backtest_comparator=backtest_comp,
            signal_assessor=signal_assessor,
            execution_tracker=execution_tracker,
            report_generator=report_gen,
            correlation_analyzer=correlation_analyzer,
            portfolio_tracker=portfolio_tracker_inst,
        )

        scheduler = EvaluationScheduler(
            orchestrator, config.get_section("evaluation"))
        logger.info("Phase 2 evaluation scheduler initialised")

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
        asyncio.create_task(equity_update_loop(engine, stop_event)),
        asyncio.create_task(reconciliation_loop(engine, stop_event)),
    ]

    # Launch all strategy runners
    for sname, srunner in runners.items():
        tasks.append(asyncio.create_task(srunner.run(stop_event)))

    # Launch performance update for primary tracker
    tasks.append(asyncio.create_task(
        performance_update_loop(tracker, config, stop_event)))

    # Phase 2: Add evaluation scheduler
    if scheduler is not None:
        tasks.append(asyncio.create_task(scheduler.run(stop_event)))
        logger.info("Phase 2 evaluation scheduler started")

    logger.info("Strategy service running: %s on %s (mode=%s, strategies=%d, phase2=%s)",
                 strategy_name, config.get("strategy_runner.symbol"),
                 run_mode, len(strategy_names), scheduler is not None)

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
