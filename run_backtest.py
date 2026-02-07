#!/usr/bin/env python3
"""CLI entry point for running a backtest.

Usage:
    python run_backtest.py [--config path/to/config.yaml] [--report]

Loads configuration, instantiates the data provider, strategy, risk manager,
and backtest engine, then runs the backtest and prints a performance summary.
"""

import argparse
import importlib
import logging
import os
import sys

import yaml
import pandas as pd

from backtest.engine import BacktestEngine
from data.provider import DataProvider
from metrics.performance import PerformanceReporter
from risk.manager import RiskManager

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Configuration helpers
# ------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load a YAML config file and apply environment-variable overrides."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # Allow environment variables to override sensitive / common settings.
    cfg["exchange"]["api_key"] = os.environ.get(
        "EXCHANGE_API_KEY", cfg["exchange"].get("api_key", "")
    )
    cfg["exchange"]["api_secret"] = os.environ.get(
        "EXCHANGE_API_SECRET", cfg["exchange"].get("api_secret", "")
    )

    if os.environ.get("BACKTEST_INITIAL_CAPITAL"):
        cfg["backtest"]["initial_capital"] = float(os.environ["BACKTEST_INITIAL_CAPITAL"])
    if os.environ.get("BACKTEST_START_DATE"):
        cfg["backtest"]["start_date"] = os.environ["BACKTEST_START_DATE"]
    if os.environ.get("BACKTEST_END_DATE"):
        cfg["backtest"]["end_date"] = os.environ["BACKTEST_END_DATE"]
    if os.environ.get("TRADING_SYMBOL"):
        cfg["symbol"] = os.environ["TRADING_SYMBOL"]
    if os.environ.get("TRADING_TIMEFRAME"):
        cfg["timeframe"] = os.environ["TRADING_TIMEFRAME"]

    return cfg


# ------------------------------------------------------------------
# Dynamic strategy loader
# ------------------------------------------------------------------

def load_strategy(strategy_name: str, params: dict):
    """Dynamically import and instantiate a strategy by name.

    Strategies live in the ``strategy`` package.  The module name is the
    snake_case strategy name (e.g. ``sma_crossover``), and the class name
    is the PascalCase equivalent with a ``Strategy`` suffix
    (e.g. ``SmaCrossoverStrategy``).
    """
    module_path = f"strategy.{strategy_name}"
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        logger.error("Strategy module '%s' not found: %s", module_path, exc)
        sys.exit(1)

    # Convert snake_case to PascalCase + "Strategy"
    class_name = "".join(part.capitalize() for part in strategy_name.split("_")) + "Strategy"
    strategy_class = getattr(module, class_name, None)
    if strategy_class is None:
        logger.error(
            "Class '%s' not found in module '%s'", class_name, module_path
        )
        sys.exit(1)

    return strategy_class(params)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a backtest on historical OHLCV data."
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to the YAML configuration file (default: config/default.yaml).",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate an HTML performance report after the backtest.",
    )
    args = parser.parse_args()

    # --- Load config ---
    cfg = load_config(args.config)

    # --- Logging ---
    log_level = cfg.get("logging", {}).get("level", "INFO")
    log_file = cfg.get("logging", {}).get("file")
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )

    logger.info("Config loaded from %s", args.config)

    # --- Data ---
    symbol = cfg["symbol"]
    timeframe = cfg["timeframe"]
    bt_cfg = cfg["backtest"]
    data_file = bt_cfg.get("data_file", "")

    if data_file:
        # Load from local CSV (no exchange connection needed).
        df = DataProvider.load_csv(data_file)
    else:
        provider = DataProvider(
            exchange_id=cfg["exchange"]["name"],
            sandbox=cfg["exchange"].get("sandbox", True),
            api_key=cfg["exchange"].get("api_key", ""),
            api_secret=cfg["exchange"].get("api_secret", ""),
        )
        df = provider.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=bt_cfg.get("start_date"),
            until=bt_cfg.get("end_date"),
        )

    if df.empty:
        logger.error("No data available — aborting backtest.")
        sys.exit(1)

    # Attach symbol as metadata so the engine can reference it.
    df.attrs["symbol"] = symbol

    logger.info(
        "Data: %d bars from %s to %s", len(df), df.index[0], df.index[-1]
    )

    # --- Strategy ---
    strat_cfg = cfg["strategy"]
    strategy = load_strategy(strat_cfg["name"], strat_cfg.get("params", {}))
    logger.info("Strategy: %s", strat_cfg["name"])

    # --- Risk manager ---
    risk_cfg = cfg.get("risk", {})
    risk_manager = RiskManager(risk_cfg)

    # --- Engine ---
    engine = BacktestEngine(
        initial_capital=bt_cfg.get("initial_capital", 10_000.0),
        commission_pct=risk_cfg.get("commission_pct", 0.001),
    )

    # --- Run ---
    result = engine.run(df, strategy, risk_manager)

    # --- Report ---
    reporter = PerformanceReporter(result)
    reporter.print_summary()

    if args.report:
        reporter.generate_report()
        print("HTML report written to backtest_report.html")


if __name__ == "__main__":
    main()
