#!/usr/bin/env python3
"""CLI entry point for paper trading.

Usage:
    python run_paper.py [--config path/to/config.yaml]

Environment variables:
    EXCHANGE_API_KEY    - Exchange API key (overrides config value)
    EXCHANGE_API_SECRET - Exchange API secret (overrides config value)
"""

import argparse
import logging
import os
import sys

import yaml

from paper.trader import PaperTrader


def load_config(path: str) -> dict:
    """Load and return the YAML configuration file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed config as a dict.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def apply_env_overrides(config: dict) -> dict:
    """Override exchange API credentials from environment variables.

    If EXCHANGE_API_KEY or EXCHANGE_API_SECRET are set in the environment,
    they take precedence over values in the config file.

    Args:
        config: The loaded config dict (modified in place).

    Returns:
        The config dict with overrides applied.
    """
    env_key = os.environ.get("EXCHANGE_API_KEY")
    env_secret = os.environ.get("EXCHANGE_API_SECRET")

    if env_key:
        config.setdefault("exchange", {})["api_key"] = env_key
    if env_secret:
        config.setdefault("exchange", {})["api_secret"] = env_secret

    return config


def setup_logging(config: dict) -> None:
    """Configure the root logger from the config's logging section.

    Args:
        config: The full config dict.
    """
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    log_file = log_cfg.get("file")

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def main() -> None:
    """Parse CLI args, load config, validate settings, and run the paper trader."""
    parser = argparse.ArgumentParser(
        description="Run the paper trading bot against a sandbox exchange."
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to the YAML config file (default: config/default.yaml)",
    )
    args = parser.parse_args()

    # Load and prepare config
    config = load_config(args.config)
    config = apply_env_overrides(config)
    setup_logging(config)

    logger = logging.getLogger(__name__)

    # Safety check: refuse to run if sandbox mode is disabled
    if not config.get("exchange", {}).get("sandbox", False):
        logger.error(
            "Sandbox mode is DISABLED in the config. "
            "Paper trading requires sandbox=true for safety. Exiting."
        )
        print(
            "\nERROR: exchange.sandbox must be true for paper trading.\n"
            "Set 'sandbox: true' in your config or use a sandbox-enabled config.\n"
        )
        sys.exit(1)

    logger.info("Config loaded from %s", args.config)
    logger.info(
        "Trading %s on %s (sandbox=%s)",
        config["symbol"],
        config["exchange"]["name"],
        config["exchange"]["sandbox"],
    )

    # Create and run the paper trader
    trader = PaperTrader(config)

    try:
        trader.run()
    except KeyboardInterrupt:
        pass  # Already handled inside trader.run()
    finally:
        logger.info("Paper trading session ended. %d trades executed.", len(trader.trades))


if __name__ == "__main__":
    main()
