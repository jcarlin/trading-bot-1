"""Structured logging setup with JSON file output and human-readable console output."""

import logging
import sys
from typing import Optional

try:
    from pythonjsonlogger import jsonlogger
    HAS_JSON_LOGGER = True
except ImportError:
    HAS_JSON_LOGGER = False


class _ComponentFilter(logging.Filter):
    """Adds a 'component' field derived from the logger name."""

    def filter(self, record):
        parts = record.name.split('.')
        record.component = parts[0] if parts else 'root'
        return True


def setup_logging(config=None) -> None:
    """Configure structured logging for the trading system.

    Console gets human-readable format, file gets JSON format.

    Args:
        config: Config object or dict with optional keys:
            logging.level (str): Log level, default 'INFO'
            logging.file (str): Log file path, default None
            logging.format (str): 'json' or 'text', controls file format
    """
    if config is None:
        level_str = 'INFO'
        log_file = None
        log_format = 'text'
    elif hasattr(config, 'get'):
        level_str = config.get('logging.level', 'INFO') if hasattr(config, 'require') else config.get('logging', {}).get('level', 'INFO')
        log_file = config.get('logging.file') if hasattr(config, 'require') else config.get('logging', {}).get('file')
        log_format = config.get('logging.format', 'text') if hasattr(config, 'require') else config.get('logging', {}).get('format', 'text')
    else:
        level_str = 'INFO'
        log_file = None
        log_format = 'text'

    level = getattr(logging, str(level_str).upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)

    # Clear existing handlers
    root.handlers.clear()

    component_filter = _ComponentFilter()

    # Console handler: human-readable
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.addFilter(component_filter)
    console_fmt = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(component)s.%(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console.setFormatter(console_fmt)
    root.addHandler(console)

    # File handler: JSON if available and requested
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.addFilter(component_filter)

        if log_format == 'json' and HAS_JSON_LOGGER:
            json_fmt = jsonlogger.JsonFormatter(
                '%(asctime)s %(levelname)s %(component)s %(name)s %(message)s',
                timestamp=True
            )
            file_handler.setFormatter(json_fmt)
        else:
            file_handler.setFormatter(console_fmt)

        root.addHandler(file_handler)

    logger = logging.getLogger(__name__)
    logger.debug("Logging configured: level=%s file=%s format=%s", level_str, log_file, log_format)
