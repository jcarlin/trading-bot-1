"""Centralized configuration loader with YAML support and env var substitution."""

import os
import re
import logging
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

# Pattern to match ${VAR:-default} or ${VAR} in config values
_ENV_PATTERN = re.compile(r'\$\{([^}^{]+)\}')


def _substitute_env_vars(value: Any) -> Any:
    """Recursively substitute ${VAR:-default} patterns in config values."""
    if isinstance(value, str):
        def _replace(match):
            expr = match.group(1)
            if ':-' in expr:
                var_name, default = expr.split(':-', 1)
            else:
                var_name = expr
                default = ''
            return os.environ.get(var_name, default)
        return _ENV_PATTERN.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]
    return value


class Config:
    """Centralized configuration with YAML loading, env var substitution, and validation."""

    def __init__(self, data: dict):
        self._data = data

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from a YAML file with environment variable substitution.

        Supports ${VAR:-default} syntax for env var interpolation.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, 'r') as f:
            raw = yaml.safe_load(f) or {}

        data = _substitute_env_vars(raw)
        config = cls(data)
        logger.info("Config loaded from %s", path)
        return config

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value by dot-separated key path.

        Example: config.get('exchange.name') returns config['exchange']['name']
        """
        keys = key.split('.')
        value = self._data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    def require(self, key: str) -> Any:
        """Get a config value, raising ValueError if not found."""
        value = self.get(key)
        if value is None:
            raise ValueError(f"Required config key missing: {key}")
        return value

    def get_section(self, key: str) -> dict:
        """Get a config section as a dict, returning empty dict if not found."""
        value = self.get(key, {})
        if not isinstance(value, dict):
            return {}
        return value

    @property
    def raw(self) -> dict:
        """Access the raw config dict for backward compatibility."""
        return self._data

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def __repr__(self) -> str:
        return f"Config(keys={list(self._data.keys())})"
