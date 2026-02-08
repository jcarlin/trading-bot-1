"""Parameter grid builder for walk-forward optimization."""

import itertools
import math
from typing import Any


class ParamGrid:
    """Parameter grid builder for walk-forward optimization.

    Supports:
    - Explicit lists: {"period": [10, 20, 30]}
    - Ranges: {"period": {"start": 10, "stop": 30, "step": 5}}
    - Log-scale: {"threshold": {"start": 0.001, "stop": 0.1, "scale": "log", "n": 5}}
    """

    def __init__(self, grid_config: dict):
        self._config = grid_config
        self._expanded_axes: dict[str, list[Any]] = {}
        self._parse()

    def _parse(self) -> None:
        """Parse grid config into expanded axes per parameter."""
        for param_name, spec in self._config.items():
            if isinstance(spec, list):
                self._expanded_axes[param_name] = list(spec)
            elif isinstance(spec, dict):
                if spec.get("scale") == "log":
                    self._expanded_axes[param_name] = self._expand_log(spec)
                else:
                    self._expanded_axes[param_name] = self._expand_range(spec)
            else:
                # Single value
                self._expanded_axes[param_name] = [spec]

    @staticmethod
    def _expand_range(spec: dict) -> list:
        """Expand a range spec into a list of values."""
        start = spec["start"]
        stop = spec["stop"]
        step = spec.get("step", 1)
        values = []
        current = start
        while current <= stop:
            if isinstance(start, int) and isinstance(step, int):
                values.append(int(current))
            else:
                values.append(round(current, 10))
            current += step
        return values

    @staticmethod
    def _expand_log(spec: dict) -> list:
        """Expand a log-scale spec into a list of values."""
        start = spec["start"]
        stop = spec["stop"]
        n = spec.get("n", 5)
        if n <= 1:
            return [start]
        log_start = math.log10(start)
        log_stop = math.log10(stop)
        step = (log_stop - log_start) / (n - 1)
        return [round(10 ** (log_start + i * step), 10) for i in range(n)]

    def expand(self) -> list[dict]:
        """Expand grid into list of parameter dictionaries (cartesian product)."""
        if not self._expanded_axes:
            return [{}]
        keys = list(self._expanded_axes.keys())
        value_lists = [self._expanded_axes[k] for k in keys]
        return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]

    def count(self) -> int:
        """Return the total number of parameter combinations."""
        if not self._expanded_axes:
            return 1
        total = 1
        for values in self._expanded_axes.values():
            total *= len(values)
        return total
