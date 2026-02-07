"""Base strategy interface for systematic trading.

All strategies must subclass BaseStrategy and implement:
- setup(): Called once to initialize indicators
- generate_signal(): Called on each bar to produce a trading signal

Strategies should be:
- Stateless where possible (state tracked via indicators/lookback)
- Config-driven (parameters via self.params)
- Deterministic (same input -> same output)
"""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from core.models import Signal
from core.types import SignalType


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies.

    Attributes:
        params: Strategy parameters from config.
        indicators: Dict of computed indicator Series, populated by setup().
    """

    def __init__(self, params: dict[str, Any]):
        self.params = params
        self.indicators: dict[str, pd.Series] = {}

    @abstractmethod
    def setup(self, df: pd.DataFrame) -> None:
        """Compute indicators on the full OHLCV DataFrame.

        Called once before backtesting or at startup for paper trading.
        Should populate self.indicators with named pd.Series.

        Args:
            df: OHLCV DataFrame with columns [open, high, low, close, volume]
                and a DatetimeIndex.
        """
        ...

    @abstractmethod
    def generate_signal(self, index: int, df: pd.DataFrame) -> Signal:
        """Generate a trading signal for the bar at the given index.

        Called on each bar during backtesting or paper trading.

        Args:
            index: Integer position of the current bar in df.
            df: Full OHLCV DataFrame (strategy can look back but not forward).

        Returns:
            A Signal object indicating the desired action.
        """
        ...

    def hold_signal(self, df: pd.DataFrame, index: int) -> Signal:
        """Convenience: return a HOLD signal."""
        return Signal(
            signal_type=SignalType.HOLD,
            price=df["close"].iloc[index],
            timestamp=df.index[index],
        )
