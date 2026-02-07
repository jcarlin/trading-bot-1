"""Simple Moving Average Crossover strategy.

Generates ENTER_LONG when the fast SMA crosses above the slow SMA,
and EXIT_LONG when the fast SMA crosses below the slow SMA.
"""

from typing import Any

import pandas as pd

from core.models import Signal
from core.types import SignalType
from strategy.base import BaseStrategy


class SmaCrossoverStrategy(BaseStrategy):
    """SMA crossover strategy.

    Params (from config):
        fast_period: Lookback for the fast (short-term) SMA.
        slow_period: Lookback for the slow (long-term) SMA.
        stop_loss_pct: Percentage below entry for the stop-loss.
        take_profit_pct: Percentage above entry for the take-profit.
    """

    def __init__(self, params: dict[str, Any]):
        super().__init__(params)
        self.fast_period: int = int(params.get("fast_period", 20))
        self.slow_period: int = int(params.get("slow_period", 50))
        self.stop_loss_pct: float = float(params.get("stop_loss_pct", 0.02))
        self.take_profit_pct: float = float(params.get("take_profit_pct", 0.05))

    def setup(self, df: pd.DataFrame) -> None:
        """Compute fast and slow SMAs and store them in self.indicators."""
        self.indicators["sma_fast"] = df["close"].rolling(window=self.fast_period).mean()
        self.indicators["sma_slow"] = df["close"].rolling(window=self.slow_period).mean()

    def generate_signal(self, index: int, df: pd.DataFrame) -> Signal:
        """Generate a signal based on SMA crossover at the given bar index.

        Returns HOLD if there isn't enough data for both SMAs. Otherwise
        checks for a crossover between the previous bar and the current bar.
        """
        # Need at least slow_period bars AND a previous bar to detect a cross
        if index < self.slow_period:
            return self.hold_signal(df, index)

        sma_fast = self.indicators["sma_fast"]
        sma_slow = self.indicators["sma_slow"]

        # Ensure indicator values are available (not NaN)
        if pd.isna(sma_fast.iloc[index]) or pd.isna(sma_slow.iloc[index]):
            return self.hold_signal(df, index)
        if pd.isna(sma_fast.iloc[index - 1]) or pd.isna(sma_slow.iloc[index - 1]):
            return self.hold_signal(df, index)

        # Current and previous relationship between fast and slow SMA
        fast_now = sma_fast.iloc[index]
        slow_now = sma_slow.iloc[index]
        fast_prev = sma_fast.iloc[index - 1]
        slow_prev = sma_slow.iloc[index - 1]

        price = df["close"].iloc[index]
        timestamp = df.index[index]

        # Bullish crossover: fast was at or below slow, now above
        if fast_prev <= slow_prev and fast_now > slow_now:
            stop_loss = price * (1 - self.stop_loss_pct)
            take_profit = price * (1 + self.take_profit_pct)
            return Signal(
                signal_type=SignalType.ENTER_LONG,
                price=price,
                timestamp=timestamp,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )

        # Bearish crossover: fast was at or above slow, now below
        if fast_prev >= slow_prev and fast_now < slow_now:
            return Signal(
                signal_type=SignalType.EXIT_LONG,
                price=price,
                timestamp=timestamp,
            )

        return self.hold_signal(df, index)
