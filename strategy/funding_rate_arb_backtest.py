"""Backtest adapter for the funding rate arbitrage strategy."""

from datetime import datetime, timezone

import pandas as pd

from core.models import Signal
from strategy.base import BaseStrategy
from strategy.funding_rate_arb import FundingRateArbStrategy
from strategy.live_strategy import MarketState


class FundingRateArbBacktestAdapter(BaseStrategy):
    """Wraps FundingRateArbStrategy for use with the BacktestEngine.

    The DataFrame must contain extra columns: funding_rate, premium,
    mark_price, open_interest.
    """

    def __init__(self, params: dict):
        super().__init__(params)
        self._live_strategy = FundingRateArbStrategy(params)

    def setup(self, df: pd.DataFrame) -> None:
        required = ["funding_rate", "mark_price", "open_interest"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

    def generate_signal(self, index: int, df: pd.DataFrame) -> Signal:
        row = df.iloc[index]
        market_state = MarketState(
            mark_price=float(row.get("mark_price", row["close"])),
            mid_price=float(row["close"]),
            bid=float(row["close"]),
            ask=float(row["close"]),
            funding_rate=float(row.get("funding_rate", 0.0)),
            premium=float(row.get("premium", 0.0)),
            open_interest=float(row.get("open_interest", 0.0)),
            equity=10000.0,  # Default for backtest
            timestamp=df.index[index] if isinstance(df.index[index], datetime) else datetime.now(timezone.utc),
        )
        return self._live_strategy.on_tick(market_state)
