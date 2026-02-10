"""BTC/ETH correlation breakdown strategy.

Entry: When BTC/ETH correlation drops below threshold AND
       the spread z-score exceeds threshold (divergence detected).
Exit: Spread z-score reverts below reversion threshold, max hold, or drawdown.

Expects MarketState.metadata["eth_price"] to be populated.
"""

import logging
import math
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from core.models import Signal
from core.types import SignalType
from strategy.base import BaseStrategy
from strategy.live_strategy import LiveStrategy, MarketState

logger = logging.getLogger(__name__)


class CorrelationDivergenceStrategy(LiveStrategy):
    """BTC/ETH correlation breakdown strategy."""

    def __init__(self, params: dict[str, Any]):
        super().__init__(params)
        self.correlation_window = int(params.get("correlation_window", 168))
        self.correlation_threshold = float(params.get("correlation_threshold", 0.7))
        self.zscore_threshold = float(params.get("zscore_threshold", 2.0))
        self.max_hold_periods = int(params.get("max_hold_periods", 48))
        self.position_size_pct = float(params.get("position_size_pct", 0.02))
        self.reversion_zscore = float(params.get("reversion_zscore", 0.5))
        self.drawdown_exit_pct = float(params.get("drawdown_exit_pct", 3.0))

        # Internal state
        self._btc_prices: list[float] = []
        self._eth_prices: list[float] = []
        self._position_side = None
        self._entry_price = None
        self._hold_periods = 0

    def on_tick(self, market_state: MarketState) -> Signal:
        price = market_state.mark_price  # BTC price
        eth_price = market_state.metadata.get("eth_price")
        timestamp = market_state.timestamp

        if price <= 0 or not eth_price or eth_price <= 0:
            return self._hold(price, timestamp)

        self._btc_prices.append(price)
        self._eth_prices.append(eth_price)

        # Keep window size
        if len(self._btc_prices) > self.correlation_window + 50:
            self._btc_prices = self._btc_prices[-(self.correlation_window + 50):]
            self._eth_prices = self._eth_prices[-(self.correlation_window + 50):]

        # Need enough data
        if len(self._btc_prices) < self.correlation_window:
            return self._hold(price, timestamp)

        # Check exit first
        if self._position_side is not None:
            return self._check_exit(price, timestamp)

        # Compute correlation and spread z-score
        corr = self._compute_correlation()
        zscore = self._compute_spread_zscore()

        # Entry: low correlation + high z-score
        if corr < self.correlation_threshold and abs(zscore) > self.zscore_threshold:
            position_size = market_state.equity * self.position_size_pct / price if price > 0 else 0

            if zscore > self.zscore_threshold:
                # BTC outperforming ETH -> expect reversion -> short BTC
                self._position_side = "short"
                self._entry_price = price
                self._hold_periods = 0
                return Signal(
                    signal_type=SignalType.ENTER_SHORT,
                    price=price,
                    timestamp=timestamp,
                    size=position_size,
                    metadata={
                        "correlation": round(corr, 4),
                        "zscore": round(zscore, 4),
                        "entry_reason": "correlation_divergence_short",
                    },
                )
            elif zscore < -self.zscore_threshold:
                # ETH outperforming BTC -> expect reversion -> long BTC
                self._position_side = "long"
                self._entry_price = price
                self._hold_periods = 0
                return Signal(
                    signal_type=SignalType.ENTER_LONG,
                    price=price,
                    timestamp=timestamp,
                    size=position_size,
                    metadata={
                        "correlation": round(corr, 4),
                        "zscore": round(zscore, 4),
                        "entry_reason": "correlation_divergence_long",
                    },
                )

        return self._hold(price, timestamp)

    def _compute_correlation(self) -> float:
        """Pearson correlation on BTC/ETH log returns over correlation_window."""
        btc = self._btc_prices[-self.correlation_window:]
        eth = self._eth_prices[-self.correlation_window:]

        btc_returns = [math.log(btc[i] / btc[i - 1]) for i in range(1, len(btc)) if btc[i - 1] > 0]
        eth_returns = [math.log(eth[i] / eth[i - 1]) for i in range(1, len(eth)) if eth[i - 1] > 0]

        if len(btc_returns) < 2 or len(eth_returns) < 2:
            return 1.0  # Assume correlated if insufficient data

        n = min(len(btc_returns), len(eth_returns))
        btc_r = btc_returns[-n:]
        eth_r = eth_returns[-n:]

        mean_b = sum(btc_r) / n
        mean_e = sum(eth_r) / n

        cov = sum((btc_r[i] - mean_b) * (eth_r[i] - mean_e) for i in range(n)) / n
        std_b = math.sqrt(sum((x - mean_b) ** 2 for x in btc_r) / n)
        std_e = math.sqrt(sum((x - mean_e) ** 2 for x in eth_r) / n)

        if std_b < 1e-12 or std_e < 1e-12:
            return 1.0

        return cov / (std_b * std_e)

    def _compute_spread_zscore(self) -> float:
        """Z-score of BTC/ETH price ratio vs its rolling mean/std."""
        btc = self._btc_prices[-self.correlation_window:]
        eth = self._eth_prices[-self.correlation_window:]

        ratios = [btc[i] / eth[i] for i in range(len(btc)) if eth[i] > 0]
        if len(ratios) < 2:
            return 0.0

        mean_r = sum(ratios) / len(ratios)
        std_r = math.sqrt(sum((r - mean_r) ** 2 for r in ratios) / len(ratios))

        if std_r < 1e-12:
            return 0.0

        current_ratio = ratios[-1]
        return (current_ratio - mean_r) / std_r

    def _check_exit(self, price, timestamp) -> Signal:
        self._hold_periods += 1
        should_exit = False
        exit_reason = ""

        # Max hold
        if self._hold_periods >= self.max_hold_periods:
            should_exit = True
            exit_reason = "max_hold_period"

        # Drawdown
        if self._entry_price and self._entry_price > 0:
            if self._position_side == "long":
                dd_pct = (self._entry_price - price) / self._entry_price * 100
            else:
                dd_pct = (price - self._entry_price) / self._entry_price * 100
            if dd_pct >= self.drawdown_exit_pct:
                should_exit = True
                exit_reason = "drawdown_exit"

        # Z-score reversion
        if not should_exit:
            zscore = self._compute_spread_zscore()
            if abs(zscore) < self.reversion_zscore:
                should_exit = True
                exit_reason = "zscore_reversion"

        if should_exit:
            signal_type = SignalType.EXIT_LONG if self._position_side == "long" else SignalType.EXIT_SHORT
            self._position_side = None
            self._entry_price = None
            self._hold_periods = 0
            return Signal(
                signal_type=signal_type,
                price=price,
                timestamp=timestamp,
                metadata={"exit_reason": exit_reason},
            )

        return self._hold(price, timestamp)

    def _hold(self, price, timestamp):
        return Signal(signal_type=SignalType.HOLD, price=price, timestamp=timestamp)

    def get_metadata(self) -> dict:
        return {
            "name": "correlation_divergence",
            "version": "1.0",
            "category": "correlation_divergence",
            "timeframes": ["1h"],
            "description": "BTC/ETH correlation breakdown strategy",
            "params": {
                "correlation_window": self.correlation_window,
                "correlation_threshold": self.correlation_threshold,
                "zscore_threshold": self.zscore_threshold,
                "max_hold_periods": self.max_hold_periods,
                "position_size_pct": self.position_size_pct,
                "reversion_zscore": self.reversion_zscore,
                "drawdown_exit_pct": self.drawdown_exit_pct,
            },
        }

    def get_state(self) -> dict:
        return {
            "position_side": self._position_side,
            "entry_price": self._entry_price,
            "hold_periods": self._hold_periods,
            "btc_prices": self._btc_prices[-200:],
            "eth_prices": self._eth_prices[-200:],
        }

    def set_state(self, state: dict) -> None:
        self._position_side = state.get("position_side")
        self._entry_price = state.get("entry_price")
        self._hold_periods = state.get("hold_periods", 0)
        self._btc_prices = state.get("btc_prices", [])
        self._eth_prices = state.get("eth_prices", [])


class CorrelationDivergenceBacktestAdapter(BaseStrategy):
    """Backtest adapter for CorrelationDivergenceStrategy."""

    def __init__(self, params: dict[str, Any]):
        super().__init__(params)
        self.live_strategy = CorrelationDivergenceStrategy(params)

    def setup(self, df: pd.DataFrame) -> None:
        if "eth_close" not in df.columns:
            raise ValueError(
                "DataFrame must contain 'eth_close' column for CorrelationDivergenceStrategy"
            )

    def generate_signal(self, index: int, df: pd.DataFrame) -> Signal:
        row = df.iloc[index]
        market_state = MarketState(
            mark_price=float(row["close"]),
            mid_price=float(row["close"]),
            bid=float(row["close"]) - 0.5,
            ask=float(row["close"]) + 0.5,
            funding_rate=0.0,
            premium=0.0,
            open_interest=0.0,
            equity=10000.0,
            cash=5000.0,
            timestamp=df.index[index] if hasattr(df.index[index], "tzinfo") else datetime.now(timezone.utc),
            metadata={"eth_price": float(row.get("eth_close", 0))},
        )
        signal = self.live_strategy.on_tick(market_state)
        signal.timestamp = df.index[index] if hasattr(df.index, "__getitem__") else signal.timestamp
        return signal
