"""Order flow imbalance strategy.

Entry: When order flow imbalance exceeds threshold, confirmed by CVD trend.
Exit: Imbalance normalization, max hold period, or drawdown.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from core.models import Signal
from core.types import SignalType
from strategy.base import BaseStrategy
from strategy.live_strategy import LiveStrategy, MarketState

logger = logging.getLogger(__name__)


class OrderFlowImbalanceStrategy(LiveStrategy):
    """Order flow imbalance strategy.

    Trades when buy/sell order flow imbalance exceeds a threshold,
    optionally confirmed by cumulative volume delta trend alignment.
    Liquidation cascades can boost entry confidence.
    """

    def __init__(self, params: dict[str, Any], order_flow_analyzer=None,
                 liquidation_aggregator=None):
        super().__init__(params)
        self.order_flow_analyzer = order_flow_analyzer
        self.liquidation_aggregator = liquidation_aggregator

        self.imbalance_threshold = float(
            params.get("imbalance_threshold", 0.65))
        self.cvd_confirmation = bool(
            params.get("cvd_confirmation", True))
        self.min_volume = float(params.get("min_volume", 1000))
        self.large_trade_boost = float(
            params.get("large_trade_boost", 0.1))
        self.max_hold_periods = int(params.get("max_hold_periods", 12))
        self.position_size_pct = float(
            params.get("position_size_pct", 0.02))
        self.drawdown_exit_pct = float(
            params.get("drawdown_exit_pct", 3.0))

        # Position state
        self._position_side: Optional[str] = None
        self._entry_price: Optional[float] = None
        self._hold_periods: int = 0

    def on_tick(self, market_state: MarketState) -> Signal:
        """Process a market state tick and return a trading signal."""
        price = market_state.mark_price
        timestamp = market_state.timestamp

        if price <= 0:
            return self._hold(price, timestamp)

        order_flow = self._get_order_flow(market_state)
        if not order_flow:
            return self._hold(price, timestamp)

        # Check exit first if in position
        if self._position_side is not None:
            return self._check_exit(price, timestamp, order_flow)

        # Check volume minimum
        buy_vol = float(order_flow.get("buy_vol", 0))
        sell_vol = float(order_flow.get("sell_vol", 0))
        total_vol = buy_vol + sell_vol
        if total_vol < self.min_volume:
            return self._hold(price, timestamp)

        imbalance_ratio = float(order_flow.get("imbalance_ratio", 0.5))
        cvd_trend = order_flow.get("cvd_trend", "neutral")

        position_size = (market_state.equity * self.position_size_pct / price
                         if price > 0 else 0)

        # Check for liquidation cascade boost
        cascade_boost = self._get_cascade_boost(market_state)

        # Long entry: high buy imbalance
        effective_threshold = self.imbalance_threshold - cascade_boost
        if imbalance_ratio >= effective_threshold:
            # CVD confirmation: trend must be bullish
            if self.cvd_confirmation and cvd_trend != "bullish":
                return self._hold(price, timestamp)

            self._position_side = "long"
            self._entry_price = price
            self._hold_periods = 0
            return Signal(
                signal_type=SignalType.ENTER_LONG,
                price=price,
                timestamp=timestamp,
                size=position_size,
                metadata={
                    "imbalance_ratio": round(imbalance_ratio, 4),
                    "cvd_trend": cvd_trend,
                    "cascade_boost": round(cascade_boost, 4),
                    "entry_reason": "order_flow_imbalance_long",
                },
            )

        # Short entry: high sell imbalance (low buy ratio)
        if (1 - imbalance_ratio) >= effective_threshold:
            # CVD confirmation: trend must be bearish
            if self.cvd_confirmation and cvd_trend != "bearish":
                return self._hold(price, timestamp)

            self._position_side = "short"
            self._entry_price = price
            self._hold_periods = 0
            return Signal(
                signal_type=SignalType.ENTER_SHORT,
                price=price,
                timestamp=timestamp,
                size=position_size,
                metadata={
                    "imbalance_ratio": round(imbalance_ratio, 4),
                    "cvd_trend": cvd_trend,
                    "cascade_boost": round(cascade_boost, 4),
                    "entry_reason": "order_flow_imbalance_short",
                },
            )

        return self._hold(price, timestamp)

    def _get_order_flow(self, market_state: MarketState) -> dict:
        """Get order flow data from analyzer or market state metadata."""
        if self.order_flow_analyzer:
            try:
                imbalance = self.order_flow_analyzer.get_imbalance()
                cvd = self.order_flow_analyzer.get_cumulative_delta()
                return {
                    "buy_vol": imbalance.get("buy_vol", 0),
                    "sell_vol": imbalance.get("sell_vol", 0),
                    "imbalance_ratio": imbalance.get("imbalance_ratio", 0.5),
                    "imbalance_pct": imbalance.get("imbalance_pct", 0),
                    "cvd": cvd.get("cvd", 0),
                    "cvd_normalized": cvd.get("cvd_normalized", 0),
                    "cvd_trend": cvd.get("trend", "neutral"),
                }
            except Exception:
                logger.debug("Failed to get order flow from analyzer")

        # Fall back to metadata
        return market_state.metadata.get("order_flow", {})

    def _get_cascade_boost(self, market_state: MarketState) -> float:
        """Get confidence boost from liquidation cascade."""
        if self.liquidation_aggregator:
            try:
                cascade = self.liquidation_aggregator.detect_cascade()
                if cascade and cascade.get("is_cascade", False):
                    return self.large_trade_boost
            except Exception:
                pass

        # Fall back to metadata
        if market_state.metadata.get("liquidation_cascade", False):
            return self.large_trade_boost

        return 0.0

    def _check_exit(self, price: float, timestamp: datetime,
                    order_flow: dict) -> Signal:
        """Check exit conditions for open position."""
        self._hold_periods += 1
        should_exit = False
        exit_reason = ""

        # Max hold period
        if self._hold_periods >= self.max_hold_periods:
            should_exit = True
            exit_reason = "max_hold_period"

        # Drawdown exit
        if not should_exit and self._entry_price and self._entry_price > 0:
            if self._position_side == "long":
                dd_pct = ((self._entry_price - price) /
                          self._entry_price * 100)
            else:
                dd_pct = ((price - self._entry_price) /
                          self._entry_price * 100)

            if dd_pct >= self.drawdown_exit_pct:
                should_exit = True
                exit_reason = "drawdown_exit"

        # Imbalance normalization exit
        if not should_exit:
            imbalance_ratio = float(
                order_flow.get("imbalance_ratio", 0.5))
            # If we're long but imbalance has normalized or reversed
            if self._position_side == "long" and imbalance_ratio < 0.5:
                should_exit = True
                exit_reason = "imbalance_normalization"
            elif self._position_side == "short" and imbalance_ratio > 0.5:
                should_exit = True
                exit_reason = "imbalance_normalization"

        if should_exit:
            if self._position_side == "long":
                signal_type = SignalType.EXIT_LONG
            else:
                signal_type = SignalType.EXIT_SHORT

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

    def _hold(self, price: float, timestamp: datetime) -> Signal:
        """Return a HOLD signal."""
        return Signal(
            signal_type=SignalType.HOLD,
            price=price,
            timestamp=timestamp,
        )

    def get_metadata(self) -> dict:
        return {
            "name": "order_flow_imbalance",
            "version": "1.0",
            "category": "order_flow_imbalance",
            "timeframes": ["1m", "5m"],
            "description": (
                "Order flow imbalance strategy with CVD confirmation"
            ),
            "params": {
                "imbalance_threshold": self.imbalance_threshold,
                "cvd_confirmation": self.cvd_confirmation,
                "min_volume": self.min_volume,
                "large_trade_boost": self.large_trade_boost,
                "max_hold_periods": self.max_hold_periods,
                "position_size_pct": self.position_size_pct,
                "drawdown_exit_pct": self.drawdown_exit_pct,
            },
        }

    def get_state(self) -> dict:
        return {
            "position_side": self._position_side,
            "entry_price": self._entry_price,
            "hold_periods": self._hold_periods,
        }

    def set_state(self, state: dict) -> None:
        self._position_side = state.get("position_side")
        self._entry_price = state.get("entry_price")
        self._hold_periods = state.get("hold_periods", 0)


# ---------------------------------------------------------------------------
# Backtest adapter
# ---------------------------------------------------------------------------


class OrderFlowImbalanceBacktestAdapter(BaseStrategy):
    """Backtest adapter wrapping OrderFlowImbalanceStrategy for bar-based engine."""

    def __init__(self, params: dict[str, Any]):
        super().__init__(params)
        self.live_strategy = OrderFlowImbalanceStrategy(params)

    def setup(self, df: pd.DataFrame) -> None:
        """Validate or synthesize buy_volume/sell_volume columns.

        If buy_volume/sell_volume not present, synthesize from volume
        with a 50/50 split.
        """
        if "buy_volume" not in df.columns:
            df["buy_volume"] = df["volume"] * 0.5
        if "sell_volume" not in df.columns:
            df["sell_volume"] = df["volume"] * 0.5

    def generate_signal(self, index: int, df: pd.DataFrame) -> Signal:
        """Convert a bar-based call into an event-based MarketState tick."""
        row = df.iloc[index]

        buy_vol = float(row.get("buy_volume", row.get("volume", 0) * 0.5))
        sell_vol = float(row.get("sell_volume", row.get("volume", 0) * 0.5))
        total_vol = buy_vol + sell_vol

        if total_vol > 0:
            imbalance_ratio = buy_vol / total_vol
            imbalance_pct = (buy_vol - sell_vol) / total_vol * 100
        else:
            imbalance_ratio = 0.5
            imbalance_pct = 0.0

        # Simple CVD from buy/sell volumes
        cvd = buy_vol - sell_vol
        cvd_normalized = cvd / total_vol if total_vol > 0 else 0.0
        if cvd > 0:
            cvd_trend = "bullish"
        elif cvd < 0:
            cvd_trend = "bearish"
        else:
            cvd_trend = "neutral"

        order_flow_meta = {
            "buy_vol": buy_vol,
            "sell_vol": sell_vol,
            "imbalance_ratio": imbalance_ratio,
            "imbalance_pct": imbalance_pct,
            "cvd": cvd,
            "cvd_normalized": cvd_normalized,
            "cvd_trend": cvd_trend,
        }

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
            timestamp=(
                df.index[index]
                if hasattr(df.index[index], 'tzinfo')
                else datetime.now(timezone.utc)
            ),
            metadata={"order_flow": order_flow_meta},
        )

        signal = self.live_strategy.on_tick(market_state)
        signal.timestamp = (
            df.index[index]
            if hasattr(df.index, '__getitem__')
            else signal.timestamp
        )
        return signal
