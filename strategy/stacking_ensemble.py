"""Stacking ensemble strategy: meta-model over sub-strategy signals.

Uses a trained meta-model (logistic regression) to combine sub-strategy
signals into a single trading decision. Falls back to weighted averaging
when no model is loaded.
"""

import json
import logging
import math
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from core.models import Signal
from core.types import SignalType
from strategy.base import BaseStrategy
from strategy.live_strategy import MarketState
from strategy.meta_strategy import MetaStrategy
from strategy.signal_aggregator import SignalAggregator

logger = logging.getLogger(__name__)


class StackingEnsembleStrategy(MetaStrategy):
    """Stacking ensemble: trained meta-model on sub-strategy signals.

    The meta-model is a simple logistic regression. When no model is loaded,
    falls back to a weighted average of sub-strategy directions.

    Params:
        model_path: Path to JSON file containing model weights. None uses fallback.
        feature_mode: "signals_only", "signals_plus_regime", or "full".
        probability_threshold: Minimum probability to enter a position.
        position_size_pct: Position size as fraction of equity.
    """

    def __init__(self, params: dict[str, Any],
                 signal_aggregator: SignalAggregator):
        super().__init__(params, signal_aggregator)
        self.model_path = params.get("model_path")
        self.feature_mode = params.get("feature_mode", "signals_only")
        self.probability_threshold = float(params.get("probability_threshold", 0.65))
        self.position_size_pct = float(params.get("position_size_pct", 0.03))

        # Model weights (logistic regression: w^T x + b)
        self._weights: Optional[list[float]] = None
        self._bias: float = 0.0
        self._model_loaded: bool = False

        # State
        self._position_side: Optional[str] = None
        self._entry_price: Optional[float] = None
        self._tick_count: int = 0

        # Try to load model
        if self.model_path:
            self._load_model(self.model_path)

    def _load_model(self, path: str) -> None:
        """Load logistic regression weights from JSON file."""
        try:
            with open(path, "r") as f:
                model_data = json.load(f)
            self._weights = model_data.get("weights", [])
            self._bias = float(model_data.get("bias", 0.0))
            self._model_loaded = bool(self._weights)
            if self._model_loaded:
                logger.info("Loaded stacking model from %s (%d weights)",
                            path, len(self._weights))
        except Exception:
            logger.warning("Failed to load stacking model from %s", path)
            self._model_loaded = False

    def on_tick(self, market_state: MarketState) -> Signal:
        """Process sub-strategy signals through meta-model."""
        price = market_state.mark_price
        timestamp = market_state.timestamp
        self._tick_count += 1

        if price <= 0:
            return self._hold(price, timestamp)

        signals = self._get_sub_signals()
        if not signals:
            return self._hold(price, timestamp)

        # Build feature vector
        features = self._build_feature_vector(signals, market_state)

        # Predict direction and probability
        direction, probability = self._predict(features, signals)

        position_size = (market_state.equity * self.position_size_pct / price
                         if price > 0 and market_state.equity > 0 else 0)

        # Update Prometheus metrics
        try:
            from monitoring.metrics import (
                meta_strategy_confidence,
                meta_strategy_sub_signal_count,
            )
            meta_strategy_confidence.labels(strategy_name="stacking_ensemble").set(probability)
            meta_strategy_sub_signal_count.labels(strategy_name="stacking_ensemble").set(len(signals))
        except Exception:
            pass

        # Check exit first
        if self._position_side is not None:
            return self._check_exit(direction, probability, price, timestamp, position_size)

        # Check entry
        if probability >= self.probability_threshold and direction != "neutral":
            return self._enter(direction, price, timestamp, position_size, probability)

        return self._hold(price, timestamp)

    def _build_feature_vector(self, signals: dict[str, dict],
                              market_state: MarketState) -> list[float]:
        """Build feature vector from sub-strategy signals and market state.

        Feature layout depends on feature_mode:
        - signals_only: [direction_1, health_1, direction_2, health_2, ...]
        - signals_plus_regime: signals_only + [regime_encoded]
        - full: signals_plus_regime + [drawdown_pct, spread]

        Direction encoding: bullish=1.0, neutral=0.0, bearish=-1.0
        """
        features = []
        direction_map = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}

        # Sort by strategy name for consistent ordering
        for name in sorted(self.aggregator.strategy_names):
            record = signals.get(name)
            if record:
                d = direction_map.get(record.get("direction", "neutral"), 0.0)
                h = record.get("health_score", 50.0) / 100.0
            else:
                d = 0.0
                h = 0.5
            features.append(d)
            features.append(h)

        if self.feature_mode in ("signals_plus_regime", "full"):
            regime = market_state.metadata.get("regime", "unknown")
            regime_map = {
                "trending_up": 1.0,
                "trending_down": -1.0,
                "ranging": 0.0,
                "volatile": 0.5,
                "unknown": 0.0,
            }
            features.append(regime_map.get(regime, 0.0))

        if self.feature_mode == "full":
            features.append(market_state.drawdown_pct / 100.0 if market_state.drawdown_pct else 0.0)
            features.append(market_state.spread)

        return features

    def _predict(self, features: list[float],
                 signals: dict[str, dict]) -> tuple[str, float]:
        """Predict direction and probability.

        Uses logistic regression if model loaded, otherwise falls back
        to weighted average of sub-strategy directions.

        Args:
            features: Feature vector.
            signals: Latest sub-strategy signals (for fallback).

        Returns:
            Tuple of (direction: "long"/"short"/"neutral", probability: 0-1).
        """
        if self._model_loaded and self._weights:
            return self._predict_with_model(features)
        return self._predict_fallback(signals)

    def _predict_with_model(self, features: list[float]) -> tuple[str, float]:
        """Predict using logistic regression model."""
        # Compute logit: w^T x + b
        n = min(len(self._weights), len(features))
        logit = sum(self._weights[i] * features[i] for i in range(n)) + self._bias

        # Sigmoid
        probability = 1.0 / (1.0 + math.exp(-max(min(logit, 500), -500)))

        # > 0.5 means long, < 0.5 means short
        if probability >= 0.5:
            direction = "long"
            conf = probability
        else:
            direction = "short"
            conf = 1.0 - probability

        return direction, conf

    def _predict_fallback(self, signals: dict[str, dict]) -> tuple[str, float]:
        """Fallback prediction using weighted average of directions."""
        if not signals:
            return "neutral", 0.0

        bullish_weight = 0.0
        bearish_weight = 0.0
        total = 0.0

        for name, record in signals.items():
            h = max(record.get("health_score", 50.0) / 100.0, 0.01)
            direction = record.get("direction", "neutral")
            if direction == "bullish":
                bullish_weight += h
            elif direction == "bearish":
                bearish_weight += h
            total += h

        if total <= 0:
            return "neutral", 0.0

        bullish_pct = bullish_weight / total
        bearish_pct = bearish_weight / total

        if bullish_pct > bearish_pct:
            return "long", bullish_pct
        elif bearish_pct > bullish_pct:
            return "short", bearish_pct
        return "neutral", 0.0

    def _enter(self, direction: str, price: float, timestamp: datetime,
               size: float, probability: float) -> Signal:
        """Generate entry signal."""
        if direction == "long":
            self._position_side = "long"
            self._entry_price = price
            return Signal(
                signal_type=SignalType.ENTER_LONG,
                price=price,
                timestamp=timestamp,
                size=size,
                metadata={
                    "strategy": "stacking_ensemble",
                    "probability": round(probability, 4),
                    "feature_mode": self.feature_mode,
                    "model_loaded": self._model_loaded,
                },
            )
        elif direction == "short":
            self._position_side = "short"
            self._entry_price = price
            return Signal(
                signal_type=SignalType.ENTER_SHORT,
                price=price,
                timestamp=timestamp,
                size=size,
                metadata={
                    "strategy": "stacking_ensemble",
                    "probability": round(probability, 4),
                    "feature_mode": self.feature_mode,
                    "model_loaded": self._model_loaded,
                },
            )
        return self._hold(price, timestamp)

    def _check_exit(self, direction: str, probability: float, price: float,
                    timestamp: datetime, size: float) -> Signal:
        """Check exit conditions."""
        should_exit = False
        exit_reason = ""

        # Exit if direction flips with confidence
        if self._position_side == "long" and direction == "short" and probability >= 0.55:
            should_exit = True
            exit_reason = "direction_flip"
        elif self._position_side == "short" and direction == "long" and probability >= 0.55:
            should_exit = True
            exit_reason = "direction_flip"

        if should_exit:
            st = (SignalType.EXIT_LONG if self._position_side == "long"
                  else SignalType.EXIT_SHORT)
            self._position_side = None
            self._entry_price = None
            return Signal(
                signal_type=st,
                price=price,
                timestamp=timestamp,
                size=size,
                metadata={"exit_reason": exit_reason, "strategy": "stacking_ensemble"},
            )

        return self._hold(price, timestamp)

    def _hold(self, price: float, timestamp: datetime) -> Signal:
        return Signal(
            signal_type=SignalType.HOLD,
            price=price,
            timestamp=timestamp,
        )

    def get_metadata(self) -> dict:
        return {
            "name": "stacking_ensemble",
            "version": "1.0",
            "category": "stacking_ensemble",
            "timeframes": ["1h"],
            "description": "Stacking ensemble with logistic regression meta-model",
            "params": {
                "model_path": self.model_path,
                "feature_mode": self.feature_mode,
                "probability_threshold": self.probability_threshold,
                "position_size_pct": self.position_size_pct,
            },
        }

    def get_state(self) -> dict:
        return {
            "position_side": self._position_side,
            "entry_price": self._entry_price,
            "tick_count": self._tick_count,
            "model_loaded": self._model_loaded,
        }

    def set_state(self, state: dict) -> None:
        self._position_side = state.get("position_side")
        self._entry_price = state.get("entry_price")
        self._tick_count = state.get("tick_count", 0)


# ---------------------------------------------------------------------------
# Backtest adapter
# ---------------------------------------------------------------------------


class StackingEnsembleBacktestAdapter(BaseStrategy):
    """Backtest adapter for StackingEnsembleStrategy.

    In backtest mode, uses indicator-based proxy signals and a simple
    weighted average to approximate the stacking ensemble.
    """

    def __init__(self, params: dict[str, Any]):
        super().__init__(params)
        self.probability_threshold = float(params.get("probability_threshold", 0.65))
        self.position_size_pct = float(params.get("position_size_pct", 0.03))

        # State
        self._position_side: Optional[str] = None

    def setup(self, df: pd.DataFrame) -> None:
        """Compute proxy sub-strategy indicators."""
        closes = df["close"]

        # Proxy 1: EMA momentum direction
        ema_fast = closes.ewm(span=12, adjust=False).mean()
        ema_slow = closes.ewm(span=26, adjust=False).mean()
        self.indicators["momentum_dir"] = ((ema_fast - ema_slow) /
                                            ema_slow.replace(0, 1)).clip(-1, 1)

        # Proxy 2: RSI normalized (-1 to +1, centered at 50)
        delta = closes.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss.replace(0, float("inf"))
        rsi = 100.0 - (100.0 / (1.0 + rs))
        self.indicators["rsi_norm"] = (rsi - 50.0) / 50.0

        # Proxy 3: BB position normalized (-1 to +1)
        bb_mid = closes.rolling(20).mean()
        bb_std = closes.rolling(20).std().replace(0, 1)
        self.indicators["bb_pos"] = ((closes - bb_mid) / (2 * bb_std)).clip(-1, 1)

        # Combined signal: simple average
        self.indicators["combined"] = (
            self.indicators["momentum_dir"] +
            self.indicators["rsi_norm"] +
            self.indicators["bb_pos"]
        ) / 3.0

    def generate_signal(self, index: int, df: pd.DataFrame) -> Signal:
        """Generate signal from combined proxy."""
        if index < 26:
            return self.hold_signal(df, index)

        price = float(df["close"].iloc[index])
        timestamp = df.index[index]
        combined = float(self.indicators["combined"].iloc[index])

        # Convert to probability-like score (sigmoid-ish)
        probability = 1.0 / (1.0 + math.exp(-3.0 * combined))
        size = 10000 * self.position_size_pct / price if price > 0 else 0

        # Check exit
        if self._position_side is not None:
            should_exit = False
            if self._position_side == "long" and probability < 0.45:
                should_exit = True
            elif self._position_side == "short" and probability > 0.55:
                should_exit = True

            if should_exit:
                st = (SignalType.EXIT_LONG if self._position_side == "long"
                      else SignalType.EXIT_SHORT)
                self._position_side = None
                return Signal(signal_type=st, price=price, timestamp=timestamp,
                              size=size, metadata={"exit_reason": "direction_flip"})

            return self.hold_signal(df, index)

        # Check entry
        if probability >= self.probability_threshold:
            self._position_side = "long"
            return Signal(signal_type=SignalType.ENTER_LONG, price=price,
                          timestamp=timestamp, size=size,
                          metadata={"probability": probability})
        elif (1.0 - probability) >= self.probability_threshold:
            self._position_side = "short"
            return Signal(signal_type=SignalType.ENTER_SHORT, price=price,
                          timestamp=timestamp, size=size,
                          metadata={"probability": 1.0 - probability})

        return self.hold_signal(df, index)
