"""Live strategy interface and market state model for real-time trading."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from core.models import Signal


@dataclass
class MarketState:
    """Snapshot of all market data for one tick."""
    mark_price: float
    mid_price: float
    bid: float
    ask: float
    funding_rate: float
    premium: float
    open_interest: float
    funding_history: list[dict] = field(default_factory=list)
    position: Optional[dict] = None
    equity: float = 0.0
    cash: float = 0.0
    drawdown_pct: float = 0.0
    recent_candles: list[dict] = field(default_factory=list)
    spread: float = 0.0
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class LiveStrategy(ABC):
    """Abstract base class for live trading strategies.

    Parallel to BaseStrategy but designed for real-time tick-by-tick
    operation rather than historical bar-by-bar backtesting.
    """

    def __init__(self, params: dict[str, Any]):
        self.params = params

    @abstractmethod
    def on_tick(self, market_state: MarketState) -> Signal:
        """Process a market state tick and return a trading signal."""
        ...

    @abstractmethod
    def get_metadata(self) -> dict:
        """Return strategy metadata (name, version, description, etc.)."""
        ...

    def get_state(self) -> dict:
        """Return serializable strategy state for checkpointing."""
        return {}

    def set_state(self, state: dict) -> None:
        """Restore strategy state from a checkpoint."""
        pass

    def get_risk_parameters(self) -> dict:
        """Return strategy-specific risk parameters."""
        return {}
