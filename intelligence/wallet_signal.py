"""Wallet signal model for real-time wallet monitoring."""

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class WalletSignal:
    """A signal detected from wallet position changes.

    Attributes:
        timestamp: When the signal was detected.
        wallet_address: The wallet that triggered the signal.
        wallet_score: Score of the wallet (0-100).
        symbol: Trading pair symbol.
        signal_type: Type of change detected.
        direction: Position direction (long/short).
        size: Current position size.
        size_change_pct: Percentage change in position size.
        is_unusual: Whether the size change is statistically unusual.
        confidence: Signal confidence based on wallet score and type.
        metadata: Additional signal context.
    """
    timestamp: datetime
    wallet_address: str
    wallet_score: float
    symbol: str
    signal_type: str  # "new_position" | "size_increase" | "size_decrease" | "closed"
    direction: str  # "long" | "short"
    size: float
    size_change_pct: float
    is_unusual: bool
    confidence: float
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to dict for storage and transport."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "wallet_address": self.wallet_address,
            "wallet_score": self.wallet_score,
            "symbol": self.symbol,
            "signal_type": self.signal_type,
            "direction": self.direction,
            "size": self.size,
            "size_change_pct": self.size_change_pct,
            "is_unusual": self.is_unusual,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }
