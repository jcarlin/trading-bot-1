"""Core type definitions and enums for the trading system."""

from enum import Enum, auto


class Side(str, Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"


class SignalType(str, Enum):
    """Trading signal type."""
    ENTER_LONG = "enter_long"
    EXIT_LONG = "exit_long"
    ENTER_SHORT = "enter_short"
    EXIT_SHORT = "exit_short"
    HOLD = "hold"


class PositionSizing(str, Enum):
    """Position sizing method."""
    FIXED_AMOUNT = "fixed_amount"
    PERCENT_EQUITY = "percent_equity"
    PERCENT_RISK = "percent_risk"


class TimeFrame(str, Enum):
    """OHLCV timeframe."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"
