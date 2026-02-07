"""Constants and types for the Hyperliquid exchange adapter."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
MAINNET_API_URL = "https://api.hyperliquid.xyz"
TESTNET_API_URL = "https://api.hyperliquid-testnet.xyz"
MAINNET_WS_URL = "wss://api.hyperliquid.xyz/ws"
TESTNET_WS_URL = "wss://api.hyperliquid-testnet.xyz/ws"

INFO_ENDPOINT = "/info"
EXCHANGE_ENDPOINT = "/exchange"

# ---------------------------------------------------------------------------
# Rate limits
# ---------------------------------------------------------------------------
MAX_REQUESTS_PER_MINUTE = 1200


class HLOrderType(str, Enum):
    """Hyperliquid native order type strings."""
    LIMIT = "Limit"
    MARKET = "Market"


@dataclass
class HLRateLimit:
    """Tracks rate-limit state for a Hyperliquid API connection."""
    requests_per_minute: int = MAX_REQUESTS_PER_MINUTE
    remaining: int = MAX_REQUESTS_PER_MINUTE
    reset_time: float = 0.0
