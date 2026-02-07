"""Wallet data providers for intelligence pipeline."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class WalletDataProvider(ABC):
    """Abstract base for wallet data sources."""

    @abstractmethod
    def get_leaderboard(self, limit: int = 100) -> list[dict]:
        """Get top wallets by PnL.

        Returns list of dicts with keys: address, pnl, roi, trade_count
        """

    @abstractmethod
    def get_wallet_trades(self, address: str, start: Optional[datetime] = None,
                          end: Optional[datetime] = None) -> list[dict]:
        """Get trade history for a wallet.

        Returns list of dicts with keys: timestamp, symbol, side, size, price, pnl
        """

    @abstractmethod
    def get_wallet_positions(self, address: str) -> list[dict]:
        """Get current open positions for a wallet.

        Returns list of dicts with keys: symbol, side, size, entry_price, unrealized_pnl
        """


class HyperliquidWalletProvider(WalletDataProvider):
    """Wallet data from Hyperliquid exchange API."""

    def __init__(self, config: dict):
        self.config = config
        # Would use hyperliquid-python-sdk Info client
        self._base_url = config.get("base_url", "https://api.hyperliquid.xyz")

    def get_leaderboard(self, limit: int = 100) -> list[dict]:
        """Fetch leaderboard from Hyperliquid API."""
        try:
            # In production, this would call the Hyperliquid info API
            # For now, return empty — requires API integration
            logger.info("Fetching leaderboard (limit=%d)", limit)
            return []
        except Exception:
            logger.exception("Failed to fetch leaderboard")
            return []

    def get_wallet_trades(self, address: str, start: Optional[datetime] = None,
                          end: Optional[datetime] = None) -> list[dict]:
        """Fetch trade history for a wallet from Hyperliquid."""
        try:
            logger.info("Fetching trades for %s", address[:10])
            return []
        except Exception:
            logger.exception("Failed to fetch wallet trades")
            return []

    def get_wallet_positions(self, address: str) -> list[dict]:
        """Fetch open positions for a wallet from Hyperliquid."""
        try:
            logger.info("Fetching positions for %s", address[:10])
            return []
        except Exception:
            logger.exception("Failed to fetch wallet positions")
            return []


class MockWalletProvider(WalletDataProvider):
    """Mock provider for testing with canned data."""

    def __init__(self, leaderboard: Optional[list[dict]] = None,
                 trades: Optional[dict[str, list[dict]]] = None,
                 positions: Optional[dict[str, list[dict]]] = None):
        self._leaderboard = leaderboard or []
        self._trades = trades or {}
        self._positions = positions or {}

    def get_leaderboard(self, limit: int = 100) -> list[dict]:
        return self._leaderboard[:limit]

    def get_wallet_trades(self, address: str, start: Optional[datetime] = None,
                          end: Optional[datetime] = None) -> list[dict]:
        trades = self._trades.get(address, [])
        if start:
            trades = [t for t in trades if t.get("timestamp", datetime.min) >= start]
        if end:
            trades = [t for t in trades if t.get("timestamp", datetime.max) <= end]
        return trades

    def get_wallet_positions(self, address: str) -> list[dict]:
        return self._positions.get(address, [])
