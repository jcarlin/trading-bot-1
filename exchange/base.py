"""Abstract base class for exchange adapters."""

from abc import ABC, abstractmethod
from typing import Optional

from core.models import Fill, Order, Position


class BaseExchange(ABC):
    """Interface that all exchange adapters must implement.

    Both CcxtExchange and HyperliquidClient conform to this interface,
    ensuring the execution engine can work with any exchange backend.
    """

    @abstractmethod
    def place_order(self, order: Order) -> Fill:
        """Submit an order and return the resulting fill."""

    @abstractmethod
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order. Returns True if cancellation succeeded."""

    @abstractmethod
    def get_open_orders(self, symbol: str) -> list[dict]:
        """Return open orders for the given symbol."""

    @abstractmethod
    def get_balance(self, currency: str = "USDT") -> float:
        """Return the available balance for the given currency."""

    @abstractmethod
    def get_ticker(self, symbol: str) -> dict:
        """Fetch current ticker with at least 'last', 'bid', 'ask' keys."""

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Return all open positions on the exchange."""

    @abstractmethod
    def get_account_state(self) -> dict:
        """Return account state including equity, margin, and balances."""
