"""Cross-venue data aggregation for multi-exchange trading.

Provides a unified interface to aggregate market data (liquidations,
funding rates, trades) from multiple exchanges behind a common ABC.
"""

from abc import ABC, abstractmethod
from typing import Optional


class ExchangeDataSource(ABC):
    """Abstract interface for exchange data feeds."""

    @abstractmethod
    def get_liquidations(self, symbol: str, start=None, end=None,
                         limit: int = 100) -> list[dict]:
        """Get recent liquidation events.

        Returns list of {timestamp, symbol, side, size, price, exchange}.
        """
        ...

    @abstractmethod
    def get_funding_rate(self, symbol: str) -> Optional[dict]:
        """Get current funding rate.

        Returns {symbol, rate, timestamp, exchange} or None.
        """
        ...

    @abstractmethod
    def get_recent_trades(self, symbol: str, limit: int = 100) -> list[dict]:
        """Get recent trades.

        Returns list of {timestamp, price, size, side, exchange}.
        """
        ...

    @abstractmethod
    def get_exchange_name(self) -> str:
        """Return the exchange identifier."""
        ...


class HyperliquidDataSource(ExchangeDataSource):
    """Hyperliquid exchange data source (placeholder)."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def get_liquidations(self, symbol: str, start=None, end=None,
                         limit: int = 100) -> list[dict]:
        return []

    def get_funding_rate(self, symbol: str) -> Optional[dict]:
        return None

    def get_recent_trades(self, symbol: str, limit: int = 100) -> list[dict]:
        return []

    def get_exchange_name(self) -> str:
        return "hyperliquid"


class BinanceDataSource(ExchangeDataSource):
    """Binance exchange data source (placeholder)."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def get_liquidations(self, symbol: str, start=None, end=None,
                         limit: int = 100) -> list[dict]:
        return []

    def get_funding_rate(self, symbol: str) -> Optional[dict]:
        return None

    def get_recent_trades(self, symbol: str, limit: int = 100) -> list[dict]:
        return []

    def get_exchange_name(self) -> str:
        return "binance"


class BybitDataSource(ExchangeDataSource):
    """Bybit exchange data source (placeholder)."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def get_liquidations(self, symbol: str, start=None, end=None,
                         limit: int = 100) -> list[dict]:
        return []

    def get_funding_rate(self, symbol: str) -> Optional[dict]:
        return None

    def get_recent_trades(self, symbol: str, limit: int = 100) -> list[dict]:
        return []

    def get_exchange_name(self) -> str:
        return "bybit"


class MockExchangeDataSource(ExchangeDataSource):
    """Mock exchange data source for testing."""

    def __init__(self, name: str = "mock", liquidations=None, funding=None,
                 trades=None):
        self.name = name
        self._liquidations = liquidations or []
        self._funding = funding
        self._trades = trades or []

    def get_liquidations(self, symbol: str, start=None, end=None,
                         limit: int = 100) -> list[dict]:
        return self._liquidations[:limit]

    def get_funding_rate(self, symbol: str) -> Optional[dict]:
        return self._funding

    def get_recent_trades(self, symbol: str, limit: int = 100) -> list[dict]:
        return self._trades[:limit]

    def get_exchange_name(self) -> str:
        return self.name


class MultiExchangeAggregator:
    """Aggregates market data from multiple exchange sources."""

    def __init__(self, sources: list[ExchangeDataSource],
                 config: dict = None):
        self.sources = sources
        self.config = config or {}

    def get_all_liquidations(self, symbol: str,
                             limit_per_exchange: int = 100) -> list[dict]:
        """Aggregate liquidations from all sources, sorted by timestamp desc."""
        all_liqs = []
        for source in self.sources:
            try:
                liqs = source.get_liquidations(symbol, limit=limit_per_exchange)
                for liq in liqs:
                    liq.setdefault("exchange", source.get_exchange_name())
                all_liqs.extend(liqs)
            except Exception:
                continue

        all_liqs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return all_liqs

    def get_all_funding_rates(self, symbol: str) -> list[dict]:
        """Collect funding rate from each source."""
        rates = []
        for source in self.sources:
            try:
                rate = source.get_funding_rate(symbol)
                if rate is not None:
                    rate.setdefault("exchange", source.get_exchange_name())
                    rates.append(rate)
            except Exception:
                continue
        return rates

    def get_funding_spread(self, symbol: str) -> dict:
        """Compute funding rate spread across exchanges.

        Returns {max_rate, min_rate, spread, exchanges: {name: rate}}.
        """
        rates = self.get_all_funding_rates(symbol)
        if not rates:
            return {
                "max_rate": 0.0,
                "min_rate": 0.0,
                "spread": 0.0,
                "exchanges": {},
            }

        exchanges = {}
        for r in rates:
            name = r.get("exchange", "unknown")
            exchanges[name] = r.get("rate", 0.0)

        rate_values = list(exchanges.values())
        max_rate = max(rate_values)
        min_rate = min(rate_values)

        return {
            "max_rate": max_rate,
            "min_rate": min_rate,
            "spread": max_rate - min_rate,
            "exchanges": exchanges,
        }
