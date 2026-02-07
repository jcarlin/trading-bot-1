"""Data provider for fetching OHLCV market data via ccxt or CSV."""

import logging
import os
import time
from datetime import datetime

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)


class DataProvider:
    """Fetches and normalizes OHLCV data from exchanges or local files.

    Supports live exchange data via ccxt and offline data via CSV.
    """

    def __init__(
        self,
        exchange_id: str,
        sandbox: bool = True,
        api_key: str = "",
        api_secret: str = "",
    ):
        api_key = api_key or os.environ.get("EXCHANGE_API_KEY", "")
        api_secret = api_secret or os.environ.get("EXCHANGE_API_SECRET", "")

        exchange_class = getattr(ccxt, exchange_id)
        self.exchange: ccxt.Exchange = exchange_class(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
            }
        )

        if sandbox:
            self.exchange.set_sandbox_mode(True)

        logger.info(
            "DataProvider initialised: exchange=%s sandbox=%s",
            exchange_id,
            sandbox,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: str | datetime | None = None,
        until: str | datetime | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars from the exchange, paginating as needed.

        Args:
            symbol: Trading pair, e.g. "BTC/USDT".
            timeframe: Bar size, e.g. "1h", "1d".
            since: Start time (ISO-8601 string or datetime). None = exchange default.
            until: End time (ISO-8601 string or datetime). None = up to now.
            limit: Max bars per request (exchange-dependent cap).

        Returns:
            DataFrame with columns [open, high, low, close, volume] and
            a tz-aware UTC DatetimeIndex named "timestamp".
        """
        since_ms = self._to_ms(since) if since else None
        until_ms = self._to_ms(until) if until else None

        all_candles: list[list] = []
        fetch_since = since_ms

        while True:
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol, timeframe, since=fetch_since, limit=limit
                )
            except ccxt.BaseError as exc:
                logger.error("ccxt error fetching OHLCV: %s", exc)
                raise

            if not candles:
                break

            # Filter out candles past the `until` boundary
            if until_ms is not None:
                candles = [c for c in candles if c[0] <= until_ms]

            all_candles.extend(candles)

            # Stop when we received fewer candles than the limit (last page)
            if len(candles) < limit:
                break

            # Advance the cursor past the last candle timestamp
            fetch_since = candles[-1][0] + 1

            if until_ms is not None and fetch_since > until_ms:
                break

            # Respect rate limits
            time.sleep(self.exchange.rateLimit / 1000)

        if not all_candles:
            logger.warning("No OHLCV data returned for %s %s", symbol, timeframe)
            return self._empty_df()

        df = self._candles_to_df(all_candles)
        logger.info(
            "Fetched %d bars for %s %s (%s -> %s)",
            len(df),
            symbol,
            timeframe,
            df.index[0],
            df.index[-1],
        )
        return df

    @staticmethod
    def load_csv(filepath: str) -> pd.DataFrame:
        """Load OHLCV data from a CSV file.

        The CSV must contain columns for timestamp (or date), open, high,
        low, close, and volume. The timestamp column is used as the index.

        Returns:
            DataFrame matching the standard OHLCV format.
        """
        df = pd.read_csv(filepath)

        # Normalise column names to lowercase
        df.columns = [c.strip().lower() for c in df.columns]

        # Detect the timestamp column
        ts_col = None
        for candidate in ("timestamp", "date", "datetime", "time"):
            if candidate in df.columns:
                ts_col = candidate
                break

        if ts_col is None:
            raise ValueError(
                f"CSV must contain a timestamp/date column. Found: {list(df.columns)}"
            )

        df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
        df = df.set_index(ts_col)
        df.index.name = "timestamp"

        expected = ["open", "high", "low", "close", "volume"]
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        df = df[expected].astype(float)
        df = df.sort_index()

        logger.info("Loaded %d bars from %s", len(df), filepath)
        return df

    def get_latest_bars(
        self, symbol: str, timeframe: str, count: int = 100
    ) -> pd.DataFrame:
        """Fetch the N most recent completed bars (for paper trading).

        Args:
            symbol: Trading pair, e.g. "BTC/USDT".
            timeframe: Bar size, e.g. "1h".
            count: Number of bars to retrieve.

        Returns:
            DataFrame with the latest *count* bars.
        """
        try:
            candles = self.exchange.fetch_ohlcv(symbol, timeframe, limit=count)
        except ccxt.BaseError as exc:
            logger.error("ccxt error fetching latest bars: %s", exc)
            raise

        if not candles:
            return self._empty_df()

        return self._candles_to_df(candles)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_ms(value: str | datetime) -> int:
        """Convert a string or datetime to millisecond Unix timestamp."""
        if isinstance(value, str):
            value = datetime.fromisoformat(value)
        return int(value.timestamp() * 1000)

    @staticmethod
    def _candles_to_df(candles: list[list]) -> pd.DataFrame:
        """Convert raw ccxt candle arrays to a standard DataFrame."""
        df = pd.DataFrame(
            candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        df = df.drop_duplicates()
        df = df.sort_index()
        return df

    @staticmethod
    def _empty_df() -> pd.DataFrame:
        """Return an empty DataFrame with the standard OHLCV schema."""
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df.index.name = "timestamp"
        return df
