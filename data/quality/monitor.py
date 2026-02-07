"""Data quality monitoring: staleness, sequence gaps, and price anomalies."""

import logging
import time
from collections import deque
from datetime import datetime
from typing import Optional

from monitoring.metrics import data_quality_status

logger = logging.getLogger(__name__)


class DataQualityMonitor:
    """Monitors data feed health across multiple sources and symbols.

    Detects three classes of issues:
    - **Staleness**: a source hasn't been updated within its expected interval.
    - **Sequence gaps**: missing sequence numbers in an ordered feed.
    - **Price anomalies**: price moves that exceed *anomaly_sigma* standard
      deviations from the rolling mean.

    Quality flags are written to Redis with a TTL so they auto-expire if the
    monitor itself stops running.
    """

    def __init__(
        self,
        redis_store,
        staleness_threshold_s: float = 30.0,
        anomaly_sigma: float = 3.0,
        price_window: int = 100,
    ):
        self._redis = redis_store
        self._staleness_threshold_s = staleness_threshold_s
        self._anomaly_sigma = anomaly_sigma
        self._price_window = price_window

        # Per-source tracking  --  key = (source, symbol)
        self._sources: dict[tuple[str, str], dict] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_source(
        self,
        name: str,
        symbol: str,
        expected_interval_s: float,
    ) -> None:
        """Register a data source for monitoring."""
        key = (name, symbol)
        self._sources[key] = {
            "expected_interval_s": expected_interval_s,
            "last_update_ts": None,
            "last_seq_num": None,
            "gaps_detected": 0,
            "prices": deque(maxlen=self._price_window),
            "anomalies_detected": 0,
            "status": "ok",
        }
        data_quality_status.labels(source=name, symbol=symbol).set(1)
        logger.info(
            "Registered quality source %s:%s (interval=%.1fs)",
            name, symbol, expected_interval_s,
        )

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def record_update(
        self,
        name: str,
        symbol: str,
        exchange_ts: datetime,
        receipt_ts: datetime,
        seq_num: Optional[int] = None,
        price: Optional[float] = None,
    ) -> None:
        """Record an incoming data update and run quality checks."""
        key = (name, symbol)
        state = self._sources.get(key)
        if state is None:
            return

        now = time.time()
        state["last_update_ts"] = now

        # --- sequence gap detection ---
        if seq_num is not None and state["last_seq_num"] is not None:
            expected = state["last_seq_num"] + 1
            if seq_num != expected:
                state["gaps_detected"] += 1
                logger.warning(
                    "Sequence gap on %s:%s: expected %d got %d",
                    name, symbol, expected, seq_num,
                )
        if seq_num is not None:
            state["last_seq_num"] = seq_num

        # --- price anomaly detection ---
        if price is not None:
            prices = state["prices"]
            if len(prices) >= 2:
                mean = sum(prices) / len(prices)
                variance = sum((p - mean) ** 2 for p in prices) / len(prices)
                std = variance ** 0.5
                if std > 0 and abs(price - mean) > self._anomaly_sigma * std:
                    state["anomalies_detected"] += 1
                    logger.warning(
                        "Price anomaly on %s:%s: price=%.4f mean=%.4f std=%.4f",
                        name, symbol, price, mean, std,
                    )
            prices.append(price)

        # --- compute status ---
        status = self._compute_status(key)
        state["status"] = status

        # --- update Redis + Prometheus ---
        latency_ms = (receipt_ts.timestamp() - exchange_ts.timestamp()) * 1000
        gauge_val = 1 if status == "ok" else 0
        data_quality_status.labels(source=name, symbol=symbol).set(gauge_val)

        self._redis.set_data_quality(
            source=name,
            symbol=symbol,
            last_update=receipt_ts.isoformat(),
            latency_ms=latency_ms,
            status=status,
            ttl=300,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_status(self, source: str, symbol: str) -> dict:
        """Return the current quality status for a single source."""
        key = (source, symbol)
        state = self._sources.get(key)
        if state is None:
            return {"status": "unknown", "source": source, "symbol": symbol}

        # Re-check staleness at query time
        status = self._compute_status(key)
        state["status"] = status

        return {
            "source": source,
            "symbol": symbol,
            "status": status,
            "last_update_ts": state["last_update_ts"],
            "gaps_detected": state["gaps_detected"],
            "anomalies_detected": state["anomalies_detected"],
        }

    def get_all_status(self) -> dict:
        """Return quality status for every registered source."""
        result: dict[str, dict] = {}
        for source, symbol in self._sources:
            result[f"{source}:{symbol}"] = self.get_status(source, symbol)
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_status(self, key: tuple[str, str]) -> str:
        """Derive a single status string from tracked state."""
        state = self._sources[key]

        # staleness
        if state["last_update_ts"] is not None:
            age = time.time() - state["last_update_ts"]
            if age > self._staleness_threshold_s:
                return "stale"

        # gaps
        if state["gaps_detected"] > 0:
            return "gap"

        # anomalies
        if state["anomalies_detected"] > 0:
            return "anomaly"

        # never updated yet
        if state["last_update_ts"] is None:
            return "pending"

        return "ok"
