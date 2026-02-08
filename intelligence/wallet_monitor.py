"""Real-time wallet position monitor.

Polls tracked wallets at configurable intervals, detects position changes,
and emits WalletSignal objects for consumption by strategies.
"""

import asyncio
import logging
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from intelligence.wallet_signal import WalletSignal

logger = logging.getLogger(__name__)

# Confidence multipliers by signal type
SIGNAL_CONFIDENCE = {
    "new_position": 0.9,
    "closed": 0.8,
    "size_increase": 0.7,
    "size_decrease": 0.6,
}


class WalletMonitor:
    """Real-time wallet position monitor.

    Polls tracked wallets and detects position changes vs last known state.

    Config:
        poll_interval_s: Seconds between polls (default 30).
        max_tracked_wallets: Maximum wallets to track (default 20).
        min_wallet_score: Minimum wallet score to accept (default 70).
        size_change_threshold_pct: Minimum % change to emit signal (default 25).
        unusual_size_std: Std devs above mean for unusual flag (default 2.0).
    """

    def __init__(self, wallet_provider, redis_store, timescale, config: dict,
                 notification_dispatcher=None):
        self.wallet_provider = wallet_provider
        self.redis_store = redis_store
        self.timescale = timescale
        self.config = config or {}
        self.notification_dispatcher = notification_dispatcher

        self.poll_interval_s = self.config.get("poll_interval_s", 30)
        self.max_tracked_wallets = self.config.get("max_tracked_wallets", 20)
        self.min_wallet_score = self.config.get("min_wallet_score", 70)
        self.size_change_threshold_pct = self.config.get("size_change_threshold_pct", 25)
        self.unusual_size_std = self.config.get("unusual_size_std", 2.0)

        # Tracked wallets: {address: {"score": float, "added_at": str}}
        self._tracked_wallets: dict[str, dict] = {}

        # Size history per wallet/symbol for unusual detection
        self._size_history: dict[str, list[float]] = defaultdict(list)

        # Recent signals buffer
        self._recent_signals: list[dict] = []
        self._max_recent_signals = 200

    async def run(self, stop_event: asyncio.Event) -> None:
        """Main loop: polls all tracked wallets each interval.

        Args:
            stop_event: Set this event to stop the monitor.
        """
        logger.info("WalletMonitor starting with %d tracked wallets, poll=%ds",
                     len(self._tracked_wallets), self.poll_interval_s)

        while not stop_event.is_set():
            try:
                addresses = list(self._tracked_wallets.keys())
                for address in addresses:
                    if stop_event.is_set():
                        break
                    signals = await self._poll_wallet(address)
                    for sig in signals:
                        self._store_signal(sig)

                        # Update Prometheus
                        try:
                            from monitoring.metrics import wallet_signal_detected_total
                            wallet_signal_detected_total.labels(
                                signal_type=sig.signal_type).inc()
                        except Exception:
                            pass

                        # Notify if unusual
                        if sig.is_unusual and self.notification_dispatcher:
                            try:
                                await self.notification_dispatcher.dispatch(
                                    message=(f"Unusual wallet signal: {sig.wallet_address[:10]}... "
                                             f"{sig.signal_type} {sig.direction} {sig.symbol} "
                                             f"size={sig.size:.4f} ({sig.size_change_pct:+.1f}%)"),
                                    level="warning",
                                    event_type="wallet_signal",
                                )
                            except Exception:
                                logger.debug("Failed to dispatch wallet signal notification")

                # Update active count metric
                try:
                    from monitoring.metrics import wallet_monitor_active_count
                    wallet_monitor_active_count.set(len(self._tracked_wallets))
                except Exception:
                    pass

            except Exception:
                logger.exception("WalletMonitor poll cycle failed")

            # Wait for next poll or stop
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.poll_interval_s)
                break  # stop_event was set
            except asyncio.TimeoutError:
                pass  # Normal timeout, continue polling

        logger.info("WalletMonitor stopped")

    async def _poll_wallet(self, address: str) -> list[WalletSignal]:
        """Poll a wallet and detect position changes vs last known state.

        Args:
            address: Wallet address to poll.

        Returns:
            List of WalletSignal objects for detected changes.
        """
        try:
            current_positions = self.wallet_provider.get_wallet_positions(address)
        except Exception:
            logger.debug("Failed to fetch positions for %s", address[:10])
            return []

        # Get previous positions from Redis
        previous_positions = []
        if self.redis_store:
            try:
                previous_positions = self.redis_store.get_wallet_positions(address)
            except Exception:
                pass

        # Detect changes
        signals = self._detect_changes(address, current_positions, previous_positions)

        # Store current positions for next poll
        if self.redis_store:
            try:
                self.redis_store.set_wallet_positions(address, current_positions)
            except Exception:
                logger.debug("Failed to store wallet positions for %s", address[:10])

        return signals

    def _detect_changes(self, address: str, current: list[dict],
                        previous: list[dict]) -> list[WalletSignal]:
        """Diff current vs previous positions to detect changes.

        Args:
            address: Wallet address.
            current: Current positions list.
            previous: Previous positions list.

        Returns:
            List of WalletSignal for each detected change.
        """
        signals = []
        now = datetime.now(timezone.utc)
        wallet_info = self._tracked_wallets.get(address, {})
        wallet_score = wallet_info.get("score", 0.0)

        # Build lookup maps by symbol
        current_map = {p["symbol"]: p for p in current if "symbol" in p}
        previous_map = {p["symbol"]: p for p in previous if "symbol" in p}

        # Detect new positions and size changes
        for symbol, pos in current_map.items():
            size = abs(float(pos.get("size", 0)))
            direction = pos.get("side", "long")

            if symbol not in previous_map:
                # New position opened
                is_unusual = self._is_unusual_size(address, size, symbol)
                confidence = self._compute_confidence(
                    wallet_score, "new_position", is_unusual)
                signals.append(WalletSignal(
                    timestamp=now,
                    wallet_address=address,
                    wallet_score=wallet_score,
                    symbol=symbol,
                    signal_type="new_position",
                    direction=direction,
                    size=size,
                    size_change_pct=100.0,
                    is_unusual=is_unusual,
                    confidence=confidence,
                ))
                self._record_size(address, symbol, size)
            else:
                # Existing position — check for size change
                prev_size = abs(float(previous_map[symbol].get("size", 0)))
                if prev_size == 0:
                    continue

                change_pct = ((size - prev_size) / prev_size) * 100

                if abs(change_pct) >= self.size_change_threshold_pct:
                    signal_type = "size_increase" if change_pct > 0 else "size_decrease"
                    is_unusual = self._is_unusual_size(address, size, symbol)
                    confidence = self._compute_confidence(
                        wallet_score, signal_type, is_unusual)
                    signals.append(WalletSignal(
                        timestamp=now,
                        wallet_address=address,
                        wallet_score=wallet_score,
                        symbol=symbol,
                        signal_type=signal_type,
                        direction=direction,
                        size=size,
                        size_change_pct=change_pct,
                        is_unusual=is_unusual,
                        confidence=confidence,
                    ))
                    self._record_size(address, symbol, size)

        # Detect closed positions
        for symbol, pos in previous_map.items():
            if symbol not in current_map:
                prev_size = abs(float(pos.get("size", 0)))
                direction = pos.get("side", "long")
                signals.append(WalletSignal(
                    timestamp=now,
                    wallet_address=address,
                    wallet_score=wallet_score,
                    symbol=symbol,
                    signal_type="closed",
                    direction=direction,
                    size=0.0,
                    size_change_pct=-100.0,
                    is_unusual=False,
                    confidence=self._compute_confidence(wallet_score, "closed", False),
                ))

        return signals

    def _is_unusual_size(self, address: str, size: float, symbol: str) -> bool:
        """Check if position size exceeds mean + unusual_size_std * std for this wallet.

        Args:
            address: Wallet address.
            size: Current position size.
            symbol: Trading pair.

        Returns:
            True if the size is statistically unusual.
        """
        key = f"{address}:{symbol}"
        history = self._size_history.get(key, [])

        if len(history) < 3:
            return False

        mean = sum(history) / len(history)
        variance = sum((s - mean) ** 2 for s in history) / len(history)
        std = math.sqrt(variance) if variance > 0 else 0.0

        if std == 0:
            return False

        return size > mean + self.unusual_size_std * std

    def _record_size(self, address: str, symbol: str, size: float) -> None:
        """Record a size observation for unusual detection."""
        key = f"{address}:{symbol}"
        self._size_history[key].append(size)
        # Keep last 50 observations
        if len(self._size_history[key]) > 50:
            self._size_history[key] = self._size_history[key][-50:]

    def _compute_confidence(self, wallet_score: float, signal_type: str,
                            is_unusual: bool) -> float:
        """Compute signal confidence from wallet score, type, and unusualness."""
        base = SIGNAL_CONFIDENCE.get(signal_type, 0.5)
        # Scale by wallet score (0-100 -> 0.0-1.0)
        score_factor = wallet_score / 100.0
        confidence = base * score_factor
        if is_unusual:
            confidence = min(1.0, confidence + 0.2)
        return round(min(1.0, max(0.0, confidence)), 3)

    def _store_signal(self, signal: WalletSignal) -> None:
        """Store signal in Redis and in-memory buffer."""
        sig_dict = signal.to_dict()
        self._recent_signals.append(sig_dict)
        if len(self._recent_signals) > self._max_recent_signals:
            self._recent_signals = self._recent_signals[-self._max_recent_signals:]

        if self.redis_store:
            try:
                self.redis_store.add_wallet_signal(sig_dict)
            except Exception:
                logger.debug("Failed to store wallet signal in Redis")

    def add_wallet(self, address: str, score: float) -> None:
        """Add a wallet to the tracking list.

        Args:
            address: Wallet address.
            score: Wallet score (0-100).
        """
        if score < self.min_wallet_score:
            logger.debug("Wallet %s score %.1f below threshold %.1f, not added",
                         address[:10], score, self.min_wallet_score)
            return

        if len(self._tracked_wallets) >= self.max_tracked_wallets:
            logger.debug("Max tracked wallets (%d) reached, not adding %s",
                         self.max_tracked_wallets, address[:10])
            return

        self._tracked_wallets[address] = {
            "score": score,
            "added_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.info("Added wallet %s (score=%.1f) to monitoring", address[:10], score)

    def remove_wallet(self, address: str) -> None:
        """Remove a wallet from the tracking list."""
        self._tracked_wallets.pop(address, None)
        logger.info("Removed wallet %s from monitoring", address[:10])

    def get_tracked_wallets(self) -> list[dict]:
        """Return list of tracked wallets with their metadata."""
        result = []
        for addr, info in self._tracked_wallets.items():
            result.append({
                "address": addr,
                "score": info["score"],
                "added_at": info["added_at"],
            })
        return result

    def get_recent_signals(self, limit: int = 50) -> list[dict]:
        """Return recent wallet signals."""
        return self._recent_signals[-limit:]
