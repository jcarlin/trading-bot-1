"""Hyperliquid LP (HLP) sentiment tracker.

HLP is typically the counterparty to retail flow on Hyperliquid.
When HLP is long, retail is implied short, and vice versa.
This module tracks HLP vault positions and derives contrarian signals.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class HLPSentimentTracker:
    """Tracks HLP vault positions and derives sentiment signals."""

    def __init__(self, wallet_provider=None, config: dict = None):
        config = config or {}
        self.wallet_provider = wallet_provider
        self.hlp_vault_address = config.get(
            "hlp_vault_address", "0xHLPVault"
        )
        self.history_max = config.get("history_max", 1000)

        self._current_positions: list[dict] = []
        self._sentiment_history: list[dict] = []

    def update(self) -> dict:
        """Poll HLP vault positions and record sentiment snapshot.

        Returns:
            {positions_count, net_notional, hlp_direction}
        """
        if not self.wallet_provider:
            return {
                "positions_count": 0,
                "net_notional": 0.0,
                "hlp_direction": "neutral",
            }

        try:
            positions = self.wallet_provider.get_wallet_positions(
                self.hlp_vault_address
            )
        except Exception:
            logger.exception("Failed to fetch HLP positions")
            return {
                "positions_count": 0,
                "net_notional": 0.0,
                "hlp_direction": "neutral",
            }

        self._current_positions = positions

        long_notional = 0.0
        short_notional = 0.0
        for pos in positions:
            side = pos.get("side", "").lower()
            size = abs(float(pos.get("size", 0)))
            price = float(pos.get("entry_price", 0))
            notional = size * price
            if side == "long":
                long_notional += notional
            elif side == "short":
                short_notional += notional

        net_notional = long_notional - short_notional

        if net_notional > 0:
            hlp_direction = "long"
        elif net_notional < 0:
            hlp_direction = "short"
        else:
            hlp_direction = "neutral"

        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "positions": len(positions),
            "net_notional": net_notional,
            "direction": hlp_direction,
        }
        self._sentiment_history.append(snapshot)

        # Trim history
        if len(self._sentiment_history) > self.history_max:
            self._sentiment_history = self._sentiment_history[-self.history_max :]

        return {
            "positions_count": len(positions),
            "net_notional": net_notional,
            "hlp_direction": hlp_direction,
        }

    def get_current_sentiment(self) -> dict:
        """Current HLP sentiment and implied retail direction.

        Returns:
            {hlp_direction, retail_implied_direction, net_notional, positions_count}
        """
        if not self._current_positions and not self._sentiment_history:
            return {
                "hlp_direction": "neutral",
                "retail_implied_direction": "neutral",
                "net_notional": 0.0,
                "positions_count": 0,
            }

        # Compute from current positions
        long_notional = 0.0
        short_notional = 0.0
        for pos in self._current_positions:
            side = pos.get("side", "").lower()
            size = abs(float(pos.get("size", 0)))
            price = float(pos.get("entry_price", 0))
            notional = size * price
            if side == "long":
                long_notional += notional
            elif side == "short":
                short_notional += notional

        net_notional = long_notional - short_notional

        if net_notional > 0:
            hlp_direction = "long"
            retail_implied = "short"
        elif net_notional < 0:
            hlp_direction = "short"
            retail_implied = "long"
        else:
            hlp_direction = "neutral"
            retail_implied = "neutral"

        return {
            "hlp_direction": hlp_direction,
            "retail_implied_direction": retail_implied,
            "net_notional": net_notional,
            "positions_count": len(self._current_positions),
        }

    def get_sentiment_signal(self, symbol: str = None) -> dict:
        """Contrarian signal: fade HLP (follow implied retail).

        Args:
            symbol: Optional symbol filter for per-asset signals.

        Returns:
            {direction, confidence, hlp_direction, rationale}
        """
        if symbol:
            # Per-symbol filtering
            long_notional = 0.0
            short_notional = 0.0
            for pos in self._current_positions:
                if pos.get("symbol", "") != symbol:
                    continue
                side = pos.get("side", "").lower()
                size = abs(float(pos.get("size", 0)))
                price = float(pos.get("entry_price", 0))
                notional = size * price
                if side == "long":
                    long_notional += notional
                elif side == "short":
                    short_notional += notional

            net = long_notional - short_notional
            total = long_notional + short_notional
        else:
            sentiment = self.get_current_sentiment()
            net = sentiment["net_notional"]
            total = abs(net)  # Approximate
            # Recompute total from positions for better confidence
            total = 0.0
            for pos in self._current_positions:
                size = abs(float(pos.get("size", 0)))
                price = float(pos.get("entry_price", 0))
                total += size * price

        if total == 0:
            return {
                "direction": "neutral",
                "confidence": 0.0,
                "hlp_direction": "neutral",
                "rationale": "No HLP positions",
            }

        if net > 0:
            hlp_direction = "long"
            direction = "short"  # Fade HLP
            rationale = "HLP is long, implying retail is short; fading HLP"
        elif net < 0:
            hlp_direction = "short"
            direction = "long"  # Fade HLP
            rationale = "HLP is short, implying retail is long; fading HLP"
        else:
            return {
                "direction": "neutral",
                "confidence": 0.0,
                "hlp_direction": "neutral",
                "rationale": "HLP is neutral",
            }

        confidence = min(1.0, abs(net) / total) if total > 0 else 0.0

        return {
            "direction": direction,
            "confidence": round(confidence, 4),
            "hlp_direction": hlp_direction,
            "rationale": rationale,
        }

    def get_sentiment_history(self, window_hours: int = 24) -> list[dict]:
        """Return sentiment history entries within window.

        Args:
            window_hours: Number of hours to look back.

        Returns:
            List of sentiment snapshot dicts.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        result = []
        for entry in self._sentiment_history:
            try:
                ts = datetime.fromisoformat(entry["timestamp"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts >= cutoff:
                    result.append(entry)
            except (ValueError, TypeError, KeyError):
                continue
        return result

    def get_sentiment_change(self, window_hours: int = 24) -> dict:
        """Detect if HLP direction has shifted within window.

        Returns:
            {is_shifting, from_direction, to_direction, magnitude}
        """
        history = self.get_sentiment_history(window_hours)

        if len(history) < 2:
            return {
                "is_shifting": False,
                "from_direction": "unknown",
                "to_direction": "unknown",
                "magnitude": 0.0,
            }

        first = history[0]
        last = history[-1]

        from_dir = first.get("direction", "neutral")
        to_dir = last.get("direction", "neutral")

        first_net = first.get("net_notional", 0.0)
        last_net = last.get("net_notional", 0.0)
        magnitude = abs(last_net - first_net)

        is_shifting = from_dir != to_dir

        return {
            "is_shifting": is_shifting,
            "from_direction": from_dir,
            "to_direction": to_dir,
            "magnitude": magnitude,
        }
