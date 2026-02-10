"""Performance-based trader ranking system.

Ranks traders into tiers (top/bottom/middle) using wallet scoring,
then derives contrarian and smart money consensus signals from
their aggregate positioning.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TraderRankingSystem:
    """Ranks traders by performance and derives positioning signals."""

    def __init__(self, wallet_provider=None, wallet_scorer=None, config: dict = None):
        config = config or {}
        self.wallet_provider = wallet_provider
        self.wallet_scorer = wallet_scorer
        self.top_n = config.get("top_n", 100)
        self.bottom_n = config.get("bottom_n", 100)
        self.min_trades = config.get("min_trades", 10)

        self._rankings: dict = {}  # {address: {score, tier, ...}}
        self._top_traders: list[dict] = []
        self._bottom_traders: list[dict] = []

    def update_rankings(self, limit: int = 500) -> dict:
        """Fetch leaderboard, score wallets, split into tiers.

        Returns:
            Summary dict with total_scored, top_count, bottom_count,
            top_avg_score, bottom_avg_score.
        """
        if not self.wallet_provider or not self.wallet_scorer:
            return {
                "total_scored": 0,
                "top_count": 0,
                "bottom_count": 0,
                "top_avg_score": 0.0,
                "bottom_avg_score": 0.0,
            }

        try:
            leaderboard = self.wallet_provider.get_leaderboard(limit=limit)
        except Exception:
            logger.exception("Failed to fetch leaderboard")
            return {
                "total_scored": 0,
                "top_count": 0,
                "bottom_count": 0,
                "top_avg_score": 0.0,
                "bottom_avg_score": 0.0,
            }

        scored: list[dict] = []
        for entry in leaderboard:
            address = entry.get("address", "")
            if not address:
                continue
            try:
                trades = self.wallet_provider.get_wallet_trades(address)
            except Exception:
                continue

            if len(trades) < self.min_trades:
                continue

            score_result = self.wallet_scorer.score_wallet(trades, address)
            score_result["positions"] = []
            try:
                score_result["positions"] = self.wallet_provider.get_wallet_positions(address)
            except Exception:
                pass
            scored.append(score_result)

        # Sort by total_score descending
        scored.sort(key=lambda w: w.get("total_score", 0), reverse=True)

        # Split into tiers
        self._top_traders = scored[: self.top_n]
        self._bottom_traders = scored[-self.bottom_n :] if len(scored) > self.top_n else []

        # Build rankings map
        self._rankings = {}
        top_addrs = {t["address"] for t in self._top_traders}
        bottom_addrs = {t["address"] for t in self._bottom_traders}
        for entry in scored:
            addr = entry["address"]
            if addr in top_addrs:
                tier = "top"
            elif addr in bottom_addrs:
                tier = "bottom"
            else:
                tier = "middle"
            self._rankings[addr] = {
                "score": entry.get("total_score", 0),
                "tier": tier,
                "grade": entry.get("grade", "F"),
            }

        top_avg = (
            sum(t.get("total_score", 0) for t in self._top_traders) / len(self._top_traders)
            if self._top_traders
            else 0.0
        )
        bottom_avg = (
            sum(t.get("total_score", 0) for t in self._bottom_traders) / len(self._bottom_traders)
            if self._bottom_traders
            else 0.0
        )

        return {
            "total_scored": len(scored),
            "top_count": len(self._top_traders),
            "bottom_count": len(self._bottom_traders),
            "top_avg_score": round(top_avg, 2),
            "bottom_avg_score": round(bottom_avg, 2),
        }

    def get_tier(self, address: str) -> str:
        """Return tier for an address: top, bottom, middle, or unranked."""
        entry = self._rankings.get(address)
        if entry is None:
            return "unranked"
        return entry.get("tier", "unranked")

    def get_top_traders(self) -> list[dict]:
        return list(self._top_traders)

    def get_bottom_traders(self) -> list[dict]:
        return list(self._bottom_traders)

    def get_tier_positioning(self, tier: str) -> dict:
        """Aggregate positions of traders in a tier.

        Returns:
            {long_count, short_count, dominant_direction, agreement_pct}
        """
        if tier == "top":
            traders = self._top_traders
        elif tier == "bottom":
            traders = self._bottom_traders
        else:
            return {
                "long_count": 0,
                "short_count": 0,
                "dominant_direction": "neutral",
                "agreement_pct": 0.0,
            }

        long_count = 0
        short_count = 0

        for trader in traders:
            positions = trader.get("positions", [])
            for pos in positions:
                side = pos.get("side", "").lower()
                if side == "long":
                    long_count += 1
                elif side == "short":
                    short_count += 1

        total = long_count + short_count
        if total == 0:
            return {
                "long_count": 0,
                "short_count": 0,
                "dominant_direction": "neutral",
                "agreement_pct": 0.0,
            }

        dominant = "long" if long_count >= short_count else "short"
        agreement_pct = max(long_count, short_count) / total * 100

        return {
            "long_count": long_count,
            "short_count": short_count,
            "dominant_direction": dominant,
            "agreement_pct": round(agreement_pct, 2),
        }

    def get_contrarian_signal(self) -> dict:
        """Fade bottom traders: opposite of their dominant direction.

        Returns:
            {direction, confidence, bottom_bias, trader_count}
        """
        positioning = self.get_tier_positioning("bottom")
        trader_count = positioning["long_count"] + positioning["short_count"]

        if trader_count == 0:
            return {
                "direction": "neutral",
                "confidence": 0.0,
                "bottom_bias": "neutral",
                "trader_count": 0,
            }

        bottom_bias = positioning["dominant_direction"]
        # Fade: opposite of bottom traders
        if bottom_bias == "long":
            direction = "short"
        elif bottom_bias == "short":
            direction = "long"
        else:
            direction = "neutral"

        confidence = positioning["agreement_pct"] / 100.0

        return {
            "direction": direction,
            "confidence": round(confidence, 4),
            "bottom_bias": bottom_bias,
            "trader_count": trader_count,
        }

    def get_smart_money_consensus(self) -> dict:
        """Follow top traders: same as their dominant direction.

        Returns:
            {direction, confidence, top_bias, trader_count}
        """
        positioning = self.get_tier_positioning("top")
        trader_count = positioning["long_count"] + positioning["short_count"]

        if trader_count == 0:
            return {
                "direction": "neutral",
                "confidence": 0.0,
                "top_bias": "neutral",
                "trader_count": 0,
            }

        top_bias = positioning["dominant_direction"]
        direction = top_bias  # Follow top traders
        confidence = positioning["agreement_pct"] / 100.0

        return {
            "direction": direction,
            "confidence": round(confidence, 4),
            "top_bias": top_bias,
            "trader_count": trader_count,
        }
