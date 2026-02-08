"""Wallet intelligence pipeline coordinator."""

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class WalletIntelligenceCoordinator:
    """Orchestrates the full wallet intelligence pipeline.

    Coordinates: discovery -> scoring -> reconstruction -> pattern analysis.
    """

    def __init__(self, wallet_provider, wallet_scorer, reconstructor,
                 pattern_analyzer, timescale=None, redis_store=None,
                 config: Optional[dict] = None):
        self.wallet_provider = wallet_provider
        self.wallet_scorer = wallet_scorer
        self.reconstructor = reconstructor
        self.pattern_analyzer = pattern_analyzer
        self.timescale = timescale
        self.redis_store = redis_store
        self.config = config or {}

        self.min_score = self.config.get("min_score", 60)
        self.max_wallets = self.config.get("max_wallets", 50)
        self.top_n_analyze = self.config.get("top_n_analyze", 10)
        self.lookback_days = self.config.get("lookback_days", 90)

    async def discover_and_score(self, limit: int = 50) -> list[dict]:
        """Full discovery pipeline: leaderboard -> trades -> score -> rank.

        Args:
            limit: Max wallets to fetch from leaderboard

        Returns:
            Ranked list of wallet scores
        """
        # Step 1: Get leaderboard
        leaderboard = self.wallet_provider.get_leaderboard(limit=limit)
        if not leaderboard:
            logger.info("No wallets found on leaderboard")
            return []

        logger.info("Discovered %d wallets from leaderboard", len(leaderboard))

        # Step 2: Fetch trades and score each wallet
        scores = []
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=self.lookback_days)

        for wallet in leaderboard:
            address = wallet.get("address", "")
            if not address:
                continue

            try:
                trades = self.wallet_provider.get_wallet_trades(
                    address, start=start, end=end)
                score = self.wallet_scorer.score_wallet(trades, address)
                scores.append(score)

                # Store score if storage available
                if self.timescale:
                    try:
                        self.timescale.insert_wallet_score(
                            address, score["total_score"],
                            score["components"],
                            datetime.now(timezone.utc),
                        )
                    except Exception:
                        logger.debug("Failed to store wallet score for %s", address[:10])

                if self.redis_store:
                    try:
                        self.redis_store.set_wallet_score(
                            address, score["total_score"],
                            score["grade"],
                            datetime.now(timezone.utc).isoformat(),
                        )
                    except Exception:
                        logger.debug("Failed to cache wallet score for %s", address[:10])

            except Exception:
                logger.debug("Failed to score wallet %s", address[:10])

        # Step 3: Rank
        ranked = self.wallet_scorer.rank_wallets(scores)

        # Store top wallets
        if self.redis_store and ranked:
            try:
                top_list = [
                    {"address": s["address"], "score": s["total_score"], "grade": s["grade"]}
                    for s in ranked[:self.max_wallets]
                ]
                self.redis_store.set_top_wallets(top_list)
            except Exception:
                logger.debug("Failed to store top wallets")

        logger.info("Scored %d wallets, top score: %.1f",
                    len(ranked), ranked[0]["total_score"] if ranked else 0)

        return ranked

    async def analyze_wallet(self, address: str,
                              lookback_days: Optional[int] = None) -> dict:
        """Deep analysis of a single wallet.

        Args:
            address: Wallet address to analyze
            lookback_days: How far back to look (default: self.lookback_days)

        Returns:
            Dict with positions, stats, patterns, and hypotheses
        """
        start_time = time.time()
        lookback = lookback_days or self.lookback_days

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback)

        # Fetch trades
        trades = self.wallet_provider.get_wallet_trades(address, start=start, end=end)
        if not trades:
            return {
                "address": address,
                "positions": [],
                "stats": {},
                "entry_patterns": {},
                "sizing_patterns": {},
                "exit_patterns": {},
                "hypotheses": [],
                "duration_seconds": time.time() - start_time,
            }

        # Reconstruct positions
        positions = self.reconstructor.reconstruct_positions(trades)
        stats = self.reconstructor.compute_trade_stats(positions)

        # Analyze patterns
        entry_patterns = self.pattern_analyzer.analyze_entry_patterns(positions)
        sizing_patterns = self.pattern_analyzer.analyze_sizing_patterns(positions)
        exit_patterns = self.pattern_analyzer.analyze_exit_patterns(positions)
        hypotheses = self.pattern_analyzer.generate_hypotheses(positions)

        duration = time.time() - start_time

        analysis = {
            "address": address,
            "positions": positions,
            "stats": stats,
            "entry_patterns": entry_patterns,
            "sizing_patterns": sizing_patterns,
            "exit_patterns": exit_patterns,
            "hypotheses": hypotheses,
            "duration_seconds": duration,
        }

        # Store analysis
        if self.timescale:
            try:
                # Don't store the full positions list (too large)
                store_analysis = {
                    "stats": stats,
                    "entry_patterns": entry_patterns,
                    "sizing_patterns": sizing_patterns,
                    "exit_patterns": exit_patterns,
                    "hypotheses": hypotheses,
                }
                self.timescale.insert_wallet_analysis(
                    address, store_analysis, datetime.now(timezone.utc))
            except Exception:
                logger.debug("Failed to store wallet analysis for %s", address[:10])

        return analysis

    async def run_discovery_cycle(self) -> dict:
        """Run a full discovery cycle: discover, score, analyze top N.

        Returns:
            Dict with discovered count, scored count, analyzed wallets, top score
        """
        start_time = time.time()

        # Discover and score
        ranked = await self.discover_and_score(limit=self.max_wallets)

        # Analyze top N recommended wallets
        analyzed = []
        recommended = [w for w in ranked if w.get("recommended", False)]
        to_analyze = recommended[:self.top_n_analyze]

        for wallet_score in to_analyze:
            address = wallet_score["address"]
            try:
                analysis = await self.analyze_wallet(address)
                analyzed.append({
                    "address": address,
                    "score": wallet_score["total_score"],
                    "grade": wallet_score["grade"],
                    "stats": analysis.get("stats", {}),
                    "hypotheses_count": len(analysis.get("hypotheses", [])),
                })
            except Exception:
                logger.debug("Failed to analyze wallet %s", address[:10])

        duration = time.time() - start_time

        # Update Prometheus metrics
        try:
            from monitoring.metrics import (
                wallet_discovery_count, wallet_top_score,
                wallet_analysis_duration_seconds,
            )
            wallet_discovery_count.set(len(ranked))
            if ranked:
                wallet_top_score.set(ranked[0]["total_score"])
            wallet_analysis_duration_seconds.observe(duration)
        except Exception:
            logger.debug("Failed to update wallet intelligence metrics")

        result = {
            "discovered": len(ranked),
            "recommended": len(recommended),
            "analyzed": len(analyzed),
            "top_score": ranked[0]["total_score"] if ranked else 0,
            "analyses": analyzed,
            "duration_seconds": duration,
        }

        logger.info(
            "Discovery cycle complete: %d discovered, %d recommended, %d analyzed in %.1fs",
            result["discovered"], result["recommended"],
            result["analyzed"], duration,
        )

        return result

    def get_recommended_for_monitoring(self, top_n: int = 20) -> list[dict]:
        """Return top-N wallets by score for real-time monitoring.

        Pulls from Redis cache first; falls back to in-memory.

        Args:
            top_n: Maximum wallets to return.

        Returns:
            List of dicts with address, score, grade.
        """
        # Try Redis first
        if self.redis_store:
            try:
                top_wallets = self.redis_store.get_top_wallets()
                if top_wallets:
                    recommended = [
                        w for w in top_wallets
                        if w.get("score", 0) >= self.min_score
                    ]
                    return recommended[:top_n]
            except Exception:
                logger.debug("Failed to get top wallets from Redis")

        return []
