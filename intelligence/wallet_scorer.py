"""Wallet scoring system for identifying profitable traders."""

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

# Default component weights
DEFAULT_WEIGHTS = {
    "risk_adjusted_return": 0.30,
    "consistency": 0.25,
    "drawdown": 0.20,
    "trade_frequency": 0.15,
    "diversity": 0.10,
}

# Grade thresholds
GRADE_THRESHOLDS = [
    (90, "A"),
    (80, "B"),
    (70, "C"),
    (60, "D"),
    (0, "F"),
]


class WalletScorer:
    """Scores wallets on risk-adjusted performance, consistency, and patterns."""

    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        self.weights = config.get("weights", DEFAULT_WEIGHTS)
        self.min_trades = config.get("min_trades", 10)
        self.recommend_threshold = config.get("recommend_threshold", 70)

    def score_wallet(self, trades: list[dict], address: str) -> dict:
        """Score a wallet based on its trade history.

        Args:
            trades: List of trade dicts with keys: pnl, symbol, size, timestamp
            address: Wallet address

        Returns:
            Dict with total_score, components, grade, recommended
        """
        if not trades or len(trades) < self.min_trades:
            return {
                "address": address,
                "total_score": 0.0,
                "components": {k: 0.0 for k in self.weights},
                "grade": "F",
                "recommended": False,
                "reason": f"Insufficient trades ({len(trades) if trades else 0} < {self.min_trades})",
            }

        components = {
            "risk_adjusted_return": self._score_risk_adjusted(trades),
            "consistency": self._score_consistency(trades),
            "drawdown": self._score_drawdown(trades),
            "trade_frequency": self._score_frequency(trades),
            "diversity": self._score_diversity(trades),
        }

        total_score = sum(
            components[k] * self.weights.get(k, 0)
            for k in components
        )
        total_score = max(0.0, min(100.0, total_score))

        grade = self._compute_grade(total_score)

        return {
            "address": address,
            "total_score": total_score,
            "components": components,
            "grade": grade,
            "recommended": total_score >= self.recommend_threshold,
        }

    def rank_wallets(self, wallet_scores: list[dict]) -> list[dict]:
        """Sort wallet scores by total_score descending."""
        return sorted(wallet_scores, key=lambda w: w.get("total_score", 0), reverse=True)

    def _score_risk_adjusted(self, trades: list[dict]) -> float:
        """Score based on estimated Sharpe ratio from PnL series."""
        pnls = [t.get("pnl", 0) for t in trades]
        if not pnls:
            return 0.0

        mean_pnl = sum(pnls) / len(pnls)
        if len(pnls) < 2:
            return 50.0 if mean_pnl > 0 else 0.0

        variance = sum((p - mean_pnl) ** 2 for p in pnls) / (len(pnls) - 1)
        std_pnl = math.sqrt(variance) if variance > 0 else 0.0

        if std_pnl == 0:
            return 100.0 if mean_pnl > 0 else 0.0

        sharpe = mean_pnl / std_pnl
        # Map sharpe to 0-100: sharpe of 0 = 30, 1.0 = 60, 2.0 = 80, 3.0+ = 100
        score = 30 + (sharpe * 23.3)
        return max(0.0, min(100.0, score))

    def _score_consistency(self, trades: list[dict]) -> float:
        """Score based on PnL variance — lower variance = higher score."""
        pnls = [t.get("pnl", 0) for t in trades]
        if len(pnls) < 2:
            return 0.0

        mean_pnl = sum(pnls) / len(pnls)
        variance = sum((p - mean_pnl) ** 2 for p in pnls) / (len(pnls) - 1)
        std_pnl = math.sqrt(variance) if variance > 0 else 0.0

        # Coefficient of variation (normalized)
        if abs(mean_pnl) < 1e-10:
            return 50.0

        cv = std_pnl / abs(mean_pnl)
        # Low CV = high consistency: CV of 0 = 100, 1.0 = 60, 2.0 = 30, 3.0+ = 0
        score = 100 - (cv * 33.3)
        return max(0.0, min(100.0, score))

    def _score_drawdown(self, trades: list[dict]) -> float:
        """Score based on max drawdown — lower DD = higher score (inverted)."""
        pnls = [t.get("pnl", 0) for t in trades]
        if not pnls:
            return 0.0

        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0

        for pnl in pnls:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd

        if peak <= 0:
            return 0.0

        dd_pct = (max_dd / peak) * 100 if peak > 0 else 0
        # Low DD% = high score: 0% = 100, 10% = 80, 25% = 50, 50%+ = 0
        score = 100 - (dd_pct * 2)
        return max(0.0, min(100.0, score))

    def _score_frequency(self, trades: list[dict]) -> float:
        """Score based on trade frequency — filters lucky one-shots."""
        count = len(trades)
        if count < self.min_trades:
            return 0.0

        # More trades = higher confidence: 10 = 30, 50 = 60, 100 = 80, 200+ = 100
        score = min(100.0, 30 + (count - self.min_trades) * 0.37)
        return max(0.0, score)

    def _score_diversity(self, trades: list[dict]) -> float:
        """Score based on asset diversity — trading multiple symbols."""
        symbols = set(t.get("symbol", "") for t in trades if t.get("symbol"))
        count = len(symbols)
        if count == 0:
            return 0.0

        # More symbols = higher diversity: 1 = 20, 3 = 50, 5 = 70, 10+ = 100
        score = min(100.0, count * 10 + 10)
        return max(0.0, score)

    def _compute_grade(self, score: float) -> str:
        """Map numeric score to letter grade."""
        for threshold, grade in GRADE_THRESHOLDS:
            if score >= threshold:
                return grade
        return "F"
