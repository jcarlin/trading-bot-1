"""Strategy health scoring from composite performance metrics."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class StrategyHealthScorer:
    """Computes a composite health score (0-100) from weighted performance metrics.

    Uses StrategyPerformanceTracker to get raw metrics, normalizes each
    to 0-100, then computes a weighted average.
    """

    DEFAULT_WEIGHTS = {
        "sharpe": 0.25,
        "sortino": 0.15,
        "max_drawdown": 0.20,
        "win_rate": 0.15,
        "profit_factor": 0.15,
        "calmar": 0.10,
    }

    # Normalization ranges: (min_for_0, max_for_100)
    NORMALIZATION = {
        "sharpe": (0.0, 3.0),
        "sortino": (0.0, 4.0),
        "max_drawdown": (10.0, 0.0),  # inverted: lower DD = higher score
        "win_rate": (0.0, 100.0),
        "profit_factor": (1.0, 3.0),
        "calmar": (0.0, 5.0),
    }

    GRADES = [
        (80, "A"),
        (60, "B"),
        (40, "C"),
        (20, "D"),
        (0, "F"),
    ]

    def __init__(self, performance_tracker, weights: Optional[dict] = None):
        self.performance_tracker = performance_tracker
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

    def compute_health_score(self, window_hours: int = 168) -> dict:
        """Compute composite health score for a strategy.

        Args:
            window_hours: Rolling window in hours (default: 168 = 1 week).

        Returns:
            dict with health_score, grade, components, raw_metrics.
        """
        raw_metrics = self.performance_tracker.compute_metrics(window_hours)

        # Map raw metrics to component names
        metric_mapping = {
            "sharpe": raw_metrics.get("sharpe_ratio", 0.0),
            "sortino": raw_metrics.get("sortino_ratio", 0.0),
            "max_drawdown": raw_metrics.get("max_drawdown", 0.0),
            "win_rate": raw_metrics.get("win_rate", 0.0),
            "profit_factor": raw_metrics.get("profit_factor", 0.0),
            "calmar": raw_metrics.get("calmar_ratio", 0.0),
        }

        # Normalize each metric to 0-100
        components = {}
        for metric_name, raw_value in metric_mapping.items():
            norm_range = self.NORMALIZATION.get(metric_name, (0.0, 1.0))
            normalized = self._normalize(raw_value, norm_range[0], norm_range[1])
            components[metric_name] = round(normalized, 2)

        # Weighted sum
        health_score = 0.0
        for metric_name, weight in self.weights.items():
            health_score += components.get(metric_name, 0.0) * weight

        health_score = round(min(100.0, max(0.0, health_score)), 2)
        grade = self._compute_grade(health_score)

        return {
            "health_score": health_score,
            "grade": grade,
            "components": components,
            "raw_metrics": raw_metrics,
        }

    def should_pause(self, window_hours: int = 168) -> tuple:
        """Determine if the strategy should be paused.

        Returns:
            (should_pause: bool, reason: str)
        """
        result = self.compute_health_score(window_hours)
        score = result["health_score"]
        max_dd = result["raw_metrics"].get("max_drawdown", 0.0)

        if score < 20:
            return (True, f"Health score critically low: {score:.1f} (grade F)")

        if max_dd > 5.0:
            return (True, f"Max drawdown exceeds limit: {max_dd:.2f}% > 5.0%")

        return (False, "")

    @staticmethod
    def _normalize(value: float, min_val: float, max_val: float) -> float:
        """Normalize a value to 0-100 scale.

        For inverted metrics (like drawdown where lower is better),
        min_val > max_val.
        """
        if min_val > max_val:
            # Inverted: e.g., drawdown where 0% = 100 score, 10% = 0 score
            if value <= max_val:
                return 100.0
            if value >= min_val:
                return 0.0
            return (min_val - value) / (min_val - max_val) * 100.0
        else:
            # Normal: higher value = higher score
            if value <= min_val:
                return 0.0
            if value >= max_val:
                return 100.0
            return (value - min_val) / (max_val - min_val) * 100.0

    def _compute_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        for threshold, grade in self.GRADES:
            if score >= threshold:
                return grade
        return "F"
