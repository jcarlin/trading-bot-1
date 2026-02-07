"""Allocation optimizer for multi-strategy portfolio management.

Supports equal_weight, risk_parity, and health_weighted allocation modes
with min/max/step constraints per strategy.
"""

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)


class AllocationOptimizer:
    """Computes optimal allocation weights across active strategies.

    Modes:
        equal_weight: Distribute capital equally across all strategies.
        risk_parity: Allocate inversely proportional to volatility.
        health_weighted: Allocate proportional to health scores.
    """

    VALID_MODES = ["equal_weight", "risk_parity", "health_weighted"]

    def __init__(self, config: dict = None):
        config = config or {}
        self.mode = config.get("mode", "equal_weight")
        self.min_allocation = config.get("min_allocation", 0.05)
        self.max_allocation = config.get("max_allocation", 0.60)
        self.step = config.get("step", 0.05)
        self.cash_reserve = config.get("cash_reserve", 0.20)

        if self.mode not in self.VALID_MODES:
            logger.warning("Invalid mode '%s', falling back to equal_weight", self.mode)
            self.mode = "equal_weight"

    def optimize(self, strategies: list[dict]) -> dict[str, float]:
        """Compute allocation weights for a list of strategies.

        Args:
            strategies: List of dicts, each with keys:
                - name (str): Strategy name.
                - health_score (float): 0-100 health score.
                - volatility (float): Annualized volatility of returns.
                - sharpe (float): Sharpe ratio.
                - status (str): "active" or "paused".

        Returns:
            Dict mapping strategy_name -> allocation weight (0.0 to 1.0).
            Weights sum to (1.0 - cash_reserve) at most.
        """
        active = [s for s in strategies if s.get("status", "active") == "active"]

        if not active:
            return {}

        if self.mode == "equal_weight":
            raw_weights = self._equal_weight(active)
        elif self.mode == "risk_parity":
            raw_weights = self._risk_parity(active)
        elif self.mode == "health_weighted":
            raw_weights = self._health_weighted(active)
        else:
            raw_weights = self._equal_weight(active)

        return self._apply_constraints(raw_weights)

    def _equal_weight(self, strategies: list[dict]) -> dict[str, float]:
        """Equal allocation across all active strategies."""
        n = len(strategies)
        available = 1.0 - self.cash_reserve
        weight = available / n
        return {s["name"]: weight for s in strategies}

    def _risk_parity(self, strategies: list[dict]) -> dict[str, float]:
        """Allocate inversely proportional to volatility.

        Lower volatility -> higher allocation.
        Falls back to equal_weight if no volatility data.
        """
        vols = {}
        for s in strategies:
            vol = s.get("volatility", 0.0)
            if vol > 0:
                vols[s["name"]] = vol

        if not vols:
            return self._equal_weight(strategies)

        # Inverse volatility weights
        inv_vols = {name: 1.0 / vol for name, vol in vols.items()}
        total_inv = sum(inv_vols.values())

        available = 1.0 - self.cash_reserve
        weights = {name: (iv / total_inv) * available for name, iv in inv_vols.items()}

        # Add equal_weight for strategies without vol data
        names_with_vol = set(vols.keys())
        missing = [s for s in strategies if s["name"] not in names_with_vol]
        if missing:
            remaining = available - sum(weights.values())
            per_missing = remaining / len(missing) if remaining > 0 else self.min_allocation
            for s in missing:
                weights[s["name"]] = per_missing

        return weights

    def _health_weighted(self, strategies: list[dict]) -> dict[str, float]:
        """Allocate proportional to health scores.

        Higher health -> higher allocation.
        Falls back to equal_weight if no health data.
        """
        scores = {}
        for s in strategies:
            score = s.get("health_score", 0.0)
            scores[s["name"]] = max(score, 1.0)  # floor at 1 to avoid zero

        total_score = sum(scores.values())
        if total_score <= 0:
            return self._equal_weight(strategies)

        available = 1.0 - self.cash_reserve
        return {name: (score / total_score) * available
                for name, score in scores.items()}

    def _apply_constraints(self, weights: dict[str, float]) -> dict[str, float]:
        """Apply min/max/step constraints and re-normalize."""
        constrained = {}

        for name, w in weights.items():
            # Apply min/max
            w = max(w, self.min_allocation)
            w = min(w, self.max_allocation)

            # Snap to step
            if self.step > 0:
                w = round(w / self.step) * self.step

            # Re-apply bounds after snapping
            w = max(w, self.min_allocation)
            w = min(w, self.max_allocation)

            constrained[name] = round(w, 4)

        # Ensure total doesn't exceed budget
        available = 1.0 - self.cash_reserve
        total = sum(constrained.values())

        if total > available and total > 0:
            scale = available / total
            constrained = {name: round(w * scale, 4)
                          for name, w in constrained.items()}

        return constrained

    def get_rebalance_actions(self, current: dict[str, float],
                              target: dict[str, float],
                              threshold: float = 0.05) -> list[dict]:
        """Compute rebalance actions needed to move from current to target.

        Args:
            current: Current allocation weights.
            target: Target allocation weights.
            threshold: Minimum change to trigger a rebalance action.

        Returns:
            List of dicts with strategy_name, current_weight,
            target_weight, and change.
        """
        actions = []
        all_names = set(list(current.keys()) + list(target.keys()))

        for name in sorted(all_names):
            cur = current.get(name, 0.0)
            tgt = target.get(name, 0.0)
            change = tgt - cur

            if abs(change) >= threshold:
                actions.append({
                    "strategy_name": name,
                    "current_weight": round(cur, 4),
                    "target_weight": round(tgt, 4),
                    "change": round(change, 4),
                })

        return actions
