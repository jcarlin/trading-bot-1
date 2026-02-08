"""Selects the optimal execution algorithm based on order characteristics."""

import logging

from execution.algorithms.adaptive_limit import AdaptiveLimitAlgorithm
from execution.algorithms.iceberg import IcebergAlgorithm
from execution.algorithms.twap import TWAPAlgorithm

logger = logging.getLogger(__name__)


class AlgorithmSelector:
    """Selects the optimal execution algorithm for an order.

    Selection logic:
    - Signal metadata override: ``signal.metadata["exec_algo"]`` wins
    - Small orders (< 1% avg volume): adaptive_limit
    - Medium orders (1-5% avg volume): TWAP
    - Large orders (> 5% avg volume): iceberg
    - Urgent / volatile regime: market (returns None algo)
    """

    ALGORITHMS = {
        "market": None,
        "twap": TWAPAlgorithm,
        "adaptive_limit": AdaptiveLimitAlgorithm,
        "iceberg": IcebergAlgorithm,
    }

    def __init__(self, config: dict = None):
        self.config = config or {}
        self._small_threshold_pct = self.config.get("small_threshold_pct", 1.0)
        self._large_threshold_pct = self.config.get("large_threshold_pct", 5.0)
        self._volatile_regimes = self.config.get(
            "volatile_regimes", ["volatile", "crisis"]
        )
        self._algo_configs = self.config.get("algo_configs", {})

    def select(
        self, order, market_state: dict = None, signal_metadata: dict = None
    ) -> tuple:
        """Select the best algorithm for the given order.

        Parameters
        ----------
        order:
            The Order to execute.
        market_state:
            Dict with optional keys: ``avg_volume``, ``regime``.
        signal_metadata:
            Optional signal metadata; may contain ``exec_algo`` key.

        Returns
        -------
        tuple
            (algo_name: str, algo_instance: ExecutionAlgorithm | None)
        """
        market_state = market_state or {}
        signal_metadata = signal_metadata or {}

        # 1. Check for explicit override in signal metadata
        override = signal_metadata.get("exec_algo")
        if override and override in self.ALGORITHMS:
            algo_cls = self.ALGORITHMS[override]
            algo = algo_cls() if algo_cls else None
            logger.info("Algorithm override from signal metadata: %s", override)
            return (override, algo)

        # 2. Volatile regime → market
        regime = market_state.get("regime", "")
        if regime in self._volatile_regimes:
            logger.info("Volatile regime (%s) — using market order", regime)
            return ("market", None)

        # 3. Size-based selection
        avg_volume = market_state.get("avg_volume", 0)
        if avg_volume > 0:
            order_pct = (order.quantity / avg_volume) * 100.0
        else:
            # Without volume data, default to adaptive_limit
            algo = AdaptiveLimitAlgorithm()
            return ("adaptive_limit", algo)

        if order_pct < self._small_threshold_pct:
            algo = AdaptiveLimitAlgorithm()
            return ("adaptive_limit", algo)
        elif order_pct <= self._large_threshold_pct:
            algo = TWAPAlgorithm()
            return ("twap", algo)
        else:
            algo = IcebergAlgorithm()
            return ("iceberg", algo)

    def get_available_algorithms(self) -> list[str]:
        """Return names of all available algorithms."""
        return list(self.ALGORITHMS.keys())

    def get_algo_config(self, algo_name: str) -> dict:
        """Return config for a specific algorithm."""
        return self._algo_configs.get(algo_name, {})
