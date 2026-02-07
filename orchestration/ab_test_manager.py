"""A/B test manager: orchestrates shadow vs live strategy comparisons."""

import asyncio
import logging
import uuid
from datetime import datetime, timezone

from orchestration.promotion_criteria import PromotionCriteria
from orchestration.shadow_runner import ShadowRunner
from strategy.live_strategy import LiveStrategy

logger = logging.getLogger(__name__)


class ABTestManager:
    """Manages A/B tests between live and shadow strategies.

    Creates ShadowRunners for candidate strategies, compares their
    hypothetical performance against live, and supports promotion/rejection.
    """

    def __init__(self, strategy_manager, promotion_criteria: PromotionCriteria,
                 timescale, redis, config):
        self.strategy_manager = strategy_manager
        self.promotion_criteria = promotion_criteria
        self.timescale = timescale
        self.redis = redis
        self.config = config

        self._tests: dict[str, dict] = {}
        self._stop_events: dict[str, asyncio.Event] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    async def start_test(self, live_name: str, shadow_strategy: LiveStrategy,
                         shadow_name: str) -> str:
        """Start a new A/B test.

        Args:
            live_name: Name of the live strategy to compare against.
            shadow_strategy: The shadow LiveStrategy instance.
            shadow_name: Name for the shadow strategy.

        Returns:
            test_id: UUID string identifying this test.
        """
        test_id = str(uuid.uuid4())

        shadow_runner = ShadowRunner(
            strategy=shadow_strategy,
            timescale=self.timescale,
            redis=self.redis,
            config=self.config,
            name=shadow_name,
        )

        stop_event = asyncio.Event()
        task = asyncio.create_task(shadow_runner.run(stop_event))

        self._tests[test_id] = {
            "live_name": live_name,
            "shadow_name": shadow_name,
            "shadow_strategy": shadow_strategy,
            "shadow_runner": shadow_runner,
            "start_time": datetime.now(timezone.utc),
            "status": "active",
        }
        self._stop_events[test_id] = stop_event
        self._tasks[test_id] = task

        logger.info("A/B test started: id=%s live=%s shadow=%s",
                     test_id, live_name, shadow_name)

        return test_id

    async def check_test(self, test_id: str) -> dict:
        """Check an A/B test's status and compare metrics.

        Args:
            test_id: The UUID of the test to check.

        Returns:
            Dict with status, shadow_metrics, live_metrics,
            recommendation, and meets_criteria.
        """
        if test_id not in self._tests:
            return {"status": "not_found"}

        test = self._tests[test_id]

        if test["status"] != "active":
            return {
                "status": test["status"],
                "shadow_metrics": {},
                "live_metrics": {},
                "recommendation": "test_concluded",
                "meets_criteria": False,
            }

        shadow_runner = test["shadow_runner"]
        shadow_metrics = shadow_runner.get_performance_summary()

        # Get live metrics from TimescaleDB
        live_metrics = self._get_live_metrics(test["live_name"])

        # Calculate duration
        now = datetime.now(timezone.utc)
        duration = now - test["start_time"]
        duration_hours = duration.total_seconds() / 3600

        # Evaluate against promotion criteria
        evaluation = self.promotion_criteria.evaluate(
            shadow_metrics, live_metrics, duration_hours)

        if evaluation["meets_criteria"]:
            recommendation = "promote"
        elif duration_hours < self.promotion_criteria.min_duration_hours:
            recommendation = "continue"
        else:
            recommendation = "reject"

        return {
            "status": "active",
            "shadow_metrics": shadow_metrics,
            "live_metrics": live_metrics,
            "recommendation": recommendation,
            "meets_criteria": evaluation["meets_criteria"],
            "checks": evaluation["checks"],
            "duration_hours": round(duration_hours, 2),
        }

    async def check_all_tests(self) -> list[dict]:
        """Check all active tests."""
        results = []
        for test_id in list(self._tests.keys()):
            if self._tests[test_id]["status"] == "active":
                result = await self.check_test(test_id)
                result["test_id"] = test_id
                results.append(result)
        return results

    async def promote(self, test_id: str) -> None:
        """Promote a shadow strategy to live.

        Stops the shadow runner and swaps the live strategy.
        """
        if test_id not in self._tests:
            logger.warning("Cannot promote: test %s not found", test_id)
            return

        test = self._tests[test_id]

        # Stop shadow runner
        await self._stop_shadow(test_id)

        # Swap live strategy via StrategyManager
        await self.strategy_manager.swap_strategy(
            test["live_name"],
            test["shadow_strategy"],
            test["shadow_name"],
        )

        test["status"] = "promoted"
        test["concluded_at"] = datetime.now(timezone.utc).isoformat()

        logger.info("A/B test %s: shadow '%s' promoted to replace '%s'",
                     test_id, test["shadow_name"], test["live_name"])

    async def reject(self, test_id: str) -> None:
        """Reject a shadow strategy and stop the test."""
        if test_id not in self._tests:
            logger.warning("Cannot reject: test %s not found", test_id)
            return

        await self._stop_shadow(test_id)

        self._tests[test_id]["status"] = "rejected"
        self._tests[test_id]["concluded_at"] = datetime.now(timezone.utc).isoformat()

        logger.info("A/B test %s: shadow '%s' rejected",
                     test_id, self._tests[test_id]["shadow_name"])

    def get_active_tests(self) -> list[dict]:
        """Return a summary of all active tests."""
        active = []
        for test_id, test in self._tests.items():
            if test["status"] == "active":
                active.append({
                    "test_id": test_id,
                    "live_name": test["live_name"],
                    "shadow_name": test["shadow_name"],
                    "start_time": test["start_time"].isoformat(),
                    "status": test["status"],
                })
        return active

    async def _stop_shadow(self, test_id: str) -> None:
        """Stop a shadow runner for the given test."""
        stop_event = self._stop_events.get(test_id)
        if stop_event:
            stop_event.set()

        task = self._tasks.get(test_id)
        if task:
            try:
                await asyncio.wait_for(task, timeout=10.0)
            except asyncio.TimeoutError:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    def _get_live_metrics(self, strategy_name: str) -> dict:
        """Get live strategy metrics from TimescaleDB."""
        try:
            from metrics.strategy_performance import StrategyPerformanceTracker
            tracker = StrategyPerformanceTracker(self.timescale, strategy_name)
            return tracker.compute_metrics(window_hours=168)
        except Exception:
            logger.debug("Could not get live metrics for %s, returning empty",
                        strategy_name)
            return {
                "trade_count": 0,
                "total_pnl": 0.0,
                "win_rate": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
            }
