"""Evaluation scheduler for periodic strategy assessment."""

import asyncio
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class EvaluationScheduler:
    """Async scheduler for periodic evaluation cycles.

    Launches parallel loops for hourly, daily, weekly, and monthly
    evaluation checkpoints.
    """

    def __init__(self, orchestrator, config: Optional[dict] = None):
        config = config or {}
        self.orchestrator = orchestrator
        self.intervals = {
            "hourly": config.get("hourly_interval", 3600),
            "daily": config.get("daily_interval", 86400),
            "weekly": config.get("weekly_interval", 604800),
            "monthly": config.get("monthly_interval", 2592000),
        }

    async def run(self, stop_event: asyncio.Event) -> None:
        """Launch all evaluation loops concurrently."""
        tasks = [
            asyncio.create_task(
                self._evaluation_loop(checkpoint_type, interval, stop_event))
            for checkpoint_type, interval in self.intervals.items()
        ]

        logger.info("EvaluationScheduler started with intervals: %s", self.intervals)

        try:
            await stop_event.wait()
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info("EvaluationScheduler stopped")

    async def _evaluation_loop(self, checkpoint_type: str, interval: int,
                                stop_event: asyncio.Event) -> None:
        """Run a single evaluation loop at the specified interval."""
        logger.info("Evaluation loop started: %s (every %ds)", checkpoint_type, interval)

        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
                break  # stop_event was set
            except asyncio.TimeoutError:
                pass  # interval elapsed, run evaluation

            try:
                start_time = time.time()
                logger.info("Running %s evaluation", checkpoint_type)
                await self.orchestrator.evaluate(checkpoint_type)
                duration = time.time() - start_time
                logger.info("%s evaluation completed in %.2fs",
                           checkpoint_type, duration)
            except Exception:
                logger.exception("Error in %s evaluation", checkpoint_type)
