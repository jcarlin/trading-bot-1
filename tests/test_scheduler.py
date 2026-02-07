"""Tests for orchestration.scheduler.EvaluationScheduler."""

import asyncio
import os
import sys
import unittest
from unittest.mock import MagicMock, AsyncMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from orchestration.scheduler import EvaluationScheduler


class TestEvaluationScheduler(unittest.TestCase):
    """Tests for EvaluationScheduler."""

    def setUp(self):
        self.mock_orchestrator = MagicMock()
        self.mock_orchestrator.evaluate = AsyncMock()

    def test_default_intervals(self):
        """Should use default intervals when no config provided."""
        scheduler = EvaluationScheduler(self.mock_orchestrator)
        self.assertEqual(scheduler.intervals["hourly"], 3600)
        self.assertEqual(scheduler.intervals["daily"], 86400)
        self.assertEqual(scheduler.intervals["weekly"], 604800)
        self.assertEqual(scheduler.intervals["monthly"], 2592000)

    def test_custom_intervals(self):
        """Should use custom intervals from config."""
        config = {
            "hourly_interval": 1800,
            "daily_interval": 43200,
        }
        scheduler = EvaluationScheduler(self.mock_orchestrator, config)
        self.assertEqual(scheduler.intervals["hourly"], 1800)
        self.assertEqual(scheduler.intervals["daily"], 43200)

    def test_scheduler_runs_and_stops(self):
        """Scheduler should run and stop cleanly on stop_event."""
        scheduler = EvaluationScheduler(self.mock_orchestrator, {
            "hourly_interval": 0.1,  # Very short for testing
            "daily_interval": 100000,
            "weekly_interval": 100000,
            "monthly_interval": 100000,
        })

        loop = asyncio.new_event_loop()
        try:
            stop_event = asyncio.Event()

            async def run_test():
                task = asyncio.create_task(scheduler.run(stop_event))
                await asyncio.sleep(0.3)  # Let it run a few cycles
                stop_event.set()
                await asyncio.sleep(0.1)
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            loop.run_until_complete(run_test())

            # Should have called evaluate at least once
            self.assertTrue(self.mock_orchestrator.evaluate.called)
            # Should have been called with "hourly"
            call_args = [
                call[0][0] for call in self.mock_orchestrator.evaluate.call_args_list
            ]
            self.assertIn("hourly", call_args)
        finally:
            loop.close()

    def test_scheduler_handles_evaluation_errors(self):
        """Scheduler should continue if evaluation raises an exception."""
        self.mock_orchestrator.evaluate = AsyncMock(
            side_effect=Exception("Eval error"))

        scheduler = EvaluationScheduler(self.mock_orchestrator, {
            "hourly_interval": 0.1,
            "daily_interval": 100000,
            "weekly_interval": 100000,
            "monthly_interval": 100000,
        })

        loop = asyncio.new_event_loop()
        try:
            stop_event = asyncio.Event()

            async def run_test():
                task = asyncio.create_task(scheduler.run(stop_event))
                await asyncio.sleep(0.3)
                stop_event.set()
                await asyncio.sleep(0.1)
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            # Should not raise
            loop.run_until_complete(run_test())
        finally:
            loop.close()


if __name__ == "__main__":
    unittest.main()
