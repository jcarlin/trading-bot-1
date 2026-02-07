"""Tests for strategy.manager.StrategyManager."""

import asyncio
import os
import sys
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from strategy.manager import StrategyManager


class TestStrategyManager(unittest.TestCase):
    """Tests for StrategyManager."""

    def setUp(self):
        self.mock_engine = MagicMock()
        self.mock_timescale = MagicMock()
        self.mock_redis = MagicMock()
        self.mock_redis._r = MagicMock()
        # Make redis._r.get return None by default (no saved state)
        self.mock_redis._r.get.return_value = None
        self.mock_config = MagicMock()
        self.mock_config.get.return_value = "BTC/USDC"
        self.mock_decision_logger = MagicMock()

        self.manager = StrategyManager(
            execution_engine=self.mock_engine,
            timescale=self.mock_timescale,
            redis_store=self.mock_redis,
            config=self.mock_config,
            decision_logger=self.mock_decision_logger,
        )

    def _make_mock_strategy(self, name="test_strategy"):
        strategy = MagicMock()
        strategy.get_metadata.return_value = {"name": name}
        strategy.get_state.return_value = {}
        return strategy

    def _run_async(self, coro):
        """Run an async coroutine with proper cleanup of created tasks."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            # Cancel all pending tasks created during the test
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(
                    *pending, return_exceptions=True))
            loop.close()

    def test_start_strategy(self):
        """Should start a strategy and track it as active."""
        strategy = self._make_mock_strategy()

        with patch("strategy.manager.StrategyRunner") as mock_runner_cls:
            mock_runner = MagicMock()
            mock_runner.run = AsyncMock()
            mock_runner_cls.return_value = mock_runner

            self._run_async(
                self.manager.start_strategy("test", strategy))

            self.assertEqual(self.manager.get_strategy_status("test"), "active")
            self.assertIn("test", self.manager.get_active_strategies())

    def test_pause_strategy(self):
        """Should pause an active strategy."""
        strategy = self._make_mock_strategy()

        with patch("strategy.manager.StrategyRunner") as mock_runner_cls:
            mock_runner = MagicMock()
            mock_runner.run = AsyncMock()
            mock_runner_cls.return_value = mock_runner

            async def _test():
                await self.manager.start_strategy("test", strategy)
                await self.manager.pause_strategy("test")

            self._run_async(_test())

            self.assertEqual(self.manager.get_strategy_status("test"), "paused")
            self.assertNotIn("test", self.manager.get_active_strategies())

    def test_resume_strategy(self):
        """Should resume a paused strategy."""
        strategy = self._make_mock_strategy()

        with patch("strategy.manager.StrategyRunner") as mock_runner_cls:
            mock_runner = MagicMock()
            mock_runner.run = AsyncMock()
            mock_runner_cls.return_value = mock_runner

            async def _test():
                await self.manager.start_strategy("test", strategy)
                await self.manager.pause_strategy("test")
                await self.manager.resume_strategy("test")

            self._run_async(_test())

            self.assertEqual(self.manager.get_strategy_status("test"), "active")

    def test_swap_strategy(self):
        """Should swap one strategy for another."""
        old_strategy = self._make_mock_strategy("old")
        new_strategy = self._make_mock_strategy("new")

        with patch("strategy.manager.StrategyRunner") as mock_runner_cls:
            mock_runner = MagicMock()
            mock_runner.run = AsyncMock()
            mock_runner_cls.return_value = mock_runner

            async def _test():
                await self.manager.start_strategy("old", old_strategy)
                await self.manager.swap_strategy("old", new_strategy, "new")

            self._run_async(_test())

            self.assertEqual(self.manager.get_strategy_status("old"), "paused")
            self.assertEqual(self.manager.get_strategy_status("new"), "active")

    def test_get_unknown_strategy_status(self):
        """Unknown strategy should return 'unknown'."""
        self.assertEqual(
            self.manager.get_strategy_status("nonexistent"), "unknown")

    def test_get_all_statuses(self):
        """Should return all strategy statuses."""
        strategy = self._make_mock_strategy()

        with patch("strategy.manager.StrategyRunner") as mock_runner_cls:
            mock_runner = MagicMock()
            mock_runner.run = AsyncMock()
            mock_runner_cls.return_value = mock_runner

            self._run_async(
                self.manager.start_strategy("s1", strategy))

            statuses = self.manager.get_all_statuses()
            self.assertEqual(statuses["s1"], "active")

    def test_pause_nonexistent_strategy(self):
        """Pausing a nonexistent strategy should not raise."""
        self._run_async(
            self.manager.pause_strategy("nonexistent"))

    def test_stop_all(self):
        """Should stop all active strategies."""
        strategy = self._make_mock_strategy()

        with patch("strategy.manager.StrategyRunner") as mock_runner_cls:
            mock_runner = MagicMock()
            mock_runner.run = AsyncMock()
            mock_runner_cls.return_value = mock_runner

            async def _test():
                await self.manager.start_strategy("s1", strategy)
                await self.manager.stop_all()

            self._run_async(_test())

            self.assertEqual(self.manager.get_strategy_status("s1"), "paused")


if __name__ == "__main__":
    unittest.main()
