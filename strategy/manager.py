"""Strategy lifecycle manager for hot-swap, pause, and resume."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from strategy.live_strategy import LiveStrategy
from strategy.runner import StrategyRunner
from strategy.decision_logger import StrategyDecisionLogger

logger = logging.getLogger(__name__)


class StrategyManager:
    """Manages strategy lifecycle: start, pause, resume, and hot-swap.

    Maintains a registry of active strategies and their runners,
    enabling runtime strategy management without system restart.
    """

    def __init__(self, execution_engine, timescale, redis_store,
                 config, decision_logger: Optional[StrategyDecisionLogger] = None):
        self.execution_engine = execution_engine
        self.timescale = timescale
        self.redis_store = redis_store
        self.config = config
        self.decision_logger = decision_logger

        self._strategies: dict[str, dict] = {}
        self._stop_events: dict[str, asyncio.Event] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    async def start_strategy(self, name: str, strategy: LiveStrategy) -> None:
        """Start a new strategy.

        Args:
            name: Unique strategy name.
            strategy: LiveStrategy instance to run.
        """
        if name in self._strategies and self._strategies[name]["status"] == "active":
            logger.warning("Strategy %s is already active", name)
            return

        # Create decision logger for this strategy
        strategy_logger = self.decision_logger or StrategyDecisionLogger(
            self.timescale, name)

        # Create runner
        runner = StrategyRunner(
            strategy=strategy,
            execution_engine=self.execution_engine,
            timescale=self.timescale,
            redis=self.redis_store,
            config=self.config,
            decision_logger=strategy_logger,
        )

        # Create stop event and launch
        stop_event = asyncio.Event()
        task = asyncio.create_task(runner.run(stop_event))

        self._strategies[name] = {
            "strategy": strategy,
            "runner": runner,
            "status": "active",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        self._stop_events[name] = stop_event
        self._tasks[name] = task

        logger.info("Strategy %s started", name)

    async def pause_strategy(self, name: str) -> None:
        """Pause a running strategy.

        Stops the tick loop, checkpoints state to Redis.
        """
        if name not in self._strategies:
            logger.warning("Strategy %s not found", name)
            return

        if self._strategies[name]["status"] != "active":
            logger.warning("Strategy %s is not active (status=%s)",
                          name, self._strategies[name]["status"])
            return

        # Signal stop
        stop_event = self._stop_events.get(name)
        if stop_event:
            stop_event.set()

        # Wait for task to complete
        task = self._tasks.get(name)
        if task:
            try:
                await asyncio.wait_for(task, timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("Strategy %s did not stop in time, cancelling", name)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Checkpoint state
        strategy = self._strategies[name]["strategy"]
        state = strategy.get_state()
        try:
            import json
            self.redis_store._r.set(
                f"strategy:{name}:full_state",
                json.dumps(state),
            )
        except Exception:
            logger.exception("Failed to checkpoint state for %s", name)

        self._strategies[name]["status"] = "paused"
        self._strategies[name]["paused_at"] = datetime.now(timezone.utc).isoformat()

        # Log the pause decision
        if self.decision_logger:
            self.decision_logger.log_state_change(
                "strategy_paused",
                old_state={"status": "active"},
                new_state={"status": "paused"},
                reason=f"Strategy {name} paused",
            )

        logger.info("Strategy %s paused", name)

    async def resume_strategy(self, name: str) -> None:
        """Resume a paused strategy.

        Restores state from Redis and restarts the tick loop.
        """
        if name not in self._strategies:
            logger.warning("Strategy %s not found", name)
            return

        if self._strategies[name]["status"] != "paused":
            logger.warning("Strategy %s is not paused (status=%s)",
                          name, self._strategies[name]["status"])
            return

        strategy = self._strategies[name]["strategy"]

        # Restore state from Redis
        try:
            import json
            state_json = self.redis_store._r.get(f"strategy:{name}:full_state")
            if state_json:
                state = json.loads(state_json)
                strategy.set_state(state)
        except Exception:
            logger.exception("Failed to restore state for %s", name)

        # Create new runner and task
        strategy_logger = self.decision_logger or StrategyDecisionLogger(
            self.timescale, name)

        runner = StrategyRunner(
            strategy=strategy,
            execution_engine=self.execution_engine,
            timescale=self.timescale,
            redis=self.redis_store,
            config=self.config,
            decision_logger=strategy_logger,
        )

        stop_event = asyncio.Event()
        task = asyncio.create_task(runner.run(stop_event))

        self._strategies[name]["runner"] = runner
        self._strategies[name]["status"] = "active"
        self._strategies[name]["resumed_at"] = datetime.now(timezone.utc).isoformat()
        self._stop_events[name] = stop_event
        self._tasks[name] = task

        # Log the resume decision
        if self.decision_logger:
            self.decision_logger.log_state_change(
                "strategy_resumed",
                old_state={"status": "paused"},
                new_state={"status": "active"},
                reason=f"Strategy {name} resumed",
            )

        logger.info("Strategy %s resumed", name)

    async def swap_strategy(self, old_name: str, new_strategy: LiveStrategy,
                            new_name: str) -> None:
        """Replace one strategy with another.

        Pauses the old strategy and starts the new one.
        """
        logger.info("Swapping strategy %s -> %s", old_name, new_name)

        # Pause old strategy if active
        if old_name in self._strategies and self._strategies[old_name]["status"] == "active":
            await self.pause_strategy(old_name)

        # Start new strategy
        await self.start_strategy(new_name, new_strategy)

        # Log the swap
        if self.decision_logger:
            self.decision_logger.log_state_change(
                "strategy_swapped",
                old_state={"strategy": old_name, "status": "paused"},
                new_state={"strategy": new_name, "status": "active"},
                reason=f"Swapped {old_name} for {new_name}",
            )

        logger.info("Strategy swap complete: %s -> %s", old_name, new_name)

    def get_active_strategies(self) -> list[str]:
        """Return names of all active strategies."""
        return [
            name for name, info in self._strategies.items()
            if info["status"] == "active"
        ]

    def get_strategy_status(self, name: str) -> str:
        """Return status of a strategy."""
        if name not in self._strategies:
            return "unknown"
        return self._strategies[name]["status"]

    def get_all_statuses(self) -> dict[str, str]:
        """Return status of all strategies."""
        return {
            name: info["status"]
            for name, info in self._strategies.items()
        }

    async def stop_all(self) -> None:
        """Stop all active strategies."""
        for name in list(self._strategies.keys()):
            if self._strategies[name]["status"] == "active":
                await self.pause_strategy(name)
        logger.info("All strategies stopped")
