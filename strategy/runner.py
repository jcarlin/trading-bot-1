"""Strategy runner: connects LiveStrategy to the ExecutionEngine."""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from core.models import Signal
from core.types import SignalType
from monitoring.metrics import strategy_signal_count
from strategy.decision_logger import StrategyDecisionLogger
from strategy.live_strategy import LiveStrategy, MarketState

logger = logging.getLogger(__name__)


class StrategyRunner:
    """Live execution loop connecting a LiveStrategy to the ExecutionEngine."""

    def __init__(self, strategy: LiveStrategy, execution_engine,
                 timescale, redis, config, decision_logger: StrategyDecisionLogger,
                 signal_aggregator=None):
        self.strategy = strategy
        self.execution_engine = execution_engine
        self.timescale = timescale
        self.redis = redis
        self.config = config
        self.decision_logger = decision_logger
        self.signal_aggregator = signal_aggregator

        self.symbol = config.get("strategy_runner.symbol", "BTC/USDC")
        self.tick_interval = int(config.get("strategy_runner.tick_interval_s", 60))
        self.funding_lookback_hours = int(config.get("strategy_runner.funding_lookback_hours", 72))
        self.candle_lookback = int(config.get("strategy_runner.candle_lookback", 100))
        self.candle_timeframe = config.get("strategy_runner.candle_timeframe", "1h")
        self.log_hold_every_n = int(config.get("strategy_runner.log_hold_every_n", 10))

        self._running = False
        self._tick_count = 0
        self._strategy_name = strategy.get_metadata().get("name", "unknown")

        logger.info("StrategyRunner initialised: strategy=%s symbol=%s interval=%ds",
                     self._strategy_name, self.symbol, self.tick_interval)

    async def run(self, stop_event: asyncio.Event) -> None:
        """Main tick loop."""
        self._running = True

        # Restore strategy state from Redis if available
        self._restore_state()

        logger.info("StrategyRunner started for %s", self._strategy_name)

        while not stop_event.is_set():
            try:
                await self._tick()
            except Exception:
                logger.exception("Error in strategy tick")

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.tick_interval)
            except asyncio.TimeoutError:
                pass

        self._running = False
        self._checkpoint_state()
        logger.info("StrategyRunner stopped for %s", self._strategy_name)

    async def _tick(self) -> None:
        """Execute a single strategy tick."""
        self._tick_count += 1

        # 1. Build market state
        market_state = self._build_market_state()
        if market_state is None:
            logger.warning("Could not build market state — skipping tick")
            return

        # 2. Generate signal
        signal = self.strategy.on_tick(market_state)

        # Record signal for meta-strategy aggregation
        if self.signal_aggregator:
            self.signal_aggregator.record_signal(
                self._strategy_name, signal, market_state, 50.0)

        # 3. Process signal
        if signal.signal_type != SignalType.HOLD:
            signal.metadata["symbol"] = self.symbol

            # Apply allocation weight to signal size
            alloc_weight = self._get_allocation_weight()
            if alloc_weight < 1.0 and signal.size is not None:
                signal.size = signal.size * alloc_weight

            await self.execution_engine.process_signal(signal, self._strategy_name)

            self.decision_logger.log_signal(
                signal, market_state,
                hypothesis=f"{signal.signal_type.value} on {self.symbol} at {signal.price}",
                confidence=0.5,
            )

            self.redis.set_strategy_state(
                self._strategy_name,
                status="active",
                last_signal=signal.signal_type.value,
                last_signal_time=datetime.now(timezone.utc).isoformat(),
            )

            strategy_signal_count.labels(
                strategy_name=self._strategy_name,
                signal_type=signal.signal_type.value,
            ).inc()
        else:
            self.decision_logger.log_hold(
                market_state, "below_threshold",
                every_n=self.log_hold_every_n,
            )

        # 4. Periodic checkpoint
        if self._tick_count % 10 == 0:
            self._checkpoint_state()

    def _build_market_state(self) -> Optional[MarketState]:
        """Assemble MarketState from Redis (hot) and TimescaleDB (cold)."""
        try:
            price_data = self.redis.get_price(self.symbol)
            funding_data = self.redis.get_funding(self.symbol)

            if price_data is None or funding_data is None:
                logger.debug("Missing price or funding data for %s", self.symbol)
                return None

            now = datetime.now(timezone.utc)

            # Funding history from TimescaleDB
            funding_start = now - timedelta(hours=self.funding_lookback_hours)
            try:
                funding_history = self.timescale.query_funding_rates(
                    self.symbol, funding_start, now)
            except Exception:
                logger.debug("Failed to query funding history")
                funding_history = []

            # Recent candles from TimescaleDB
            candle_start = now - timedelta(hours=self.candle_lookback)
            try:
                recent_candles = self.timescale.query_candles(
                    self.symbol, self.candle_timeframe, candle_start, now)
            except Exception:
                logger.debug("Failed to query candles")
                recent_candles = []

            # Position and account from Redis
            position = self.redis.get_position(self.symbol)
            account = self.redis.get_account_state()
            orderbook = self.redis.get_orderbook(self.symbol)

            bid = float(price_data.get("bid", 0.0))
            ask = float(price_data.get("ask", 0.0))
            mid_price = (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0

            # Regime metadata for strategies that use it
            metadata = {}
            try:
                regime_data = self.redis.get_market_regime(self.symbol)
                if regime_data:
                    metadata["regime"] = regime_data.get("regime", "unknown")
                    metadata["regime_confidence"] = float(regime_data.get("confidence", 0.0))
            except Exception:
                logger.debug("Failed to get regime data")

            return MarketState(
                mark_price=float(funding_data.get("mark_price", mid_price)),
                mid_price=mid_price,
                bid=bid,
                ask=ask,
                funding_rate=float(funding_data.get("rate", 0.0)),
                premium=float(funding_data.get("premium", 0.0)),
                open_interest=0.0,  # Not in Redis funding data
                funding_history=funding_history,
                position=position,
                equity=float(account.get("total_equity", 0.0)) if account else 0.0,
                cash=float(account.get("cash", 0.0)) if account else 0.0,
                drawdown_pct=float(account.get("drawdown_pct", 0.0)) if account else 0.0,
                recent_candles=recent_candles,
                spread=float(orderbook.get("spread", 0.0)) if orderbook else 0.0,
                bid_depth=0.0,
                ask_depth=0.0,
                timestamp=now,
                metadata=metadata,
            )
        except Exception:
            logger.exception("Failed to build market state")
            return None

    def _restore_state(self) -> None:
        """Restore strategy state from Redis."""
        try:
            redis_state = self.redis.get_strategy_state(self._strategy_name)
            if redis_state and "strategy_state" in redis_state:
                state = json.loads(redis_state["strategy_state"])
                self.strategy.set_state(state)
                logger.info("Restored strategy state for %s", self._strategy_name)
        except Exception:
            logger.debug("No strategy state to restore for %s", self._strategy_name)

    def _checkpoint_state(self) -> None:
        """Save strategy state to Redis."""
        try:
            state = self.strategy.get_state()
            self.redis.set_strategy_state(
                self._strategy_name,
                status="active",
                last_signal="checkpoint",
                last_signal_time=datetime.now(timezone.utc).isoformat(),
            )
            # Store full strategy state as a separate Redis key
            self.redis._r.set(
                f"strategy:{self._strategy_name}:full_state",
                json.dumps(state),
            )
        except Exception:
            logger.exception("Failed to checkpoint strategy state")

    def _get_allocation_weight(self) -> float:
        """Read allocation weight from Redis. Defaults to 1.0."""
        try:
            raw = self.redis._r.get(f"strategy:{self._strategy_name}:allocation")
            if raw is not None:
                return float(raw)
        except Exception:
            logger.debug("Could not read allocation weight for %s", self._strategy_name)
        return 1.0

    async def stop(self) -> None:
        """Signal the runner to stop."""
        self._running = False
