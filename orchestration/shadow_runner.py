"""Shadow runner: mirrors StrategyRunner tick loop without executing orders."""

import asyncio
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Optional

from core.models import Signal
from core.types import SignalType
from strategy.live_strategy import LiveStrategy, MarketState

logger = logging.getLogger(__name__)


class ShadowRunner:
    """Mirrors StrategyRunner tick loop but records signals without execution.

    Tracks hypothetical PnL to evaluate shadow strategies in A/B tests.
    """

    def __init__(self, strategy: LiveStrategy, timescale, redis, config,
                 name: str):
        self.strategy = strategy
        self.timescale = timescale
        self.redis = redis
        self.config = config
        self.name = name

        self.symbol = config.get("strategy_runner.symbol", "BTC/USDC")
        self.tick_interval = int(config.get("strategy_runner.tick_interval_s", 60))
        self.funding_lookback_hours = int(config.get("strategy_runner.funding_lookback_hours", 72))
        self.candle_lookback = int(config.get("strategy_runner.candle_lookback", 100))
        self.candle_timeframe = config.get("strategy_runner.candle_timeframe", "1h")

        self._running = False
        self._tick_count = 0

        # Hypothetical position tracking
        self._position: Optional[dict] = None  # {side, entry_price, size}
        self._signals: list[dict] = []

        # Performance tracking
        self._cumulative_pnl = 0.0
        self._peak_pnl = 0.0
        self._max_drawdown = 0.0
        self._trade_count = 0
        self._wins = 0
        self._losses = 0
        self._pnl_history: list[float] = []

        logger.info("ShadowRunner initialised: name=%s symbol=%s interval=%ds",
                     self.name, self.symbol, self.tick_interval)

    async def run(self, stop_event: asyncio.Event) -> None:
        """Main tick loop — same rhythm as StrategyRunner.run()."""
        self._running = True
        logger.info("ShadowRunner started for %s", self.name)

        while not stop_event.is_set():
            try:
                await self._tick()
            except Exception:
                logger.exception("Error in shadow tick")

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.tick_interval)
            except asyncio.TimeoutError:
                pass

        self._running = False
        logger.info("ShadowRunner stopped for %s", self.name)

    async def _tick(self) -> None:
        """Execute a single shadow tick."""
        self._tick_count += 1

        market_state = self._build_market_state()
        if market_state is None:
            logger.warning("ShadowRunner: could not build market state — skipping tick")
            return

        signal = self.strategy.on_tick(market_state)

        # Record the signal
        self._signals.append({
            "tick": self._tick_count,
            "signal_type": signal.signal_type.value,
            "price": signal.price,
            "size": signal.size,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mid_price": market_state.mid_price,
        })

        # Process hypothetical position changes
        self._process_signal(signal, market_state)

    def _process_signal(self, signal: Signal, market_state: MarketState) -> None:
        """Update hypothetical position based on signal."""
        if signal.signal_type == SignalType.ENTER_LONG:
            if self._position is None:
                size = signal.size if signal.size else 1.0
                self._position = {
                    "side": "long",
                    "entry_price": market_state.mid_price,
                    "size": size,
                }

        elif signal.signal_type == SignalType.EXIT_LONG:
            if self._position and self._position["side"] == "long":
                pnl = (market_state.mid_price - self._position["entry_price"]) * self._position["size"]
                self._record_trade(pnl)
                self._position = None

        elif signal.signal_type == SignalType.ENTER_SHORT:
            if self._position is None:
                size = signal.size if signal.size else 1.0
                self._position = {
                    "side": "short",
                    "entry_price": market_state.mid_price,
                    "size": size,
                }

        elif signal.signal_type == SignalType.EXIT_SHORT:
            if self._position and self._position["side"] == "short":
                pnl = (self._position["entry_price"] - market_state.mid_price) * self._position["size"]
                self._record_trade(pnl)
                self._position = None

    def _record_trade(self, pnl: float) -> None:
        """Record a completed hypothetical trade."""
        self._cumulative_pnl += pnl
        self._trade_count += 1
        self._pnl_history.append(pnl)

        if pnl > 0:
            self._wins += 1
        else:
            self._losses += 1

        # Update peak and drawdown
        if self._cumulative_pnl > self._peak_pnl:
            self._peak_pnl = self._cumulative_pnl

        if self._peak_pnl > 0:
            dd = (self._peak_pnl - self._cumulative_pnl) / self._peak_pnl * 100
            if dd > self._max_drawdown:
                self._max_drawdown = dd

    def get_performance_summary(self) -> dict:
        """Return performance summary of shadow trading."""
        win_rate = (self._wins / self._trade_count * 100) if self._trade_count > 0 else 0.0

        # Estimate Sharpe from PnL history
        sharpe = 0.0
        if len(self._pnl_history) >= 2:
            mean_pnl = sum(self._pnl_history) / len(self._pnl_history)
            variance = sum((p - mean_pnl) ** 2 for p in self._pnl_history) / (len(self._pnl_history) - 1)
            std_pnl = math.sqrt(variance) if variance > 0 else 0.0
            if std_pnl > 0:
                sharpe = mean_pnl / std_pnl * math.sqrt(365)

        return {
            "total_pnl": round(self._cumulative_pnl, 4),
            "trade_count": self._trade_count,
            "win_rate": round(win_rate, 2),
            "max_drawdown": round(self._max_drawdown, 2),
            "sharpe": round(sharpe, 4),
            "wins": self._wins,
            "losses": self._losses,
        }

    def get_signals(self) -> list[dict]:
        """Return all recorded signals."""
        return list(self._signals)

    def _build_market_state(self) -> Optional[MarketState]:
        """Assemble MarketState from Redis (hot) and TimescaleDB (cold).

        Reuses the same data queries as StrategyRunner._build_market_state().
        """
        try:
            price_data = self.redis.get_price(self.symbol)
            funding_data = self.redis.get_funding(self.symbol)

            if price_data is None or funding_data is None:
                return None

            now = datetime.now(timezone.utc)

            funding_start = now - timedelta(hours=self.funding_lookback_hours)
            try:
                funding_history = self.timescale.query_funding_rates(
                    self.symbol, funding_start, now)
            except Exception:
                funding_history = []

            candle_start = now - timedelta(hours=self.candle_lookback)
            try:
                recent_candles = self.timescale.query_candles(
                    self.symbol, self.candle_timeframe, candle_start, now)
            except Exception:
                recent_candles = []

            position = self.redis.get_position(self.symbol)
            account = self.redis.get_account_state()
            orderbook = self.redis.get_orderbook(self.symbol)

            bid = float(price_data.get("bid", 0.0))
            ask = float(price_data.get("ask", 0.0))
            mid_price = (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0

            metadata = {}
            try:
                regime_data = self.redis.get_market_regime(self.symbol)
                if regime_data:
                    metadata["regime"] = regime_data.get("regime", "unknown")
                    metadata["regime_confidence"] = float(regime_data.get("confidence", 0.0))
            except Exception:
                pass

            return MarketState(
                mark_price=float(funding_data.get("mark_price", mid_price)),
                mid_price=mid_price,
                bid=bid,
                ask=ask,
                funding_rate=float(funding_data.get("rate", 0.0)),
                premium=float(funding_data.get("premium", 0.0)),
                open_interest=0.0,
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
            logger.exception("ShadowRunner: failed to build market state")
            return None
