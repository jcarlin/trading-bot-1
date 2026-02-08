"""Core execution engine: translates signals into orders with full risk controls."""

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from core.models import Fill, Order, Position, Signal
from core.types import OrderType, Side, SignalType
from monitoring.metrics import (
    algo_execution_count,
    algo_fill_rate,
    algo_slippage_bps,
    fill_latency_seconds,
    orders_filled_total,
    orders_placed_total,
    orders_rejected_total,
    portfolio_drawdown_pct,
    portfolio_equity_usd,
    portfolio_unrealized_pnl_usd,
    slippage_bps,
)

from .kill_switch import KillSwitch
from .reconciler import PositionReconciler
from .risk_controls import ExecutionRiskControls

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """Orchestrates the full signal-to-fill pipeline with risk controls."""

    def __init__(
        self,
        exchange,
        risk_manager,
        risk_controls: ExecutionRiskControls,
        timescale,
        redis,
        config,
        algorithm_selector=None,
        notification_dispatcher=None,
    ):
        self.exchange = exchange
        self.risk_manager = risk_manager
        self.risk_controls = risk_controls
        self.timescale = timescale
        self.redis = redis
        self.config = config
        self.algorithm_selector = algorithm_selector
        self.notification_dispatcher = notification_dispatcher

        self.reconciler = PositionReconciler(exchange, redis, timescale, config)
        self.kill_switch = KillSwitch(exchange, redis, timescale, config)

        self._peak_equity: float = 0.0

        logger.info("ExecutionEngine initialised")

    # ------------------------------------------------------------------
    # Signal processing
    # ------------------------------------------------------------------

    async def process_signal(
        self, signal: Signal, strategy_name: str
    ) -> Optional[Fill]:
        """Process a trading signal through the full execution pipeline.

        Steps:
            1. Check kill switch.
            2. Validate signal via RiskManager.
            3. Calculate position size.
            4. Build Order from Signal.
            5. Validate order via ExecutionRiskControls.
            6. Check portfolio limits.
            7. Execute order.
            8. Log decision.

        Returns:
            Fill if the order was executed, None otherwise.
        """
        # 1. Kill switch check
        if self.kill_switch.is_active():
            logger.warning("Signal rejected — kill switch active")
            orders_rejected_total.labels(reason="kill_switch_active").inc()
            self._log_decision(
                decision_type="signal_rejected",
                strategy=strategy_name,
                context={"signal": str(signal.signal_type.value)},
                action={"rejected": True},
                hypothesis="Kill switch is active; no trades allowed",
            )
            return None

        # 2. Validate signal
        positions = self.exchange.get_positions()
        has_position = any(p.symbol == signal.metadata.get("symbol") for p in positions)

        if not self.risk_manager.validate_signal(signal, has_position):
            logger.info("Signal rejected by risk manager validation")
            orders_rejected_total.labels(reason="signal_validation").inc()
            return None

        # 3. Calculate position size
        symbol = signal.metadata.get("symbol", self.config.get("symbol", ""))
        current_price = signal.price
        equity = self._get_equity()

        quantity = self.risk_manager.calculate_position_size(
            signal, equity, current_price
        )
        if quantity <= 0:
            logger.info("Calculated position size is zero — skipping")
            return None

        # Use signal.size if explicitly set
        if signal.size is not None and signal.size > 0:
            quantity = signal.size

        # 4. Create Order from Signal
        side = self._signal_to_side(signal.signal_type)
        if side is None:
            return None

        order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=current_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            timestamp=datetime.now(timezone.utc),
            order_id=str(uuid.uuid4()),
        )

        # 5. Validate order via risk controls
        ok, reason = self.risk_controls.validate_order(order, equity, positions)
        if not ok:
            logger.warning("Order rejected by risk controls: %s", reason)
            orders_rejected_total.labels(reason="risk_controls").inc()
            self._log_decision(
                decision_type="order_rejected",
                strategy=strategy_name,
                context={"symbol": symbol, "side": side.value, "qty": quantity},
                action={"rejected": True, "reason": reason},
                hypothesis=f"Risk controls blocked: {reason}",
            )
            return None

        # 6. Check portfolio limits
        ok, reason = self.risk_controls.check_portfolio_limits(
            equity, self._peak_equity, positions
        )
        if not ok:
            logger.warning("Portfolio limits breached: %s", reason)
            orders_rejected_total.labels(reason="portfolio_limits").inc()
            if self.risk_controls.should_halt():
                await self.flatten_all()
            return None

        # 7. Execute order (via algorithm if selected, otherwise market)
        fill = None
        if self.algorithm_selector is not None:
            market_state = self._build_market_state(symbol)
            algo_name, algo = self.algorithm_selector.select(
                order, market_state, signal.metadata
            )
            if algo is not None:
                fill = await self.execute_with_algorithm(
                    order, algo, algo_name, strategy_name
                )

        if fill is None:
            fill = await self.execute_order(order, strategy_name)

        # 8. Log decision
        self._log_decision(
            decision_type="signal_executed",
            strategy=strategy_name,
            context={"symbol": symbol, "signal": signal.signal_type.value},
            action={
                "order_id": order.order_id,
                "side": side.value,
                "quantity": quantity,
                "price": current_price,
            },
            hypothesis=(
                f"Executing {signal.signal_type.value} on {symbol} "
                f"at {current_price} qty={quantity}"
            ),
        )

        return fill

    # ------------------------------------------------------------------
    # Algorithm-based execution
    # ------------------------------------------------------------------

    async def execute_with_algorithm(
        self, order: Order, algorithm, algo_name: str, strategy_name: str = ""
    ) -> Optional[Fill]:
        """Execute an order using an execution algorithm.

        Returns the first Fill from the algorithm's child orders,
        or None if the algorithm produced no fills.
        """
        algo_config = {}
        if self.algorithm_selector:
            algo_config = self.algorithm_selector.get_algo_config(algo_name)

        algo_execution_count.labels(
            algo_name=algo_name, symbol=order.symbol
        ).inc()

        submit_time = time.monotonic()
        try:
            child_fills = await algorithm.execute(order, self.exchange, algo_config)
        except Exception:
            logger.exception(
                "Algorithm %s failed for order %s", algo_name, order.order_id
            )
            return None

        if not child_fills:
            return None

        fill_time = time.monotonic()

        # Compute VWAP across all child fills
        total_qty = sum(f.get("quantity", 0) for f in child_fills if f)
        total_cost = sum(
            f.get("price", 0) * f.get("quantity", 0) for f in child_fills if f
        )
        total_fee = sum(f.get("fee", 0) for f in child_fills if f)

        if total_qty <= 0:
            return None

        vwap = total_cost / total_qty

        # Metrics
        fill_rate = total_qty / order.quantity * 100 if order.quantity > 0 else 0
        algo_fill_rate.labels(algo_name=algo_name).set(fill_rate)

        if order.price and order.price > 0:
            slippage_improvement = (
                abs(order.price - vwap) / order.price * 10000
            )
            algo_slippage_bps.labels(algo_name=algo_name).observe(
                slippage_improvement
            )

        # Build a Fill object from the aggregated result
        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=total_qty,
            fill_price=vwap,
            timestamp=datetime.now(timezone.utc),
            commission=total_fee,
        )

        # Record in DB
        latency = fill_time - submit_time
        self.timescale.update_order_status(order.order_id, "filled")
        try:
            self.timescale.insert_fill({
                "time": fill.timestamp,
                "fill_id": str(uuid.uuid4()),
                "order_id": fill.order_id,
                "symbol": fill.symbol,
                "side": fill.side.value,
                "quantity": fill.quantity,
                "price": fill.fill_price,
                "commission": fill.commission,
            })
        except Exception:
            logger.exception("Failed to insert algo fill to DB")

        orders_filled_total.labels(
            symbol=fill.symbol, side=fill.side.value
        ).inc()
        fill_latency_seconds.labels(symbol=fill.symbol).observe(latency)

        logger.info(
            "Algo %s filled: %s %s %s qty=%s vwap=%s (%d children, %.3fs)",
            algo_name, fill.order_id, fill.side.value, fill.symbol,
            fill.quantity, vwap, len(child_fills), latency,
        )

        # Dispatch trade notification
        if self.notification_dispatcher:
            try:
                import asyncio
                asyncio.ensure_future(self.notification_dispatcher.dispatch(
                    message=f"Algo fill ({algo_name}): {fill.side.value} {fill.symbol} qty={fill.quantity} vwap={vwap:.4f}",
                    level="trade",
                    event_type="algo_fill",
                    metadata={"order_id": fill.order_id, "algo": algo_name,
                              "symbol": fill.symbol, "strategy": strategy_name},
                ))
            except Exception:
                logger.debug("Failed to dispatch algo trade notification")

        return fill

    def _build_market_state(self, symbol: str) -> dict:
        """Gather market state for algorithm selection."""
        state = {}
        try:
            acct = self.redis.get_account_state()
            if acct:
                state["regime"] = acct.get("regime", "")
        except Exception:
            pass
        try:
            ticker = self.exchange.get_ticker(symbol)
            state["mid"] = ticker.get("mid", 0)
        except Exception:
            pass
        # avg_volume could be loaded from Redis/TimescaleDB; default to 0
        state.setdefault("avg_volume", 0)
        return state

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    async def execute_order(
        self, order: Order, strategy_name: str = ""
    ) -> Optional[Fill]:
        """Place an order on the exchange and record the fill.

        Returns:
            Fill object if successful, None on error.
        """
        now = datetime.now(timezone.utc)

        # 1. Insert order to DB
        try:
            self.timescale.insert_order({
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "type": order.order_type.value,
                "quantity": order.quantity,
                "price": order.price,
                "stop_loss": order.stop_loss,
                "take_profit": order.take_profit,
                "status": "pending",
                "strategy_name": strategy_name,
            })
        except Exception:
            logger.exception("Failed to insert order to DB")

        # 2. Submit to exchange
        orders_placed_total.labels(
            symbol=order.symbol,
            side=order.side.value,
            type=order.order_type.value,
        ).inc()

        submit_time = time.monotonic()
        try:
            fill = self.exchange.place_order(order)
        except Exception:
            logger.exception("Exchange order placement failed for %s", order.order_id)
            self.timescale.update_order_status(order.order_id, "failed")
            return None

        fill_time = time.monotonic()
        latency = fill_time - submit_time

        # 3. Update order status
        self.timescale.update_order_status(order.order_id, "filled")

        # 4. Insert fill to DB
        try:
            self.timescale.insert_fill({
                "time": fill.timestamp,
                "fill_id": str(uuid.uuid4()),
                "order_id": fill.order_id,
                "symbol": fill.symbol,
                "side": fill.side.value,
                "quantity": fill.quantity,
                "price": fill.fill_price,
                "commission": fill.commission,
            })
        except Exception:
            logger.exception("Failed to insert fill to DB")

        # 5. Update position in Redis
        try:
            if fill.side in (Side.BUY,):
                self.redis.set_position(
                    symbol=fill.symbol,
                    entry_price=fill.fill_price,
                    quantity=fill.quantity,
                    side=fill.side.value,
                    entry_time=fill.timestamp.isoformat(),
                )
            else:
                # On sell, clear the position
                self.redis._r.delete(f"position:{fill.symbol}")
        except Exception:
            logger.exception("Failed to update Redis position")

        # 6. Update monitoring metrics
        orders_filled_total.labels(
            symbol=fill.symbol,
            side=fill.side.value,
        ).inc()

        fill_latency_seconds.labels(symbol=fill.symbol).observe(latency)

        if order.price and order.price > 0:
            slip = abs(fill.fill_price - order.price) / order.price * 10_000
            slippage_bps.labels(symbol=fill.symbol).observe(slip)

        logger.info(
            "Order filled: %s %s %s qty=%s @ %s (latency=%.3fs)",
            fill.order_id, fill.side.value, fill.symbol,
            fill.quantity, fill.fill_price, latency,
        )

        # Dispatch trade notification
        if self.notification_dispatcher:
            try:
                import asyncio
                asyncio.ensure_future(self.notification_dispatcher.dispatch(
                    message=f"Fill: {fill.side.value} {fill.symbol} qty={fill.quantity} @ {fill.fill_price}",
                    level="trade",
                    event_type="fill",
                    metadata={"order_id": fill.order_id, "symbol": fill.symbol,
                              "side": fill.side.value, "strategy": strategy_name},
                ))
            except Exception:
                logger.debug("Failed to dispatch trade notification")

        return fill

    # ------------------------------------------------------------------
    # Emergency flatten
    # ------------------------------------------------------------------

    async def flatten_all(self) -> None:
        """Emergency: activate the kill switch to flatten everything."""
        if self.notification_dispatcher:
            try:
                await self.notification_dispatcher.dispatch(
                    message="EMERGENCY FLATTEN: Kill switch activated by ExecutionEngine",
                    level="critical",
                    event_type="flatten_all",
                )
            except Exception:
                logger.debug("Failed to dispatch flatten notification")
        await self.kill_switch.activate("ExecutionEngine triggered emergency flatten")

    # ------------------------------------------------------------------
    # Equity tracking
    # ------------------------------------------------------------------

    async def update_equity(self) -> None:
        """Fetch account state, compute equity metrics, persist, and check limits."""
        try:
            state = self.exchange.get_account_state()
        except Exception:
            logger.exception("Failed to fetch account state")
            return

        equity = float(state.get("equity", 0.0))
        cash = float(state.get("cash", 0.0))
        position_value = float(state.get("position_value", 0.0))
        unrealized_pnl = float(state.get("unrealized_pnl", 0.0))
        realized_pnl = float(state.get("realized_pnl", 0.0))

        if equity > self._peak_equity:
            self._peak_equity = equity

        drawdown_pct = 0.0
        if self._peak_equity > 0:
            drawdown_pct = (self._peak_equity - equity) / self._peak_equity * 100

        now = datetime.now(timezone.utc)

        # Persist to TimescaleDB
        self.timescale.insert_equity_snapshot({
            "time": now,
            "total_equity": equity,
            "cash": cash,
            "position_value": position_value,
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": realized_pnl,
            "peak_equity": self._peak_equity,
            "drawdown_pct": drawdown_pct,
        })

        # Persist to Redis
        self.redis.set_account_state(
            total_equity=equity,
            cash=cash,
            position_value=position_value,
            unrealized_pnl=unrealized_pnl,
            drawdown_pct=drawdown_pct,
            timestamp=now.isoformat(),
        )

        # Update Prometheus gauges
        portfolio_equity_usd.set(equity)
        portfolio_drawdown_pct.set(drawdown_pct)
        portfolio_unrealized_pnl_usd.set(unrealized_pnl)

        # Check portfolio limits — activate kill switch if breached
        positions = self.exchange.get_positions()
        ok, reason = self.risk_controls.check_portfolio_limits(
            equity, self._peak_equity, positions
        )
        if not ok:
            logger.warning("Portfolio limits breached during equity update: %s", reason)
            if self.risk_controls.should_halt():
                await self.kill_switch.activate(reason)

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    async def reconcile_positions(self):
        """Delegate to the PositionReconciler."""
        return await self.reconciler.reconcile()

    # ------------------------------------------------------------------
    # Decision logging
    # ------------------------------------------------------------------

    def _log_decision(
        self,
        decision_type: str,
        strategy: str = "",
        context: dict = None,
        action: dict = None,
        hypothesis: str = "",
        alternatives: list = None,
        confidence: float = 0.0,
    ) -> None:
        """Insert a decision record into the decision_log table."""
        try:
            self.timescale.insert_decision({
                "time": datetime.now(timezone.utc),
                "decision_type": decision_type,
                "strategy": strategy,
                "context": context or {},
                "hypothesis": hypothesis,
                "action": action or {},
                "alternatives": alternatives or [],
                "confidence": confidence,
                "outcome": {},
            })
        except Exception:
            logger.exception("Failed to log decision")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_equity(self) -> float:
        """Best-effort equity fetch from Redis, falling back to exchange."""
        state = self.redis.get_account_state()
        if state:
            return state.get("total_equity", 0.0)
        try:
            ex_state = self.exchange.get_account_state()
            return float(ex_state.get("equity", 0.0))
        except Exception:
            return 0.0

    @staticmethod
    def _signal_to_side(signal_type: SignalType) -> Optional[Side]:
        """Map a SignalType to an order Side."""
        if signal_type in (SignalType.ENTER_LONG, SignalType.EXIT_SHORT):
            return Side.BUY
        if signal_type in (SignalType.ENTER_SHORT, SignalType.EXIT_LONG):
            return Side.SELL
        return None
