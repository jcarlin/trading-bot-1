"""Tests for the ExecutionEngine and related components."""

import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from core.models import Fill, Order, Position, Signal
from core.types import OrderType, Side, SignalType


def _make_config():
    config = MagicMock()
    config.get.side_effect = lambda key, default=None: {
        "symbol": "BTC/USDT",
        "symbols": ["BTC/USDT"],
        "reconciler.auto_correct": False,
    }.get(key, default)
    config.get_section.return_value = {
        "max_position_pct_per_asset": 0.05,
        "max_portfolio_exposure_pct": 1.50,
        "max_strategy_drawdown_pct": 0.05,
        "max_portfolio_drawdown_pct": 0.15,
        "max_single_trade_loss_pct": 0.01,
        "min_cash_reserve_pct": 0.20,
    }
    return config


def _make_signal(signal_type=SignalType.ENTER_LONG, price=100.0, symbol="BTC/USDT"):
    return Signal(
        signal_type=signal_type,
        price=price,
        timestamp=datetime.now(timezone.utc),
        stop_loss=95.0,
        take_profit=110.0,
        size=0.01,
        metadata={"symbol": symbol},
    )


def _make_fill(order_id="order-123", symbol="BTC/USDT", side=Side.BUY):
    return Fill(
        order_id=order_id,
        symbol=symbol,
        side=side,
        quantity=0.01,
        fill_price=100.0,
        timestamp=datetime.now(timezone.utc),
        commission=0.001,
    )


def _make_position(symbol="BTC/USDT", side=Side.BUY):
    return Position(
        symbol=symbol,
        side=side,
        entry_price=100.0,
        quantity=0.01,
        entry_time=datetime.now(timezone.utc),
    )


# Patch all Prometheus metrics so they don't collide across tests
_METRIC_PATCHES = [
    "execution.engine.orders_placed_total",
    "execution.engine.orders_filled_total",
    "execution.engine.fill_latency_seconds",
    "execution.engine.slippage_bps",
    "execution.engine.orders_rejected_total",
    "execution.engine.portfolio_equity_usd",
    "execution.engine.portfolio_drawdown_pct",
    "execution.engine.portfolio_unrealized_pnl_usd",
    "execution.risk_controls.risk_check_passed_total",
    "execution.risk_controls.risk_check_failed_total",
    "execution.kill_switch.circuit_breaker_status",
    "execution.kill_switch.kill_switch_activations_total",
]


def _apply_metric_patches(test_case):
    """Start all metric patches and return a cleanup list."""
    patchers = []
    for target in _METRIC_PATCHES:
        p = patch(target, MagicMock())
        p.start()
        patchers.append(p)
    return patchers


class TestExecutionEngine(unittest.IsolatedAsyncioTestCase):
    """Async tests for ExecutionEngine."""

    def setUp(self):
        self._patchers = _apply_metric_patches(self)

        self.config = _make_config()
        self.exchange = MagicMock()
        self.risk_manager = MagicMock()
        self.timescale = MagicMock()
        self.redis = MagicMock()

        # Default exchange behaviour
        self.exchange.get_positions.return_value = []
        self.exchange.get_account_state.return_value = {
            "equity": 100_000.0,
            "cash": 80_000.0,
            "position_value": 20_000.0,
            "unrealized_pnl": 500.0,
            "realized_pnl": 100.0,
        }
        self.exchange.get_open_orders.return_value = []
        self.exchange.place_order.return_value = _make_fill()

        # Default risk manager behaviour
        self.risk_manager.validate_signal.return_value = True
        self.risk_manager.calculate_position_size.return_value = 0.01

        # Default Redis behaviour
        self.redis.get_account_state.return_value = {
            "total_equity": 100_000.0,
            "cash": 80_000.0,
            "position_value": 20_000.0,
            "unrealized_pnl": 500.0,
            "drawdown_pct": 0.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.redis.get_circuit_breaker.return_value = None

        from execution.risk_controls import ExecutionRiskControls
        from execution.engine import ExecutionEngine

        self.risk_controls = ExecutionRiskControls(self.config)
        self.engine = ExecutionEngine(
            exchange=self.exchange,
            risk_manager=self.risk_manager,
            risk_controls=self.risk_controls,
            timescale=self.timescale,
            redis=self.redis,
            config=self.config,
        )

    def tearDown(self):
        for p in self._patchers:
            p.stop()

    # ------------------------------------------------------------------
    # process_signal
    # ------------------------------------------------------------------

    async def test_process_signal_success(self):
        """Valid signal should flow through to a fill."""
        signal = _make_signal()
        fill = await self.engine.process_signal(signal, "test_strategy")

        self.assertIsNotNone(fill)
        self.assertEqual(fill.symbol, "BTC/USDT")
        self.exchange.place_order.assert_called_once()
        self.timescale.insert_order.assert_called_once()
        self.timescale.insert_fill.assert_called_once()
        self.timescale.insert_decision.assert_called()

    async def test_process_signal_risk_rejection(self):
        """Signal rejected by risk manager should return None."""
        self.risk_manager.validate_signal.return_value = False
        signal = _make_signal()

        fill = await self.engine.process_signal(signal, "test_strategy")

        self.assertIsNone(fill)
        self.exchange.place_order.assert_not_called()

    async def test_process_signal_kill_switch_active(self):
        """Signal should be rejected when the kill switch is active."""
        self.redis.get_circuit_breaker.return_value = {
            "active": True,
            "reason": "test halt",
            "activated_at": datetime.now(timezone.utc).isoformat(),
        }
        signal = _make_signal()

        fill = await self.engine.process_signal(signal, "test_strategy")

        self.assertIsNone(fill)
        self.exchange.place_order.assert_not_called()

    # ------------------------------------------------------------------
    # execute_order
    # ------------------------------------------------------------------

    async def test_execute_order(self):
        """Order should be placed and fill returned with DB/Redis updates."""
        order = Order(
            symbol="BTC/USDT",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=0.01,
            price=100.0,
            timestamp=datetime.now(timezone.utc),
            order_id="order-456",
        )

        fill = await self.engine.execute_order(order, "test_strategy")

        self.assertIsNotNone(fill)
        self.exchange.place_order.assert_called_once_with(order)
        self.timescale.insert_order.assert_called_once()
        self.timescale.update_order_status.assert_called_with("order-456", "filled")
        self.timescale.insert_fill.assert_called_once()

    # ------------------------------------------------------------------
    # update_equity
    # ------------------------------------------------------------------

    async def test_update_equity(self):
        """Equity update should persist snapshot and update metrics."""
        await self.engine.update_equity()

        self.exchange.get_account_state.assert_called_once()
        self.timescale.insert_equity_snapshot.assert_called_once()
        self.redis.set_account_state.assert_called_once()

        # Verify snapshot content
        snapshot = self.timescale.insert_equity_snapshot.call_args[0][0]
        self.assertEqual(snapshot["total_equity"], 100_000.0)
        self.assertEqual(snapshot["cash"], 80_000.0)

    # ------------------------------------------------------------------
    # flatten_all
    # ------------------------------------------------------------------

    async def test_flatten_all(self):
        """flatten_all should activate the kill switch."""
        await self.engine.flatten_all()

        self.timescale.insert_system_event.assert_called()
        self.redis.set_circuit_breaker.assert_called()

        # Verify circuit breaker was set to active
        cb_call = self.redis.set_circuit_breaker.call_args
        self.assertTrue(cb_call[1].get("active", cb_call[0][0] if cb_call[0] else False))

    # ------------------------------------------------------------------
    # reconcile_positions
    # ------------------------------------------------------------------

    async def test_reconcile_positions(self):
        """reconcile_positions should delegate to the reconciler."""
        self.exchange.get_positions.return_value = []
        self.redis._r = MagicMock()
        self.redis._r.keys.return_value = []
        self.redis.get_position.return_value = None

        discrepancies = await self.engine.reconcile_positions()

        self.assertIsInstance(discrepancies, list)

    # ------------------------------------------------------------------
    # decision logging
    # ------------------------------------------------------------------

    async def test_decision_logging(self):
        """Verify that decisions are logged with correct fields."""
        signal = _make_signal()
        await self.engine.process_signal(signal, "test_strategy")

        self.timescale.insert_decision.assert_called()
        decision = self.timescale.insert_decision.call_args[0][0]

        self.assertIn("time", decision)
        self.assertIn("decision_type", decision)
        self.assertIn("strategy", decision)
        self.assertIn("context", decision)
        self.assertIn("hypothesis", decision)
        self.assertIn("action", decision)
        self.assertEqual(decision["strategy"], "test_strategy")


class TestKillSwitch(unittest.IsolatedAsyncioTestCase):
    """Tests for the KillSwitch component."""

    def setUp(self):
        self._patchers = _apply_metric_patches(self)

        self.exchange = MagicMock()
        self.redis = MagicMock()
        self.timescale = MagicMock()
        self.config = _make_config()

        self.exchange.get_positions.return_value = [_make_position()]
        self.exchange.get_open_orders.return_value = [{"order_id": "o1"}]
        self.exchange.cancel_order.return_value = True
        self.exchange.place_order.return_value = _make_fill()

        from execution.kill_switch import KillSwitch
        self.ks = KillSwitch(self.exchange, self.redis, self.timescale, self.config)

    def tearDown(self):
        for p in self._patchers:
            p.stop()

    async def test_activate(self):
        """Activate should cancel orders, flatten positions, update state."""
        await self.ks.activate("test reason")

        self.timescale.insert_system_event.assert_called()
        self.exchange.cancel_order.assert_called()
        self.exchange.place_order.assert_called()
        self.redis.set_circuit_breaker.assert_called()

        cb_call = self.redis.set_circuit_breaker.call_args
        self.assertTrue(cb_call[1].get("active", cb_call[0][0] if cb_call[0] else False))

    async def test_deactivate(self):
        """Deactivate should clear circuit breaker and log event."""
        await self.ks.deactivate()

        self.redis.set_circuit_breaker.assert_called()
        self.timescale.insert_system_event.assert_called()

    def test_is_active_false(self):
        """is_active should return False when no state exists."""
        self.redis.get_circuit_breaker.return_value = None
        self.assertFalse(self.ks.is_active())

    def test_is_active_true(self):
        """is_active should return True when active in Redis."""
        self.redis.get_circuit_breaker.return_value = {
            "active": True,
            "reason": "test",
            "activated_at": "2026-01-01",
        }
        self.assertTrue(self.ks.is_active())


class TestPositionReconciler(unittest.IsolatedAsyncioTestCase):
    """Tests for the PositionReconciler."""

    def setUp(self):
        self._patchers = _apply_metric_patches(self)

        self.exchange = MagicMock()
        self.redis = MagicMock()
        self.timescale = MagicMock()
        self.config = _make_config()

        self.redis._r = MagicMock()
        self.redis._r.keys.return_value = []

    def tearDown(self):
        for p in self._patchers:
            p.stop()

    async def test_no_discrepancies(self):
        """No positions on either side should yield empty list."""
        self.exchange.get_positions.return_value = []
        self.redis.get_position.return_value = None

        from execution.reconciler import PositionReconciler
        rec = PositionReconciler(self.exchange, self.redis, self.timescale, self.config)

        result = await rec.reconcile()

        self.assertEqual(result, [])

    async def test_exchange_has_position_local_missing(self):
        """Position on exchange but not locally should be a critical discrepancy."""
        pos = _make_position()
        self.exchange.get_positions.return_value = [pos]
        self.redis.get_position.return_value = None

        from execution.reconciler import PositionReconciler
        rec = PositionReconciler(self.exchange, self.redis, self.timescale, self.config)

        result = await rec.reconcile()

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].field, "existence")
        self.assertEqual(result[0].severity, "critical")

    async def test_matching_positions_no_discrepancy(self):
        """Matching positions should yield no discrepancies."""
        pos = _make_position()
        self.exchange.get_positions.return_value = [pos]
        self.redis.get_position.return_value = {
            "entry_price": 100.0,
            "quantity": 0.01,
            "side": "buy",
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "unrealized_pnl": 0.0,
        }

        from execution.reconciler import PositionReconciler
        rec = PositionReconciler(self.exchange, self.redis, self.timescale, self.config)

        result = await rec.reconcile()

        self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main()
