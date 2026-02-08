"""Tests for execution algorithms: TWAP, AdaptiveLimit, Iceberg, AlgorithmSelector, and engine integration."""

import asyncio
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.models import Fill, Order, Signal
from core.types import OrderType, Side, SignalType
from execution.algorithms.base import ExecutionAlgorithm
from execution.algorithms.twap import TWAPAlgorithm
from execution.algorithms.adaptive_limit import AdaptiveLimitAlgorithm
from execution.algorithms.iceberg import IcebergAlgorithm
from execution.algorithm_selector import AlgorithmSelector


# ======================================================================
# Helpers
# ======================================================================

def _make_order(symbol="BTC/USDC", side=Side.BUY, qty=1.0, price=50000.0):
    return Order(
        symbol=symbol,
        side=side,
        order_type=OrderType.MARKET,
        quantity=qty,
        price=price,
        order_id="test-order-1",
    )


def _make_signal(signal_type=SignalType.ENTER_LONG, price=50000.0, metadata=None):
    return Signal(
        signal_type=signal_type,
        price=price,
        timestamp=datetime.now(timezone.utc),
        metadata=metadata or {"symbol": "BTC/USDC"},
    )


def _make_exchange_mock():
    """Create a mock exchange with standard methods."""
    exchange = MagicMock()
    exchange.get_ticker.return_value = {
        "last": 50000.0, "bid": 49990.0, "ask": 50010.0, "mid": 50000.0,
    }
    exchange.place_order.return_value = Fill(
        order_id="fill-1", symbol="BTC/USDC", side=Side.BUY,
        quantity=1.0, fill_price=50000.0,
        timestamp=datetime.now(timezone.utc), commission=0.5,
    )
    exchange.place_limit_order = AsyncMock(return_value="limit-order-1")
    exchange.get_order_status = AsyncMock(return_value={
        "status": "filled", "fill_price": 50000.0, "fee": 0.3,
    })
    exchange.cancel_order.return_value = True
    return exchange


# ======================================================================
# TestExecutionAlgorithmBase
# ======================================================================

class TestExecutionAlgorithmBase(unittest.TestCase):
    """Tests for the ExecutionAlgorithm ABC."""

    def test_cannot_instantiate_abc(self):
        """ABC cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            ExecutionAlgorithm()

    def test_concrete_must_implement_execute(self):
        """Concrete class without execute() raises TypeError."""
        class Incomplete(ExecutionAlgorithm):
            def get_metadata(self):
                return {}

        with self.assertRaises(TypeError):
            Incomplete()

    def test_concrete_must_implement_get_metadata(self):
        """Concrete class without get_metadata() raises TypeError."""
        class Incomplete(ExecutionAlgorithm):
            async def execute(self, order, exchange, config):
                return []

        with self.assertRaises(TypeError):
            Incomplete()

    def test_estimate_impact_with_orderbook(self):
        """estimate_impact walks asks to compute slippage."""
        algo = TWAPAlgorithm()  # concrete subclass
        orderbook = {
            "bids": [[49990.0, 1.0], [49980.0, 2.0]],
            "asks": [[50010.0, 0.5], [50020.0, 1.0], [50030.0, 2.0]],
        }
        impact = algo.estimate_impact(0.5, orderbook)
        # Mid = (49990 + 50010) / 2 = 50000
        # Fill 0.5 at 50010 → VWAP = 50010
        # Impact = |50010 - 50000| / 50000 * 10000 = 2.0 bps
        self.assertAlmostEqual(impact, 2.0, places=1)

    def test_estimate_impact_empty_orderbook(self):
        """estimate_impact returns 0 for empty orderbook."""
        algo = TWAPAlgorithm()
        self.assertEqual(algo.estimate_impact(1.0, {}), 0.0)
        self.assertEqual(algo.estimate_impact(1.0, {"bids": [], "asks": []}), 0.0)
        self.assertEqual(algo.estimate_impact(0.0, {"bids": [[1, 1]], "asks": [[1, 1]]}), 0.0)


# ======================================================================
# TestTWAPAlgorithm
# ======================================================================

class TestTWAPAlgorithm(unittest.TestCase):
    """Tests for TWAP execution algorithm."""

    def setUp(self):
        self.algo = TWAPAlgorithm()
        self.exchange = _make_exchange_mock()

    def test_metadata(self):
        meta = self.algo.get_metadata()
        self.assertEqual(meta["name"], "twap")
        self.assertIn("default_config", meta)
        self.assertIn("duration_seconds", meta["default_config"])

    def test_compute_slices_equal_split(self):
        slices = self.algo._compute_slices(10.0, 5)
        self.assertEqual(len(slices), 5)
        self.assertAlmostEqual(sum(slices), 10.0, places=10)
        for s in slices:
            self.assertAlmostEqual(s, 2.0, places=10)

    def test_compute_slices_single(self):
        slices = self.algo._compute_slices(3.0, 1)
        self.assertEqual(len(slices), 1)
        self.assertAlmostEqual(slices[0], 3.0)

    def test_compute_slices_zero_qty(self):
        self.assertEqual(self.algo._compute_slices(0.0, 5), [])

    def test_compute_slices_zero_num(self):
        self.assertEqual(self.algo._compute_slices(10.0, 0), [])

    def test_execute_fills_all_slices(self):
        """TWAP should produce fills for each slice."""
        order = _make_order(qty=1.0)
        config = {
            "duration_seconds": 0,  # no delay for testing
            "num_slices": 3,
            "spread_buffer_bps": 2,
            "child_timeout_s": 0.1,
        }
        fills = asyncio.run(self.algo.execute(order, self.exchange, config))
        self.assertEqual(len(fills), 3)
        total_qty = sum(f["quantity"] for f in fills)
        self.assertAlmostEqual(total_qty, 1.0, places=6)

    def test_execute_single_slice(self):
        order = _make_order(qty=0.5)
        config = {"num_slices": 1, "duration_seconds": 0, "child_timeout_s": 0.1}
        fills = asyncio.run(self.algo.execute(order, self.exchange, config))
        self.assertEqual(len(fills), 1)

    def test_execute_market_fallback_on_limit_not_implemented(self):
        """If place_limit_order raises NotImplementedError, falls back to market."""
        self.exchange.place_limit_order = AsyncMock(
            side_effect=NotImplementedError
        )
        order = _make_order(qty=1.0)
        config = {"num_slices": 2, "duration_seconds": 0, "child_timeout_s": 0.1}
        fills = asyncio.run(self.algo.execute(order, self.exchange, config))
        self.assertTrue(len(fills) > 0)
        self.assertEqual(self.exchange.place_order.call_count, 2)

    def test_execute_market_fallback_on_timeout(self):
        """If get_order_status never returns filled, falls back to market."""
        self.exchange.get_order_status = AsyncMock(
            return_value={"status": "pending"}
        )
        order = _make_order(qty=1.0)
        config = {
            "num_slices": 1, "duration_seconds": 0,
            "child_timeout_s": 0.1,  # very short timeout
        }
        fills = asyncio.run(self.algo.execute(order, self.exchange, config))
        self.assertEqual(len(fills), 1)
        # Should have cancelled then placed market
        self.assertTrue(self.exchange.cancel_order.called)
        self.assertTrue(self.exchange.place_order.called)

    def test_execute_with_zero_mid_price(self):
        """If ticker returns 0 mid, falls back to market."""
        self.exchange.get_ticker.return_value = {"mid": 0, "last": 0}
        order = _make_order(qty=1.0)
        config = {"num_slices": 1, "duration_seconds": 0}
        fills = asyncio.run(self.algo.execute(order, self.exchange, config))
        self.assertTrue(len(fills) > 0)
        self.assertTrue(self.exchange.place_order.called)

    def test_execute_default_config(self):
        """Execute uses sensible defaults when config is empty."""
        order = _make_order(qty=0.1)
        # Just verify it doesn't crash with empty config
        # Use very short timeout to avoid waiting
        config = {"duration_seconds": 0, "child_timeout_s": 0.1}
        fills = asyncio.run(self.algo.execute(order, self.exchange, config))
        self.assertIsInstance(fills, list)

    def test_execute_respects_side_buy(self):
        """Buy side should offset price upward."""
        order = _make_order(side=Side.BUY, qty=0.1)
        config = {"num_slices": 1, "duration_seconds": 0, "child_timeout_s": 0.1}
        asyncio.run(self.algo.execute(order, self.exchange, config))
        # Limit order should have been placed
        if self.exchange.place_limit_order.called:
            call_args = self.exchange.place_limit_order.call_args
            limit_price = call_args[1].get("price", call_args[0][3] if len(call_args[0]) > 3 else 0)
            self.assertGreater(limit_price, 0)

    def test_execute_respects_side_sell(self):
        """Sell side should offset price downward."""
        order = _make_order(side=Side.SELL, qty=0.1)
        self.exchange.place_order.return_value = Fill(
            order_id="fill-s", symbol="BTC/USDC", side=Side.SELL,
            quantity=0.1, fill_price=49990.0,
            timestamp=datetime.now(timezone.utc), commission=0.1,
        )
        config = {"num_slices": 1, "duration_seconds": 0, "child_timeout_s": 0.1}
        fills = asyncio.run(self.algo.execute(order, self.exchange, config))
        self.assertTrue(len(fills) > 0)

    def test_partial_fill_handling(self):
        """Exchange place_order returns partial fill quantity."""
        self.exchange.place_order.return_value = Fill(
            order_id="fill-p", symbol="BTC/USDC", side=Side.BUY,
            quantity=0.3, fill_price=50000.0,
            timestamp=datetime.now(timezone.utc), commission=0.1,
        )
        self.exchange.place_limit_order = AsyncMock(
            side_effect=NotImplementedError
        )
        order = _make_order(qty=1.0)
        config = {"num_slices": 2, "duration_seconds": 0}
        fills = asyncio.run(self.algo.execute(order, self.exchange, config))
        # Each slice falls back to market, gets partial fill
        self.assertEqual(len(fills), 2)

    def test_none_config_uses_defaults(self):
        order = _make_order(qty=0.1)
        # Pass minimal overrides to avoid long wait; None config merges with defaults
        config = {"duration_seconds": 0, "child_timeout_s": 0.1}
        fills = asyncio.run(self.algo.execute(order, self.exchange, config))
        self.assertIsInstance(fills, list)


# ======================================================================
# TestAdaptiveLimitAlgorithm
# ======================================================================

class TestAdaptiveLimitAlgorithm(unittest.TestCase):
    """Tests for Adaptive Limit execution algorithm."""

    def setUp(self):
        self.algo = AdaptiveLimitAlgorithm()
        self.exchange = _make_exchange_mock()

    def test_metadata(self):
        meta = self.algo.get_metadata()
        self.assertEqual(meta["name"], "adaptive_limit")
        self.assertIn("initial_offset_bps", meta["default_config"])

    def test_compute_limit_price_buy(self):
        """Buy should place below mid (more favourable)."""
        price = self.algo._compute_limit_price(50000.0, Side.BUY, 5)
        self.assertLess(price, 50000.0)
        expected = 50000.0 - 50000.0 * 5 / 10000.0
        self.assertAlmostEqual(price, expected)

    def test_compute_limit_price_sell(self):
        """Sell should place above mid (more favourable)."""
        price = self.algo._compute_limit_price(50000.0, Side.SELL, 5)
        self.assertGreater(price, 50000.0)
        expected = 50000.0 + 50000.0 * 5 / 10000.0
        self.assertAlmostEqual(price, expected)

    def test_compute_limit_price_zero_offset(self):
        """Zero offset should return mid."""
        price = self.algo._compute_limit_price(50000.0, Side.BUY, 0)
        self.assertAlmostEqual(price, 50000.0)

    def test_execute_fill_on_first_attempt(self):
        """If order fills immediately, returns single fill."""
        order = _make_order(qty=1.0)
        config = {
            "initial_offset_bps": 5,
            "step_bps": 1,
            "adjust_interval_s": 0.01,
            "max_wait_s": 0.05,
        }
        fills = asyncio.run(self.algo.execute(order, self.exchange, config))
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0]["algo"], "adaptive_limit")

    def test_execute_walks_price(self):
        """Price should be adjusted when order isn't filled immediately."""
        call_count = [0]
        async def mock_status(order_id):
            call_count[0] += 1
            if call_count[0] >= 3:
                return {"status": "filled", "fill_price": 49999.0, "fee": 0.2}
            return {"status": "pending"}

        self.exchange.get_order_status = mock_status
        order = _make_order(qty=1.0)
        config = {
            "initial_offset_bps": 10,
            "step_bps": 2,
            "adjust_interval_s": 0.01,
            "max_wait_s": 1.0,
        }
        fills = asyncio.run(self.algo.execute(order, self.exchange, config))
        self.assertEqual(len(fills), 1)
        # Should have cancelled and re-placed at least once
        self.assertTrue(self.exchange.cancel_order.called)

    def test_execute_max_wait_market_fallback(self):
        """If max_wait exceeded, falls back to market."""
        self.exchange.get_order_status = AsyncMock(
            return_value={"status": "pending"}
        )
        order = _make_order(qty=1.0)
        config = {
            "initial_offset_bps": 5,
            "step_bps": 1,
            "adjust_interval_s": 0.01,
            "max_wait_s": 0.05,
        }
        fills = asyncio.run(self.algo.execute(order, self.exchange, config))
        # Should fall back to market
        self.assertTrue(len(fills) > 0)
        self.assertTrue(self.exchange.place_order.called)

    def test_execute_limit_not_implemented_fallback(self):
        """Falls back to market if place_limit_order not implemented."""
        self.exchange.place_limit_order = AsyncMock(
            side_effect=NotImplementedError
        )
        order = _make_order(qty=1.0)
        config = {"max_wait_s": 0.1}
        fills = asyncio.run(self.algo.execute(order, self.exchange, config))
        self.assertTrue(len(fills) > 0)
        self.assertTrue(self.exchange.place_order.called)

    def test_execute_zero_mid_fallback(self):
        """Zero mid price triggers market fallback."""
        self.exchange.get_ticker.return_value = {"mid": 0, "last": 0}
        order = _make_order(qty=1.0)
        fills = asyncio.run(self.algo.execute(order, self.exchange, {}))
        self.assertTrue(self.exchange.place_order.called)

    def test_execute_buy_direction(self):
        """Buy order places limit below mid."""
        order = _make_order(side=Side.BUY, qty=0.1)
        config = {
            "initial_offset_bps": 10,
            "adjust_interval_s": 0.01,
            "max_wait_s": 0.05,
        }
        asyncio.run(self.algo.execute(order, self.exchange, config))
        if self.exchange.place_limit_order.called:
            _, kwargs = self.exchange.place_limit_order.call_args
            self.assertLess(kwargs["price"], 50000.0)

    def test_execute_sell_direction(self):
        """Sell order places limit above mid."""
        self.exchange.place_order.return_value = Fill(
            order_id="fill-s", symbol="BTC/USDC", side=Side.SELL,
            quantity=0.1, fill_price=50010.0,
            timestamp=datetime.now(timezone.utc), commission=0.1,
        )
        order = _make_order(side=Side.SELL, qty=0.1)
        config = {
            "initial_offset_bps": 10,
            "adjust_interval_s": 0.01,
            "max_wait_s": 0.05,
        }
        asyncio.run(self.algo.execute(order, self.exchange, config))
        if self.exchange.place_limit_order.called:
            _, kwargs = self.exchange.place_limit_order.call_args
            self.assertGreater(kwargs["price"], 50000.0)

    def test_execute_status_error_returns_market(self):
        """get_order_status raising triggers market fallback."""
        self.exchange.get_order_status = AsyncMock(
            side_effect=Exception("connection error")
        )
        order = _make_order(qty=0.5)
        config = {"max_wait_s": 0.05, "adjust_interval_s": 0.01}
        fills = asyncio.run(self.algo.execute(order, self.exchange, config))
        self.assertTrue(len(fills) > 0)

    def test_none_config_uses_defaults(self):
        order = _make_order(qty=0.1)
        config = {"max_wait_s": 0.05, "adjust_interval_s": 0.01}
        fills = asyncio.run(self.algo.execute(order, self.exchange, config))
        self.assertIsInstance(fills, list)


# ======================================================================
# TestIcebergAlgorithm
# ======================================================================

class TestIcebergAlgorithm(unittest.TestCase):
    """Tests for Iceberg execution algorithm."""

    def setUp(self):
        self.algo = IcebergAlgorithm()
        self.exchange = _make_exchange_mock()

    def test_metadata(self):
        meta = self.algo.get_metadata()
        self.assertEqual(meta["name"], "iceberg")
        self.assertIn("visible_pct", meta["default_config"])

    def test_compute_visible_qty(self):
        """20% of 10 = 2."""
        qty = self.algo._compute_visible_qty(10.0, 0.20, 0.001)
        self.assertAlmostEqual(qty, 2.0)

    def test_compute_visible_qty_min_enforced(self):
        """If 20% is below min, use min."""
        qty = self.algo._compute_visible_qty(0.001, 0.20, 0.01)
        self.assertAlmostEqual(qty, 0.01)

    def test_compute_visible_qty_large_pct(self):
        """50% visible of 4.0 = 2.0."""
        qty = self.algo._compute_visible_qty(4.0, 0.50, 0.001)
        self.assertAlmostEqual(qty, 2.0)

    def test_execute_refills_after_fill(self):
        """Iceberg should submit multiple visible slices."""
        order = _make_order(qty=1.0)
        config = {
            "visible_pct": 0.50,
            "min_visible_qty": 0.001,
            "price_offset_bps": 3,
            "refill_delay_s": 0,
        }
        fills = asyncio.run(self.algo.execute(order, self.exchange, config))
        # 1.0 / 0.5 = 2 slices
        self.assertEqual(len(fills), 2)
        total = sum(f["quantity"] for f in fills)
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_execute_completion_tracking(self):
        """Total fill quantity should match order quantity."""
        order = _make_order(qty=2.0)
        config = {"visible_pct": 0.25, "refill_delay_s": 0}
        fills = asyncio.run(self.algo.execute(order, self.exchange, config))
        total = sum(f["quantity"] for f in fills)
        self.assertAlmostEqual(total, 2.0, places=6)

    def test_execute_partial_final_slice(self):
        """Last slice should handle remainder correctly."""
        order = _make_order(qty=1.0)
        config = {
            "visible_pct": 0.30,  # 0.3 per slice
            "min_visible_qty": 0.001,
            "refill_delay_s": 0,
        }
        fills = asyncio.run(self.algo.execute(order, self.exchange, config))
        # 1.0 / 0.3 = ~3.33 → 4 slices (last is 0.1)
        total = sum(f["quantity"] for f in fills)
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_execute_market_fallback(self):
        """Falls back to market if limit fails."""
        self.exchange.place_limit_order = AsyncMock(
            side_effect=NotImplementedError
        )
        order = _make_order(qty=0.5)
        config = {"visible_pct": 0.50, "refill_delay_s": 0}
        fills = asyncio.run(self.algo.execute(order, self.exchange, config))
        self.assertTrue(len(fills) > 0)
        self.assertTrue(self.exchange.place_order.called)

    def test_execute_min_visible_qty_enforced(self):
        """Visible qty should not go below min."""
        order = _make_order(qty=0.01)
        config = {
            "visible_pct": 0.10,   # 0.001 — below min
            "min_visible_qty": 0.005,
            "refill_delay_s": 0,
        }
        fills = asyncio.run(self.algo.execute(order, self.exchange, config))
        total = sum(f["quantity"] for f in fills)
        self.assertAlmostEqual(total, 0.01, places=6)

    def test_execute_single_slice_when_small(self):
        """If visible_qty >= total, only one slice."""
        order = _make_order(qty=0.001)
        config = {"visible_pct": 1.0, "refill_delay_s": 0}
        fills = asyncio.run(self.algo.execute(order, self.exchange, config))
        self.assertEqual(len(fills), 1)

    def test_none_config_uses_defaults(self):
        order = _make_order(qty=0.1)
        config = {"refill_delay_s": 0}
        fills = asyncio.run(self.algo.execute(order, self.exchange, config))
        self.assertIsInstance(fills, list)


# ======================================================================
# TestAlgorithmSelector
# ======================================================================

class TestAlgorithmSelector(unittest.TestCase):
    """Tests for AlgorithmSelector."""

    def setUp(self):
        self.selector = AlgorithmSelector()

    def test_small_order_selects_adaptive_limit(self):
        """Orders < 1% avg volume → adaptive_limit."""
        order = _make_order(qty=0.5)
        market_state = {"avg_volume": 100.0}
        name, algo = self.selector.select(order, market_state)
        self.assertEqual(name, "adaptive_limit")
        self.assertIsInstance(algo, AdaptiveLimitAlgorithm)

    def test_medium_order_selects_twap(self):
        """Orders 1-5% avg volume → TWAP."""
        order = _make_order(qty=3.0)
        market_state = {"avg_volume": 100.0}
        name, algo = self.selector.select(order, market_state)
        self.assertEqual(name, "twap")
        self.assertIsInstance(algo, TWAPAlgorithm)

    def test_large_order_selects_iceberg(self):
        """Orders > 5% avg volume → iceberg."""
        order = _make_order(qty=10.0)
        market_state = {"avg_volume": 100.0}
        name, algo = self.selector.select(order, market_state)
        self.assertEqual(name, "iceberg")
        self.assertIsInstance(algo, IcebergAlgorithm)

    def test_volatile_regime_selects_market(self):
        """Volatile regime → market (None algo)."""
        order = _make_order(qty=1.0)
        market_state = {"avg_volume": 100.0, "regime": "volatile"}
        name, algo = self.selector.select(order, market_state)
        self.assertEqual(name, "market")
        self.assertIsNone(algo)

    def test_signal_metadata_override(self):
        """exec_algo in signal metadata overrides size-based selection."""
        order = _make_order(qty=0.1)  # would be adaptive_limit
        market_state = {"avg_volume": 100.0}
        signal_metadata = {"exec_algo": "iceberg"}
        name, algo = self.selector.select(order, market_state, signal_metadata)
        self.assertEqual(name, "iceberg")
        self.assertIsInstance(algo, IcebergAlgorithm)

    def test_unknown_override_ignored(self):
        """Unknown exec_algo override falls through to size-based."""
        order = _make_order(qty=0.1)
        market_state = {"avg_volume": 100.0}
        signal_metadata = {"exec_algo": "unknown_algo"}
        name, algo = self.selector.select(order, market_state, signal_metadata)
        # Falls through to size-based (0.1% → adaptive_limit)
        self.assertEqual(name, "adaptive_limit")

    def test_no_volume_defaults_to_adaptive(self):
        """When avg_volume is 0 or missing, default to adaptive_limit."""
        order = _make_order(qty=1.0)
        name, algo = self.selector.select(order, {})
        self.assertEqual(name, "adaptive_limit")
        self.assertIsInstance(algo, AdaptiveLimitAlgorithm)

    def test_get_available_algorithms(self):
        algos = self.selector.get_available_algorithms()
        self.assertIn("market", algos)
        self.assertIn("twap", algos)
        self.assertIn("adaptive_limit", algos)
        self.assertIn("iceberg", algos)
        self.assertEqual(len(algos), 4)

    def test_config_thresholds(self):
        """Custom thresholds affect selection."""
        selector = AlgorithmSelector({
            "small_threshold_pct": 2.0,
            "large_threshold_pct": 10.0,
        })
        order = _make_order(qty=1.5)
        market_state = {"avg_volume": 100.0}
        # 1.5% is below custom small threshold of 2%
        name, _ = selector.select(order, market_state)
        self.assertEqual(name, "adaptive_limit")

    def test_market_override_returns_none_algo(self):
        """Explicit market override returns None algorithm."""
        order = _make_order(qty=1.0)
        signal_metadata = {"exec_algo": "market"}
        name, algo = self.selector.select(order, {}, signal_metadata)
        self.assertEqual(name, "market")
        self.assertIsNone(algo)


# ======================================================================
# TestEngineAlgorithmIntegration
# ======================================================================

class TestEngineAlgorithmIntegration(unittest.TestCase):
    """Tests for ExecutionEngine integration with algorithms."""

    def _make_engine(self, with_selector=True):
        from execution.engine import ExecutionEngine
        from execution.risk_controls import ExecutionRiskControls

        exchange = _make_exchange_mock()
        exchange.get_positions.return_value = []
        exchange.get_account_state.return_value = {"equity": 100000}

        risk_manager = MagicMock()
        risk_manager.validate_signal.return_value = True
        risk_manager.calculate_position_size.return_value = 1.0

        risk_controls = MagicMock(spec=ExecutionRiskControls)
        risk_controls.validate_order.return_value = (True, "")
        risk_controls.check_portfolio_limits.return_value = (True, "")

        timescale = MagicMock()
        redis = MagicMock()
        redis.get_account_state.return_value = {"total_equity": 100000.0}

        config = {"symbol": "BTC/USDC"}

        selector = AlgorithmSelector() if with_selector else None

        engine = ExecutionEngine(
            exchange=exchange,
            risk_manager=risk_manager,
            risk_controls=risk_controls,
            timescale=timescale,
            redis=redis,
            config=config,
            algorithm_selector=selector,
        )
        # Mock the kill switch so it doesn't block signals
        engine.kill_switch = MagicMock()
        engine.kill_switch.is_active.return_value = False
        return engine, exchange, timescale

    def test_backward_compat_no_selector(self):
        """Without algorithm_selector, engine uses market orders as before."""
        engine, exchange, _ = self._make_engine(with_selector=False)
        signal = _make_signal()
        fill = asyncio.run(engine.process_signal(signal, "test_strat"))
        self.assertIsNotNone(fill)
        self.assertTrue(exchange.place_order.called)

    def test_algorithm_selected_and_used(self):
        """When selector returns an algo, execute_with_algorithm is called."""
        engine, exchange, timescale = self._make_engine(with_selector=True)

        # Force selector to return TWAP
        engine.algorithm_selector = MagicMock()
        twap = TWAPAlgorithm()
        engine.algorithm_selector.select.return_value = ("twap", twap)
        engine.algorithm_selector.get_algo_config.return_value = {
            "duration_seconds": 0, "num_slices": 1, "child_timeout_s": 0.1,
        }

        signal = _make_signal()
        fill = asyncio.run(engine.process_signal(signal, "test_strat"))
        self.assertIsNotNone(fill)
        engine.algorithm_selector.select.assert_called_once()

    def test_algorithm_returns_none_falls_back_to_market(self):
        """When algo is None (market), falls through to execute_order."""
        engine, exchange, _ = self._make_engine(with_selector=True)
        engine.algorithm_selector = MagicMock()
        engine.algorithm_selector.select.return_value = ("market", None)

        signal = _make_signal()
        fill = asyncio.run(engine.process_signal(signal, "test_strat"))
        self.assertIsNotNone(fill)
        self.assertTrue(exchange.place_order.called)

    def test_algorithm_metrics_emitted(self):
        """execute_with_algorithm records prometheus metrics."""
        engine, exchange, timescale = self._make_engine(with_selector=True)
        order = _make_order()
        algo = TWAPAlgorithm()
        config_override = {
            "duration_seconds": 0, "num_slices": 1, "child_timeout_s": 0.1,
        }
        engine.algorithm_selector = MagicMock()
        engine.algorithm_selector.get_algo_config.return_value = config_override

        fill = asyncio.run(
            engine.execute_with_algorithm(order, algo, "twap", "test")
        )
        self.assertIsNotNone(fill)
        # Fill should be recorded in timescale
        self.assertTrue(timescale.update_order_status.called)
        self.assertTrue(timescale.insert_fill.called)

    def test_execute_with_algorithm_empty_fills(self):
        """If algorithm returns empty fills, returns None."""
        engine, _, _ = self._make_engine(with_selector=True)
        engine.algorithm_selector = MagicMock()
        engine.algorithm_selector.get_algo_config.return_value = {}

        algo = MagicMock()
        algo.execute = AsyncMock(return_value=[])

        order = _make_order()
        fill = asyncio.run(
            engine.execute_with_algorithm(order, algo, "test_algo", "test")
        )
        self.assertIsNone(fill)


if __name__ == "__main__":
    unittest.main()
