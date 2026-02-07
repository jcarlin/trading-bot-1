"""Tests for execution.risk_controls.ExecutionRiskControls."""

import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from core.models import Order, Position
from core.types import OrderType, Side


class TestExecutionRiskControls(unittest.TestCase):
    """Test all risk limit scenarios for ExecutionRiskControls."""

    def _make_controls(self, overrides=None):
        from execution.risk_controls import ExecutionRiskControls

        config = MagicMock()
        section = {
            "max_position_pct_per_asset": 0.05,
            "max_portfolio_exposure_pct": 1.50,
            "max_strategy_drawdown_pct": 0.05,
            "max_portfolio_drawdown_pct": 0.15,
            "max_single_trade_loss_pct": 0.01,
            "min_cash_reserve_pct": 0.20,
        }
        if overrides:
            section.update(overrides)
        config.get_section.return_value = section
        return ExecutionRiskControls(config)

    def _make_order(self, quantity=1.0, price=100.0, stop_loss=None):
        return Order(
            symbol="BTC/USDT",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=price,
            stop_loss=stop_loss,
            timestamp=datetime.now(timezone.utc),
            order_id="test-001",
        )

    def _make_position(self, symbol="ETH/USDT", entry_price=50.0, quantity=1.0):
        return Position(
            symbol=symbol,
            side=Side.BUY,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=datetime.now(timezone.utc),
        )

    # ------------------------------------------------------------------
    # validate_order tests
    # ------------------------------------------------------------------

    @patch("execution.risk_controls.risk_check_passed_total")
    @patch("execution.risk_controls.risk_check_failed_total")
    def test_order_within_limits(self, mock_fail, mock_pass):
        """An order well within the 5% position limit should pass."""
        rc = self._make_controls()
        order = self._make_order(quantity=0.04, price=100.0)  # value = 4.0
        equity = 10_000.0
        positions = []

        ok, reason = rc.validate_order(order, equity, positions)

        self.assertTrue(ok)
        self.assertEqual(reason, "")

    @patch("execution.risk_controls.risk_check_passed_total")
    @patch("execution.risk_controls.risk_check_failed_total")
    def test_order_exceeds_position_limit(self, mock_fail, mock_pass):
        """An order exceeding 5% of equity should be rejected."""
        rc = self._make_controls()
        # value = 10 * 100 = 1000, which is 10% of 10_000
        order = self._make_order(quantity=10.0, price=100.0)
        equity = 10_000.0

        ok, reason = rc.validate_order(order, equity, [])

        self.assertFalse(ok)
        self.assertIn("exceeds", reason)

    @patch("execution.risk_controls.risk_check_passed_total")
    @patch("execution.risk_controls.risk_check_failed_total")
    def test_single_trade_loss_limit(self, mock_fail, mock_pass):
        """Order with potential loss > 1% of equity should be rejected."""
        rc = self._make_controls()
        # price=100, stop_loss=80 -> loss_per_unit=20, qty=10 -> loss=200
        # 200 / 10_000 = 2% > 1% limit
        order = self._make_order(quantity=10.0, price=100.0, stop_loss=80.0)
        # But first the position size check would catch this (10*100=1000 > 5% of 10k=500)
        # Use smaller qty that passes position check but fails loss check
        # qty=4, value=400 (4% of 10k, OK). loss=20*4=80, 80/10k=0.8% OK
        # Need: loss > 1%. qty=5.1, value=510 (5.1%, fails position check)
        # Let's just adjust equity so position passes but loss fails
        # qty=1, price=100, value=100. equity=2000 -> 5% OK
        # stop_loss=80, loss=20*1=20, 20/2000=1% -> exactly at limit
        # Use stop_loss=70: loss=30, 30/2000=1.5% > 1%
        order2 = self._make_order(quantity=1.0, price=100.0, stop_loss=70.0)
        equity = 2000.0

        ok, reason = rc.validate_order(order2, equity, [])

        self.assertFalse(ok)
        self.assertIn("potential loss", reason)

    @patch("execution.risk_controls.risk_check_passed_total")
    @patch("execution.risk_controls.risk_check_failed_total")
    def test_cash_reserve_limit(self, mock_fail, mock_pass):
        """Order that would reduce cash below 20% reserve should be rejected."""
        rc = self._make_controls()
        # equity=10_000, existing positions value=7500
        # new order value=1000 -> total=8500 -> cash=1500 -> 15% < 20%
        existing = self._make_position(entry_price=7500.0, quantity=1.0)
        order = self._make_order(quantity=10.0, price=100.0)  # value=1000
        # position check: 1000/10000 = 10% > 5% — this would fail first
        # Use smaller order that passes position but fails cash
        order2 = self._make_order(quantity=4.0, price=100.0)  # value=400, 4% OK
        # existing=7500, new_total=7900, cash=2100, 21% > 20% — passes
        # Make existing bigger: entry_price=8200, qty=1 -> existing=8200
        existing2 = self._make_position(entry_price=8200.0, quantity=1.0)
        # new_total=8200+400=8600, cash=1400, 14% < 20%
        ok, reason = rc.validate_order(order2, 10_000.0, [existing2])

        self.assertFalse(ok)
        self.assertIn("cash reserve", reason)

    # ------------------------------------------------------------------
    # check_portfolio_limits tests
    # ------------------------------------------------------------------

    @patch("execution.risk_controls.risk_check_passed_total")
    @patch("execution.risk_controls.risk_check_failed_total")
    def test_portfolio_exposure_limit(self, mock_fail, mock_pass):
        """Total exposure > 150% should be rejected."""
        rc = self._make_controls()
        # equity=10_000, total positions = 16_000 -> 160% > 150%
        positions = [self._make_position(entry_price=16_000.0, quantity=1.0)]

        ok, reason = rc.check_portfolio_limits(10_000.0, 10_000.0, positions)

        self.assertFalse(ok)
        self.assertIn("exposure", reason)

    @patch("execution.risk_controls.risk_check_passed_total")
    @patch("execution.risk_controls.risk_check_failed_total")
    def test_portfolio_drawdown_halt(self, mock_fail, mock_pass):
        """Drawdown > 15% should trigger rejection and set halt flag."""
        rc = self._make_controls()
        peak = 10_000.0
        equity = 8_000.0  # drawdown = 20% > 15%

        ok, reason = rc.check_portfolio_limits(equity, peak, [])

        self.assertFalse(ok)
        self.assertIn("drawdown", reason)

    @patch("execution.risk_controls.risk_check_passed_total")
    @patch("execution.risk_controls.risk_check_failed_total")
    def test_should_halt(self, mock_fail, mock_pass):
        """should_halt() should return True after drawdown breach."""
        rc = self._make_controls()

        self.assertFalse(rc.should_halt())

        # Trigger drawdown breach
        rc.check_portfolio_limits(8_000.0, 10_000.0, [])

        self.assertTrue(rc.should_halt())

    @patch("execution.risk_controls.risk_check_passed_total")
    @patch("execution.risk_controls.risk_check_failed_total")
    def test_strategy_drawdown_config(self, mock_fail, mock_pass):
        """Strategy drawdown limit should be configurable."""
        rc = self._make_controls({"max_strategy_drawdown_pct": 0.10})
        self.assertEqual(rc.max_strategy_drawdown_pct, 0.10)

    @patch("execution.risk_controls.risk_check_passed_total")
    @patch("execution.risk_controls.risk_check_failed_total")
    def test_zero_equity_rejected(self, mock_fail, mock_pass):
        """Zero equity should reject order."""
        rc = self._make_controls()
        order = self._make_order()

        ok, reason = rc.validate_order(order, 0.0, [])
        self.assertFalse(ok)

        ok2, reason2 = rc.check_portfolio_limits(0.0, 10_000.0, [])
        self.assertFalse(ok2)


if __name__ == "__main__":
    unittest.main()
