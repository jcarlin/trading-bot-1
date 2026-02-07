"""Tests for testing.risk_stress_tester and testing.circuit_breaker_validator."""

import asyncio
import sys
import os
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.models import Order, Position
from core.types import OrderType, Side
from testing.risk_stress_tester import RiskStressTester, ScenarioResult
from testing.circuit_breaker_validator import CircuitBreakerValidator


# Prometheus metric patches required by ExecutionRiskControls
_METRIC_PATCHES = [
    "execution.risk_controls.risk_check_passed_total",
    "execution.risk_controls.risk_check_failed_total",
    "execution.kill_switch.circuit_breaker_status",
    "execution.kill_switch.kill_switch_activations_total",
]


def _apply_metric_patches():
    """Start all metric patches and return a list of patchers."""
    patchers = []
    for target in _METRIC_PATCHES:
        p = patch(target, MagicMock())
        p.start()
        patchers.append(p)
    return patchers


def _make_risk_controls(overrides=None):
    """Create a real ExecutionRiskControls with default config."""
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


class TestRiskStressTester(unittest.TestCase):
    """Tests the stress tester against actual risk control code."""

    def setUp(self):
        self._patchers = _apply_metric_patches()
        self.risk_controls = _make_risk_controls()
        self.engine = MagicMock()
        self.kill_switch = MagicMock()
        self.strategy_manager = MagicMock()

        self.tester = RiskStressTester(
            execution_engine=self.engine,
            risk_controls=self.risk_controls,
            kill_switch=self.kill_switch,
            strategy_manager=self.strategy_manager,
            config={},
        )

    def tearDown(self):
        for p in self._patchers:
            p.stop()

    # ------------------------------------------------------------------
    # Flash crash scenario
    # ------------------------------------------------------------------

    def test_flash_crash_triggers_dd_breach(self):
        """50% drawdown should trigger portfolio DD breach and halt signal."""
        result = asyncio.run(self.tester.scenario_flash_crash())

        self.assertIsInstance(result, ScenarioResult)
        self.assertTrue(result.passed)
        self.assertEqual(result.scenario_name, "flash_crash")
        self.assertIn("portfolio_drawdown_limit", result.triggered_controls)
        self.assertIn("halt_signal", result.triggered_controls)

    # ------------------------------------------------------------------
    # Portfolio drawdown breach
    # ------------------------------------------------------------------

    def test_portfolio_dd_breach_detected(self):
        """16% drawdown (above 15% limit) should fail limits check."""
        result = asyncio.run(self.tester.scenario_portfolio_dd_breach())

        self.assertTrue(result.passed)
        self.assertEqual(result.scenario_name, "portfolio_dd_breach")
        self.assertIn("portfolio_drawdown_limit", result.triggered_controls)

    def test_portfolio_dd_within_limits(self):
        """10% drawdown (below 15% limit) should pass."""
        equity = 9000.0
        peak_equity = 10000.0

        passed, reason = self.risk_controls.check_portfolio_limits(
            equity, peak_equity, [])

        self.assertTrue(passed)
        self.assertEqual(reason, "")

    # ------------------------------------------------------------------
    # Oversized order
    # ------------------------------------------------------------------

    def test_oversized_order_rejected(self):
        """$50k order on $100k equity (50%) exceeds 5% position limit."""
        result = asyncio.run(self.tester.scenario_oversized_order())

        self.assertTrue(result.passed)
        self.assertEqual(result.scenario_name, "oversized_order")
        self.assertIn("position_size_limit", result.triggered_controls)

    def test_small_order_passes(self):
        """Small order within limits should be accepted."""
        order = Order(
            symbol="BTC/USDT",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=0.04,
            price=100.0,  # $4 notional on $10k equity = 0.04%
        )
        equity = 10000.0

        passed, reason = self.risk_controls.validate_order(order, equity, [])

        self.assertTrue(passed)
        self.assertEqual(reason, "")

    # ------------------------------------------------------------------
    # Cash reserve violation
    # ------------------------------------------------------------------

    def test_cash_reserve_violation(self):
        """Large order vs small equity should be rejected."""
        result = asyncio.run(self.tester.scenario_cash_reserve_violation())

        self.assertTrue(result.passed)
        self.assertEqual(result.scenario_name, "cash_reserve_violation")

    # ------------------------------------------------------------------
    # Exchange outage
    # ------------------------------------------------------------------

    def test_exchange_outage_graceful(self):
        """Outage scenario should report passed (graceful handling)."""
        result = asyncio.run(self.tester.scenario_exchange_outage())

        self.assertTrue(result.passed)
        self.assertEqual(result.scenario_name, "exchange_outage")
        self.assertEqual(result.triggered_controls, [])

    # ------------------------------------------------------------------
    # run_all_scenarios
    # ------------------------------------------------------------------

    def test_run_all_scenarios(self):
        """run_all_scenarios should return a list of ScenarioResults."""
        results = asyncio.run(self.tester.run_all_scenarios())

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 5)
        for r in results:
            self.assertIsInstance(r, ScenarioResult)
            self.assertIsInstance(r.scenario_name, str)
            self.assertIsInstance(r.passed, bool)

    # ------------------------------------------------------------------
    # ScenarioResult dataclass
    # ------------------------------------------------------------------

    def test_scenario_result_dataclass(self):
        """Verify ScenarioResult fields and defaults."""
        result = ScenarioResult(
            scenario_name="test",
            passed=True,
            expected_behavior="expected",
            actual_behavior="actual",
        )

        self.assertEqual(result.scenario_name, "test")
        self.assertTrue(result.passed)
        self.assertEqual(result.expected_behavior, "expected")
        self.assertEqual(result.actual_behavior, "actual")
        self.assertEqual(result.triggered_controls, [])
        self.assertEqual(result.details, {})
        self.assertEqual(result.duration_seconds, 0.0)

    # ------------------------------------------------------------------
    # Stress report generation
    # ------------------------------------------------------------------

    def test_stress_report_generated(self):
        """generate_stress_report should return a formatted string."""
        results = [
            ScenarioResult("s1", True, "pass expected", "passed"),
            ScenarioResult("s2", False, "fail expected", "failed"),
        ]
        report = self.tester.generate_stress_report(results)

        self.assertIsInstance(report, str)
        self.assertIn("RISK STRESS TEST REPORT", report)
        self.assertIn("Total Scenarios: 2", report)
        self.assertIn("Passed: 1/2", report)
        self.assertIn("Failed: 1/2", report)

    def test_stress_report_shows_pass_fail(self):
        """Report should contain [PASS] and [FAIL] markers."""
        results = [
            ScenarioResult(
                "passing_scenario", True, "exp", "act",
                triggered_controls=["ctrl1"],
            ),
            ScenarioResult("failing_scenario", False, "exp", "act"),
        ]
        report = self.tester.generate_stress_report(results)

        self.assertIn("[PASS] passing_scenario", report)
        self.assertIn("[FAIL] failing_scenario", report)
        self.assertIn("Triggered: ctrl1", report)


class TestCircuitBreakerValidator(unittest.TestCase):
    """Tests for CircuitBreakerValidator with mocked dependencies."""

    def setUp(self):
        self._patchers = _apply_metric_patches()

        # KillSwitch has async activate/deactivate but sync is_active
        self.kill_switch = MagicMock()
        self.kill_switch.activate = AsyncMock()
        self.kill_switch.deactivate = AsyncMock()
        self.risk_controls = MagicMock()
        self.engine = MagicMock()
        self.redis = MagicMock()
        self.timescale = MagicMock()

        self.validator = CircuitBreakerValidator(
            kill_switch=self.kill_switch,
            risk_controls=self.risk_controls,
            execution_engine=self.engine,
            redis_store=self.redis,
            timescale=self.timescale,
        )

    def tearDown(self):
        for p in self._patchers:
            p.stop()

    # ------------------------------------------------------------------
    # Activation
    # ------------------------------------------------------------------

    def test_validate_activation_sets_state(self):
        """Activation should set Redis and kill switch to active."""
        self.redis.get_circuit_breaker.return_value = {"active": True}
        self.kill_switch.is_active.return_value = True

        result = asyncio.run(self.validator.validate_activation())

        self.assertTrue(result["passed"])
        self.assertTrue(result["redis_active"])
        self.assertTrue(result["kill_switch_active"])
        self.kill_switch.activate.assert_called_once_with("stress_test_validation")

    def test_validate_activation_fails_when_redis_not_set(self):
        """Activation should fail if Redis doesn't report active."""
        self.redis.get_circuit_breaker.return_value = {"active": False}
        self.kill_switch.is_active.return_value = True

        result = asyncio.run(self.validator.validate_activation())

        self.assertFalse(result["passed"])

    # ------------------------------------------------------------------
    # Deactivation
    # ------------------------------------------------------------------

    def test_validate_deactivation_clears_state(self):
        """Deactivation should clear Redis and kill switch state."""
        self.redis.get_circuit_breaker.return_value = {"active": False}
        self.kill_switch.is_active.return_value = False

        result = asyncio.run(self.validator.validate_deactivation())

        self.assertTrue(result["passed"])
        self.assertTrue(result["redis_inactive"])
        self.assertTrue(result["kill_switch_inactive"])
        self.kill_switch.deactivate.assert_called_once()

    def test_validate_deactivation_fails_when_still_active(self):
        """Deactivation should fail if kill switch still reports active."""
        self.redis.get_circuit_breaker.return_value = {"active": False}
        self.kill_switch.is_active.return_value = True

        result = asyncio.run(self.validator.validate_deactivation())

        self.assertFalse(result["passed"])

    # ------------------------------------------------------------------
    # Order rejection when active
    # ------------------------------------------------------------------

    def test_validate_order_rejection(self):
        """When kill switch is active, order rejection check should pass."""
        self.kill_switch.is_active.return_value = True

        result = asyncio.run(self.validator.validate_order_rejection_when_active())

        self.assertTrue(result["passed"])
        self.assertTrue(result["kill_switch_active"])

    def test_validate_order_rejection_fails_when_inactive(self):
        """When kill switch is not active, order rejection check should fail."""
        self.kill_switch.is_active.return_value = False

        result = asyncio.run(self.validator.validate_order_rejection_when_active())

        self.assertFalse(result["passed"])

    # ------------------------------------------------------------------
    # validate_all
    # ------------------------------------------------------------------

    def test_validate_all_runs_checks(self):
        """validate_all should return a list of results for all checks."""
        self.redis.get_circuit_breaker.return_value = {"active": True}
        self.kill_switch.is_active.return_value = True

        results = asyncio.run(self.validator.validate_all())

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIn("check", r)
            self.assertIn("passed", r)

    def test_validate_handles_errors(self):
        """Exception in one check shouldn't stop others."""
        # Make activation raise an error
        self.kill_switch.activate.side_effect = RuntimeError("connection failed")
        # But deactivation and order rejection still work
        self.kill_switch.is_active.return_value = False
        self.redis.get_circuit_breaker.return_value = {"active": False}

        results = asyncio.run(self.validator.validate_all())

        self.assertEqual(len(results), 3)
        # First check (activation) should have failed with error
        self.assertFalse(results[0]["passed"])
        self.assertIn("error", results[0])
        self.assertIn("connection failed", results[0]["error"])


if __name__ == "__main__":
    unittest.main()
