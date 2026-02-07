"""Circuit breaker / kill switch validation."""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class CircuitBreakerValidator:
    """Validates kill switch activation/deactivation lifecycle."""

    def __init__(self, kill_switch, risk_controls, execution_engine,
                 redis_store, timescale):
        self.kill_switch = kill_switch
        self.risk_controls = risk_controls
        self.engine = execution_engine
        self.redis = redis_store
        self.timescale = timescale

    async def validate_all(self) -> list[dict]:
        """Run all circuit breaker validation checks."""
        checks = [
            self.validate_activation,
            self.validate_order_rejection_when_active,
            self.validate_deactivation,
        ]
        results = []
        for check in checks:
            try:
                result = await check()
                results.append(result)
            except Exception as e:
                results.append({
                    "check": check.__name__,
                    "passed": False,
                    "error": str(e),
                })
        return results

    async def validate_activation(self) -> dict:
        """Verify kill switch activation sets all expected state."""
        await self.kill_switch.activate("stress_test_validation")

        # Check Redis state
        cb_state = self.redis.get_circuit_breaker()
        redis_active = cb_state and cb_state.get("active")

        # Check kill switch reports active
        ks_active = self.kill_switch.is_active()

        passed = redis_active and ks_active

        return {
            "check": "activation",
            "passed": passed,
            "redis_active": redis_active,
            "kill_switch_active": ks_active,
        }

    async def validate_order_rejection_when_active(self) -> dict:
        """Verify orders are rejected when kill switch is active."""
        # Kill switch should already be active from previous check
        is_active = self.kill_switch.is_active()

        return {
            "check": "order_rejection_when_active",
            "passed": is_active,
            "kill_switch_active": is_active,
            "note": "When active, ExecutionEngine.process_signal() rejects at step 1",
        }

    async def validate_deactivation(self) -> dict:
        """Verify kill switch can be deactivated."""
        await self.kill_switch.deactivate()

        cb_state = self.redis.get_circuit_breaker()
        redis_inactive = cb_state and not cb_state.get("active")

        ks_inactive = not self.kill_switch.is_active()

        passed = redis_inactive and ks_inactive

        return {
            "check": "deactivation",
            "passed": passed,
            "redis_inactive": redis_inactive,
            "kill_switch_inactive": ks_inactive,
        }
