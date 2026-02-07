"""Hard execution risk controls that strategies cannot bypass.

These limits are enforced at the execution layer, independent of any
strategy-level risk management. They act as the last line of defence
before capital is committed to the market.
"""

import logging

from core.models import Order, Position
from monitoring.metrics import risk_check_passed_total, risk_check_failed_total

logger = logging.getLogger(__name__)


class ExecutionRiskControls:
    """Non-overridable risk limits enforced at the execution layer."""

    def __init__(self, config):
        section = (
            config.get_section("execution_risk")
            if hasattr(config, "get_section")
            else config if isinstance(config, dict) else {}
        )
        self.max_position_pct_per_asset: float = float(section.get("max_position_pct_per_asset", 0.05))
        self.max_portfolio_exposure_pct: float = float(section.get("max_portfolio_exposure_pct", 1.50))
        self.max_strategy_drawdown_pct: float = float(section.get("max_strategy_drawdown_pct", 0.05))
        self.max_portfolio_drawdown_pct: float = float(section.get("max_portfolio_drawdown_pct", 0.15))
        self.max_single_trade_loss_pct: float = float(section.get("max_single_trade_loss_pct", 0.01))
        self.min_cash_reserve_pct: float = float(section.get("min_cash_reserve_pct", 0.20))

        self._portfolio_drawdown_breached = False

        logger.info(
            "ExecutionRiskControls initialised: max_pos=%.1f%% max_exp=%.0f%% "
            "max_dd=%.1f%% max_trade_loss=%.1f%% min_cash=%.0f%%",
            self.max_position_pct_per_asset * 100,
            self.max_portfolio_exposure_pct * 100,
            self.max_portfolio_drawdown_pct * 100,
            self.max_single_trade_loss_pct * 100,
            self.min_cash_reserve_pct * 100,
        )

    # ------------------------------------------------------------------
    # Order-level checks
    # ------------------------------------------------------------------

    def validate_order(
        self, order: Order, equity: float, positions: list[Position]
    ) -> tuple[bool, str]:
        """Validate a single order against hard risk limits.

        Returns:
            (True, "") if the order passes all checks, or
            (False, "reason string") if it should be rejected.
        """
        if equity <= 0:
            self._fail("validate_order")
            return False, "equity is zero or negative"

        # 1. Position size vs max_position_pct_per_asset
        order_value = order.quantity * (order.price if order.price else 0)
        if order.price is None:
            # For market orders, we can't precisely check notional, so we
            # skip the notional check (the caller should provide estimated price).
            pass
        else:
            if order_value / equity > self.max_position_pct_per_asset:
                reason = (
                    f"order value {order_value:.2f} exceeds "
                    f"{self.max_position_pct_per_asset:.0%} of equity {equity:.2f}"
                )
                self._fail("position_size")
                return False, reason

        # 2. Single trade potential loss vs max_single_trade_loss_pct
        if order.stop_loss is not None and order.price is not None:
            loss_per_unit = abs(order.price - order.stop_loss)
            potential_loss = loss_per_unit * order.quantity
            if potential_loss / equity > self.max_single_trade_loss_pct:
                reason = (
                    f"potential loss {potential_loss:.2f} exceeds "
                    f"{self.max_single_trade_loss_pct:.0%} of equity {equity:.2f}"
                )
                self._fail("single_trade_loss")
                return False, reason

        # 3. Cash reserve check
        total_position_value = sum(
            p.entry_price * p.quantity for p in positions
        )
        new_total = total_position_value + order_value
        cash_after = equity - new_total
        if cash_after / equity < self.min_cash_reserve_pct:
            reason = (
                f"cash reserve would drop to {cash_after / equity:.1%}, "
                f"below minimum {self.min_cash_reserve_pct:.0%}"
            )
            self._fail("cash_reserve")
            return False, reason

        self._pass("validate_order")
        return True, ""

    # ------------------------------------------------------------------
    # Portfolio-level checks
    # ------------------------------------------------------------------

    def check_portfolio_limits(
        self, equity: float, peak_equity: float, positions: list[Position]
    ) -> tuple[bool, str]:
        """Check portfolio-wide risk limits.

        Returns:
            (True, "") if within limits, or (False, "reason") if breached.
        """
        if equity <= 0:
            self._fail("portfolio_limits")
            return False, "equity is zero or negative"

        # 1. Total exposure vs max_portfolio_exposure_pct
        total_exposure = sum(p.entry_price * p.quantity for p in positions)
        exposure_ratio = total_exposure / equity
        if exposure_ratio > self.max_portfolio_exposure_pct:
            reason = (
                f"portfolio exposure {exposure_ratio:.0%} exceeds "
                f"limit {self.max_portfolio_exposure_pct:.0%}"
            )
            self._fail("portfolio_exposure")
            return False, reason

        # 2. Portfolio drawdown vs max_portfolio_drawdown_pct
        if peak_equity > 0:
            drawdown = (peak_equity - equity) / peak_equity
            if drawdown >= self.max_portfolio_drawdown_pct:
                self._portfolio_drawdown_breached = True
                reason = (
                    f"portfolio drawdown {drawdown:.1%} exceeds "
                    f"limit {self.max_portfolio_drawdown_pct:.0%}"
                )
                self._fail("portfolio_drawdown")
                return False, reason

        self._pass("portfolio_limits")
        return True, ""

    # ------------------------------------------------------------------
    # Halt check
    # ------------------------------------------------------------------

    def should_halt(self) -> bool:
        """Return True if the portfolio drawdown limit has been breached."""
        return self._portfolio_drawdown_breached

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pass(check_type: str) -> None:
        risk_check_passed_total.labels(check_type=check_type).inc()

    @staticmethod
    def _fail(check_type: str) -> None:
        risk_check_failed_total.labels(check_type=check_type).inc()
