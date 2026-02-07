"""Risk manager for position sizing, drawdown checks, and signal validation."""

from core.models import Signal
from core.types import PositionSizing, SignalType


class RiskManager:
    """Manages risk controls for the trading system.

    Responsibilities:
        - Calculate position size based on the configured sizing method.
        - Enforce maximum drawdown limits.
        - Apply commission costs.
        - Validate signals against current position state.
    """

    def __init__(self, config: dict):
        """Initialize from the ``risk`` section of the config.

        Args:
            config: Dict with keys position_sizing, max_position_pct,
                    max_drawdown_pct, commission_pct, and optionally
                    fixed_amount and risk_per_trade.
        """
        self.position_sizing = PositionSizing(config.get("position_sizing", "percent_equity"))
        self.max_position_pct: float = float(config.get("max_position_pct", 0.95))
        self.max_drawdown_pct: float = float(config.get("max_drawdown_pct", 0.20))
        self.commission_pct: float = float(config.get("commission_pct", 0.001))

        # Optional params used by specific sizing methods
        self.fixed_amount: float = float(config.get("fixed_amount", 1000.0))
        self.risk_per_trade: float = float(config.get("risk_per_trade", 0.01))

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def calculate_position_size(
        self, signal: Signal, equity: float, current_price: float
    ) -> float:
        """Return the quantity (in base units) to trade.

        Args:
            signal: The trading signal (used for stop_loss in PERCENT_RISK mode).
            equity: Current account equity in quote currency.
            current_price: Latest price of the asset.

        Returns:
            Quantity of the asset to buy/sell.
        """
        if current_price <= 0:
            return 0.0

        if self.position_sizing == PositionSizing.FIXED_AMOUNT:
            return self.fixed_amount / current_price

        if self.position_sizing == PositionSizing.PERCENT_EQUITY:
            dollar_amount = self.max_position_pct * equity
            return dollar_amount / current_price

        if self.position_sizing == PositionSizing.PERCENT_RISK:
            # Risk-based sizing: size the position so that hitting the stop
            # loss results in losing at most (equity * risk_per_trade).
            if signal.stop_loss is None or signal.stop_loss >= current_price:
                # Cannot compute risk without a valid stop loss below entry;
                # fall back to percent-equity sizing.
                dollar_amount = self.max_position_pct * equity
                return dollar_amount / current_price
            risk_per_unit = current_price - signal.stop_loss
            max_loss = equity * self.risk_per_trade
            return max_loss / risk_per_unit

        return 0.0

    # ------------------------------------------------------------------
    # Drawdown check
    # ------------------------------------------------------------------

    def check_drawdown(self, equity: float, peak_equity: float) -> bool:
        """Check whether trading should continue.

        Args:
            equity: Current account equity.
            peak_equity: Highest equity value observed so far.

        Returns:
            True if trading may continue, False if max drawdown is breached.
        """
        if peak_equity <= 0:
            return False
        drawdown = (peak_equity - equity) / peak_equity
        return drawdown < self.max_drawdown_pct

    # ------------------------------------------------------------------
    # Commission
    # ------------------------------------------------------------------

    def apply_commission(self, quantity: float, price: float) -> float:
        """Calculate the commission for a trade.

        Args:
            quantity: Number of units traded.
            price: Execution price per unit.

        Returns:
            Commission amount in quote currency.
        """
        return abs(quantity) * price * self.commission_pct

    # ------------------------------------------------------------------
    # Signal validation
    # ------------------------------------------------------------------

    def validate_signal(self, signal: Signal, has_position: bool) -> bool:
        """Validate a signal against the current position state.

        Rules:
            - HOLD signals are always valid (no action needed).
            - ENTER_LONG / ENTER_SHORT: invalid if already in a position.
            - EXIT_LONG / EXIT_SHORT: invalid if not in a position.

        Args:
            signal: The signal to validate.
            has_position: Whether the account currently holds a position.

        Returns:
            True if the signal is valid and should be acted upon.
        """
        if signal.signal_type == SignalType.HOLD:
            return True

        # Don't open a new position when one is already open
        if signal.signal_type in (SignalType.ENTER_LONG, SignalType.ENTER_SHORT):
            return not has_position

        # Don't try to close a position that doesn't exist
        if signal.signal_type in (SignalType.EXIT_LONG, SignalType.EXIT_SHORT):
            return has_position

        return False
