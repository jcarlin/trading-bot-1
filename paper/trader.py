"""Paper trading loop for live-simulated strategy execution.

Connects to a sandbox exchange, fetches live bars, runs a strategy,
and executes paper trades via the exchange's testnet.
"""

import importlib
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import ccxt

from core.models import Fill, Order, Position, Signal, Trade
from core.types import OrderType, Side, SignalType
from data.provider import DataProvider
from exchange.ccxt_exchange import CcxtExchange
from risk.manager import RiskManager
from strategy.base import BaseStrategy

logger = logging.getLogger(__name__)


def _load_strategy(config: dict) -> BaseStrategy:
    """Dynamically load a strategy class based on config.

    Converts a snake_case strategy name (e.g. "sma_crossover") to the
    module path ``strategy.sma_crossover`` and class name
    ``SmaCrossoverStrategy``.

    Args:
        config: The full config dict containing strategy.name and
                strategy.params.

    Returns:
        An instantiated strategy object.
    """
    name = config["strategy"]["name"]  # e.g. "sma_crossover"
    params = config["strategy"].get("params", {})

    # Build the class name: sma_crossover -> SmaCrossover -> SmaCrossoverStrategy
    class_name = "".join(part.capitalize() for part in name.split("_")) + "Strategy"
    module_path = f"strategy.{name}"

    module = importlib.import_module(module_path)
    strategy_class = getattr(module, class_name)
    return strategy_class(params)


class PaperTrader:
    """Paper trading engine that runs a strategy against a live sandbox exchange.

    Attributes:
        config: Full configuration dict.
        symbol: Trading pair (e.g. "BTC/USDT").
        timeframe: OHLCV bar timeframe (e.g. "1h").
        data_provider: Fetches live OHLCV bars.
        exchange: Places orders on the sandbox exchange.
        strategy: The active trading strategy.
        risk_manager: Validates signals and sizes positions.
        cash: Available cash in quote currency.
        equity: Total account equity (cash + position value).
        peak_equity: Highest equity observed (for drawdown tracking).
        position: Current open position, or None.
        trades: List of completed trades.
    """

    def __init__(self, config: dict):
        self.config = config
        self.symbol: str = config["symbol"]
        self.timeframe: str = config["timeframe"]
        self.poll_interval: int = int(config["paper"].get("poll_interval_seconds", 60))

        # Exchange connection details
        ex_cfg = config["exchange"]
        exchange_id = ex_cfg["name"]
        sandbox = ex_cfg.get("sandbox", True)
        api_key = ex_cfg.get("api_key", "")
        api_secret = ex_cfg.get("api_secret", "")

        # Initialize components
        self.data_provider = DataProvider(
            exchange_id=exchange_id,
            sandbox=sandbox,
            api_key=api_key,
            api_secret=api_secret,
        )
        self.exchange = CcxtExchange(
            exchange_id=exchange_id,
            sandbox=sandbox,
            api_key=api_key,
            api_secret=api_secret,
        )
        self.strategy: BaseStrategy = _load_strategy(config)
        self.risk_manager = RiskManager(config.get("risk", {}))

        # Account state
        initial_capital = float(config["paper"].get("initial_capital", 10000.0))
        self.cash: float = initial_capital
        self.equity: float = initial_capital
        self.peak_equity: float = initial_capital
        self.position: Optional[Position] = None
        self.trades: list[Trade] = []

        logger.info(
            "PaperTrader initialised: symbol=%s timeframe=%s capital=%.2f",
            self.symbol,
            self.timeframe,
            initial_capital,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the paper trading loop until interrupted.

        On each iteration:
        1. Fetch the latest bars and run the strategy.
        2. Check stop-loss / take-profit against the live price.
        3. Act on entry/exit signals.
        4. Log the current state.
        5. Sleep until the next poll.
        """
        logger.info("Starting paper trading loop (poll every %ds)...", self.poll_interval)
        print(f"\n{'='*60}")
        print(f"  Paper Trading: {self.symbol} ({self.timeframe})")
        print(f"  Strategy: {self.config['strategy']['name']}")
        print(f"  Initial Capital: ${self.equity:,.2f}")
        print(f"{'='*60}\n")

        try:
            while True:
                try:
                    self._tick()
                except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as exc:
                    logger.warning("Exchange connectivity issue, will retry: %s", exc)
                except ccxt.BaseError as exc:
                    logger.error("Exchange error: %s", exc)
                except Exception:
                    logger.exception("Unexpected error in trading loop")

                time.sleep(self.poll_interval)

        except KeyboardInterrupt:
            print("\n\nTrading stopped by user.")
            self._print_summary()

    # ------------------------------------------------------------------
    # Single iteration
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        """Execute a single iteration of the trading loop."""
        now = datetime.now(timezone.utc)

        # 1. Fetch latest bars
        df = self.data_provider.get_latest_bars(
            self.symbol, self.timeframe, count=100
        )
        if df.empty:
            logger.warning("No bar data received, skipping tick.")
            return

        # 2. Compute strategy indicators
        self.strategy.setup(df)

        # 3. Generate signal for the most recent bar
        last_index = len(df) - 1
        signal = self.strategy.generate_signal(last_index, df)

        # 4. Get current live price
        ticker = self.exchange.get_ticker(self.symbol)
        current_price = float(ticker["last"])

        # 5. Check stop-loss / take-profit
        stopped = self._check_stops(current_price)

        # 6. Act on signals (only if stops didn't already close the position)
        if not stopped and self.risk_manager.validate_signal(signal, self.position is not None):
            if signal.signal_type == SignalType.ENTER_LONG and self.position is None:
                self._open_position(signal, current_price)
            elif signal.signal_type == SignalType.EXIT_LONG and self.position is not None:
                self._close_position("signal_exit", current_price)

        # 7. Update equity
        self._update_equity(current_price)

        # 8. Check drawdown limit
        if not self.risk_manager.check_drawdown(self.equity, self.peak_equity):
            logger.warning("Max drawdown breached! Stopping trading.")
            print("\n*** MAX DRAWDOWN BREACHED — stopping. ***")
            self._print_summary()
            raise KeyboardInterrupt  # Exit the loop cleanly

        # 9. Log status
        self._log_status(now, current_price, signal)

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _check_stops(self, current_price: float) -> bool:
        """Check stop-loss and take-profit levels; close position if hit.

        Args:
            current_price: The latest market price.

        Returns:
            True if the position was closed by a stop.
        """
        if self.position is None:
            return False

        # Stop-loss check
        if self.position.stop_loss is not None and current_price <= self.position.stop_loss:
            logger.info(
                "Stop-loss triggered at %.2f (stop=%.2f)",
                current_price,
                self.position.stop_loss,
            )
            self._close_position("stop_loss", current_price)
            return True

        # Take-profit check
        if self.position.take_profit is not None and current_price >= self.position.take_profit:
            logger.info(
                "Take-profit triggered at %.2f (tp=%.2f)",
                current_price,
                self.position.take_profit,
            )
            self._close_position("take_profit", current_price)
            return True

        return False

    def _open_position(self, signal: Signal, current_price: float) -> None:
        """Open a new long position based on the signal.

        Calculates position size via the risk manager, places a market buy
        order, and records the resulting position.

        Args:
            signal: The ENTER_LONG signal.
            current_price: Current market price for sizing.
        """
        quantity = self.risk_manager.calculate_position_size(
            signal, self.equity, current_price
        )
        if quantity <= 0:
            logger.warning("Position size is zero; skipping entry.")
            return

        order = Order(
            symbol=self.symbol,
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=quantity,
            timestamp=datetime.now(timezone.utc),
        )

        try:
            fill: Fill = self.exchange.place_order(order)
        except ccxt.BaseError as exc:
            logger.error("Failed to open position: %s", exc)
            return

        commission = self.risk_manager.apply_commission(fill.quantity, fill.fill_price)
        cost = fill.quantity * fill.fill_price + commission
        self.cash -= cost

        self.position = Position(
            symbol=self.symbol,
            side=Side.BUY,
            entry_price=fill.fill_price,
            quantity=fill.quantity,
            entry_time=fill.timestamp,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )

        logger.info(
            "Opened LONG: %.6f %s @ %.2f (cost=%.2f, commission=%.4f)",
            fill.quantity,
            self.symbol,
            fill.fill_price,
            cost,
            commission,
        )
        print(
            f"  >>> OPEN LONG: {fill.quantity:.6f} {self.symbol} "
            f"@ ${fill.fill_price:,.2f}  |  SL=${signal.stop_loss or 0:,.2f}  "
            f"TP=${signal.take_profit or 0:,.2f}"
        )

    def _close_position(self, reason: str, price: float) -> None:
        """Close the current position and record the trade.

        Args:
            reason: Why the position is being closed (e.g. "signal_exit",
                    "stop_loss", "take_profit").
            price: The price at which to close.
        """
        if self.position is None:
            return

        order = Order(
            symbol=self.symbol,
            side=Side.SELL,
            order_type=OrderType.MARKET,
            quantity=self.position.quantity,
            timestamp=datetime.now(timezone.utc),
        )

        try:
            fill: Fill = self.exchange.place_order(order)
        except ccxt.BaseError as exc:
            logger.error("Failed to close position: %s", exc)
            return

        commission = self.risk_manager.apply_commission(fill.quantity, fill.fill_price)
        proceeds = fill.quantity * fill.fill_price - commission
        self.cash += proceeds

        pnl = (fill.fill_price - self.position.entry_price) * self.position.quantity - commission
        trade = Trade(
            symbol=self.symbol,
            side=Side.BUY,
            entry_price=self.position.entry_price,
            exit_price=fill.fill_price,
            quantity=self.position.quantity,
            entry_time=self.position.entry_time,
            exit_time=fill.timestamp,
            pnl=pnl,
            commission=commission,
            exit_reason=reason,
        )
        self.trades.append(trade)

        logger.info(
            "Closed LONG (%s): %.6f %s @ %.2f -> %.2f  PnL=%.2f",
            reason,
            self.position.quantity,
            self.symbol,
            self.position.entry_price,
            fill.fill_price,
            pnl,
        )
        print(
            f"  <<< CLOSE LONG ({reason}): {self.position.quantity:.6f} {self.symbol} "
            f"@ ${fill.fill_price:,.2f}  |  PnL=${pnl:,.2f}"
        )

        self.position = None

    # ------------------------------------------------------------------
    # Equity tracking
    # ------------------------------------------------------------------

    def _update_equity(self, current_price: float) -> None:
        """Recalculate equity based on cash and open position value."""
        position_value = 0.0
        if self.position is not None:
            self.position.update_pnl(current_price)
            position_value = self.position.quantity * current_price

        self.equity = self.cash + position_value
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

    # ------------------------------------------------------------------
    # Logging / output
    # ------------------------------------------------------------------

    def _log_status(
        self, timestamp: datetime, price: float, signal: Signal
    ) -> None:
        """Print a one-line status update for the current tick."""
        pos_str = "FLAT"
        if self.position is not None:
            pos_str = (
                f"LONG {self.position.quantity:.6f} "
                f"@ ${self.position.entry_price:,.2f} "
                f"(uPnL=${self.position.unrealized_pnl:,.2f})"
            )

        drawdown = 0.0
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - self.equity) / self.peak_equity

        print(
            f"  [{timestamp:%H:%M:%S}] "
            f"Price=${price:,.2f}  "
            f"Signal={signal.signal_type.value:<11s}  "
            f"Position={pos_str}  "
            f"Equity=${self.equity:,.2f}  "
            f"DD={drawdown:.1%}"
        )

    def _print_summary(self) -> None:
        """Print a summary of all trades executed during the session."""
        print(f"\n{'='*60}")
        print("  PAPER TRADING SUMMARY")
        print(f"{'='*60}")
        print(f"  Total trades: {len(self.trades)}")

        if not self.trades:
            print("  No trades executed.")
            print(f"  Final equity: ${self.equity:,.2f}")
            print(f"{'='*60}\n")
            return

        winners = [t for t in self.trades if t.is_winner]
        losers = [t for t in self.trades if not t.is_winner]
        total_pnl = sum(t.pnl for t in self.trades)
        total_commission = sum(t.commission for t in self.trades)

        print(f"  Winners: {len(winners)}  |  Losers: {len(losers)}")
        if self.trades:
            print(f"  Win rate: {len(winners) / len(self.trades):.1%}")
        print(f"  Total PnL: ${total_pnl:,.2f}")
        print(f"  Total commission: ${total_commission:,.2f}")
        print(f"  Final equity: ${self.equity:,.2f}")

        print(f"\n  {'Exit Reason':<15s} {'Entry':>10s} {'Exit':>10s} {'PnL':>10s} {'Return':>8s}")
        print(f"  {'-'*55}")
        for t in self.trades:
            print(
                f"  {t.exit_reason:<15s} "
                f"${t.entry_price:>9,.2f} "
                f"${t.exit_price:>9,.2f} "
                f"${t.pnl:>9,.2f} "
                f"{t.return_pct:>7.1%}"
            )

        print(f"{'='*60}\n")
