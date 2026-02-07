"""Bar-by-bar backtesting engine.

Iterates through historical OHLCV data one bar at a time, executing a
strategy's signals against a simulated portfolio.  Tracks equity, handles
stop-loss / take-profit exits, and collects completed trades.
"""

from dataclasses import dataclass, field
import logging
from typing import Optional

import pandas as pd

from core.models import Position, Signal, Trade
from core.types import Side, SignalType
from strategy.base import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest output data."""

    trades: list[Trade]
    equity_curve: pd.Series  # Indexed by timestamp, values are total equity
    initial_capital: float
    final_equity: float


class BacktestEngine:
    """Simulates strategy execution on historical OHLCV bars.

    The engine walks through the DataFrame bar-by-bar, calling the strategy
    for signals and applying simple fill assumptions (fill at close or at
    stop/take-profit level within the bar's range).

    Args:
        initial_capital: Starting cash balance.
        commission_pct: Round-trip commission as a decimal fraction
                        (e.g. 0.001 = 0.1 %).
    """

    def __init__(self, initial_capital: float, commission_pct: float = 0.001):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        strategy: BaseStrategy,
        risk_manager,
    ) -> BacktestResult:
        """Run the backtest over *df* using *strategy* and *risk_manager*.

        Args:
            df: OHLCV DataFrame with a DatetimeIndex and columns
                [open, high, low, close, volume].
            strategy: A concrete BaseStrategy whose ``setup`` and
                ``generate_signal`` methods will be called.
            risk_manager: Object with a ``calculate_position_size(equity,
                price, stop_loss)`` method returning a float quantity.

        Returns:
            A BacktestResult with completed trades, an equity curve, and
            capital figures.
        """
        # 1. Let the strategy compute its indicators once on the full data.
        strategy.setup(df)

        cash: float = self.initial_capital
        position: Optional[Position] = None
        trades: list[Trade] = []
        equity_values: list[float] = []
        equity_timestamps: list = []

        symbol = df.attrs.get("symbol", "UNKNOWN")

        # 2. Iterate bar-by-bar.
        for i in range(len(df)):
            bar_open = df["open"].iloc[i]
            bar_high = df["high"].iloc[i]
            bar_low = df["low"].iloc[i]
            bar_close = df["close"].iloc[i]
            bar_ts = df.index[i]

            # --- 2a. Check stop-loss / take-profit on the current bar ---
            if position is not None:
                closed = self._check_sl_tp(
                    position, bar_high, bar_low, bar_ts, cash, trades, symbol
                )
                if closed:
                    cash = closed  # updated cash after closing
                    position = None

            # --- 2b. Generate a signal from the strategy ---
            signal: Signal = strategy.generate_signal(i, df)

            # --- 2c. Act on the signal ---
            if signal.signal_type == SignalType.ENTER_LONG and position is None:
                entry_price = bar_close
                stop_loss = signal.stop_loss
                take_profit = signal.take_profit

                qty = risk_manager.calculate_position_size(
                    signal=signal,
                    equity=cash,
                    current_price=entry_price,
                )

                if qty > 0:
                    cost = qty * entry_price
                    commission = cost * self.commission_pct
                    cash -= cost + commission

                    position = Position(
                        symbol=symbol,
                        side=Side.BUY,
                        entry_price=entry_price,
                        quantity=qty,
                        entry_time=bar_ts,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                    )
                    logger.debug(
                        "ENTER_LONG @ %.4f  qty=%.6f  sl=%.4f  tp=%.4f  ts=%s",
                        entry_price,
                        qty,
                        stop_loss or 0,
                        take_profit or 0,
                        bar_ts,
                    )

            elif signal.signal_type == SignalType.EXIT_LONG and position is not None:
                cash = self._close_position(
                    position, bar_close, bar_ts, "signal", cash, trades
                )
                position = None

            # --- 2d. Record equity ---
            equity = cash
            if position is not None:
                equity += position.quantity * bar_close
            equity_values.append(equity)
            equity_timestamps.append(bar_ts)

        # 3. Close any remaining position at the last bar's close.
        if position is not None:
            last_close = df["close"].iloc[-1]
            last_ts = df.index[-1]
            cash = self._close_position(
                position, last_close, last_ts, "end_of_data", cash, trades
            )
            # Update the final equity point.
            equity_values[-1] = cash

        equity_curve = pd.Series(
            equity_values, index=pd.DatetimeIndex(equity_timestamps), name="equity"
        )

        result = BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            initial_capital=self.initial_capital,
            final_equity=equity_values[-1] if equity_values else self.initial_capital,
        )

        logger.info(
            "Backtest complete: %d trades, final equity = %.2f",
            len(trades),
            result.final_equity,
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_sl_tp(
        self,
        position: Position,
        bar_high: float,
        bar_low: float,
        bar_ts,
        cash: float,
        trades: list[Trade],
        symbol: str,
    ) -> Optional[float]:
        """Check if the bar triggers a stop-loss or take-profit exit.

        Returns the updated cash balance if the position was closed,
        otherwise ``None``.

        When both SL and TP could be hit in the same bar, stop-loss wins
        (conservative assumption — adverse move assumed to happen first).
        """
        sl_hit = position.stop_loss is not None and bar_low <= position.stop_loss
        tp_hit = position.take_profit is not None and bar_high >= position.take_profit

        if sl_hit:
            return self._close_position(
                position, position.stop_loss, bar_ts, "stop_loss", cash, trades
            )
        if tp_hit:
            return self._close_position(
                position, position.take_profit, bar_ts, "take_profit", cash, trades
            )
        return None

    def _close_position(
        self,
        position: Position,
        exit_price: float,
        exit_time,
        exit_reason: str,
        cash: float,
        trades: list[Trade],
    ) -> float:
        """Close *position* at *exit_price* and record a Trade.

        Returns the updated cash balance.
        """
        proceeds = position.quantity * exit_price
        commission = proceeds * self.commission_pct
        pnl = (exit_price - position.entry_price) * position.quantity - commission
        # Also subtract entry commission already taken from cash on open,
        # so pnl here reflects only the exit side commission.
        # Entry commission was already deducted from cash at open time.

        trade = Trade(
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            entry_time=position.entry_time,
            exit_time=exit_time,
            pnl=pnl,
            commission=commission + (position.quantity * position.entry_price * self.commission_pct),
            exit_reason=exit_reason,
        )
        trades.append(trade)

        logger.debug(
            "CLOSE %s @ %.4f  pnl=%.4f  reason=%s  ts=%s",
            position.symbol,
            exit_price,
            pnl,
            exit_reason,
            exit_time,
        )

        return cash + proceeds - commission
