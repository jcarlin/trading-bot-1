"""Reconstruct position timeline from raw trades."""

import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class TradeHistoryReconstructor:
    """Groups raw trades into positions and computes statistics."""

    def __init__(self):
        pass

    def reconstruct_positions(self, trades: list[dict]) -> list[dict]:
        """Group trades into positions (entry -> adds -> reduces -> exit).

        Args:
            trades: List of trade dicts sorted by timestamp with keys:
                    timestamp, symbol, side, size, price, pnl

        Returns:
            List of position dicts with keys: symbol, side, entries, exits,
            total_pnl, hold_duration, max_size, avg_entry, avg_exit
        """
        if not trades:
            return []

        # Sort by timestamp
        sorted_trades = sorted(trades, key=lambda t: t.get("timestamp", datetime.min))

        # Group by symbol
        by_symbol: dict[str, list[dict]] = {}
        for trade in sorted_trades:
            symbol = trade.get("symbol", "UNKNOWN")
            by_symbol.setdefault(symbol, []).append(trade)

        positions = []
        for symbol, symbol_trades in by_symbol.items():
            positions.extend(self._reconstruct_symbol(symbol, symbol_trades))

        return positions

    def _reconstruct_symbol(self, symbol: str, trades: list[dict]) -> list[dict]:
        """Reconstruct positions for a single symbol."""
        positions = []
        current_position: Optional[dict] = None
        net_size = 0.0

        for trade in trades:
            side = trade.get("side", "").lower()
            size = abs(trade.get("size", 0))

            if side in ("buy", "long"):
                direction = 1
            elif side in ("sell", "short"):
                direction = -1
            else:
                continue

            trade_size = size * direction

            if current_position is None or net_size == 0:
                # New position
                if current_position is not None and net_size == 0:
                    positions.append(current_position)

                current_position = {
                    "symbol": symbol,
                    "side": "long" if direction > 0 else "short",
                    "entries": [trade],
                    "exits": [],
                    "total_pnl": trade.get("pnl", 0),
                    "max_size": size,
                    "entry_time": trade.get("timestamp"),
                    "exit_time": None,
                }
                net_size = trade_size

            elif (net_size > 0 and direction > 0) or (net_size < 0 and direction < 0):
                # Adding to position
                current_position["entries"].append(trade)
                net_size += trade_size
                current_position["max_size"] = max(
                    current_position["max_size"], abs(net_size))
                current_position["total_pnl"] += trade.get("pnl", 0)

            else:
                # Reducing/closing position
                current_position["exits"].append(trade)
                net_size += trade_size
                current_position["total_pnl"] += trade.get("pnl", 0)

                if abs(net_size) < 1e-10:
                    # Position fully closed
                    net_size = 0
                    current_position["exit_time"] = trade.get("timestamp")

        # Handle any remaining open position
        if current_position is not None:
            positions.append(current_position)

        # Compute derived fields
        for pos in positions:
            pos["hold_duration"] = self._compute_hold_duration(pos)
            pos["avg_entry"] = self._compute_avg_price(pos["entries"])
            pos["avg_exit"] = self._compute_avg_price(pos["exits"])

        return positions

    def compute_trade_stats(self, positions: list[dict]) -> dict:
        """Compute aggregate statistics from reconstructed positions.

        Returns:
            Dict with total_trades, win_rate, avg_hold_hours, avg_pnl,
            max_win, max_loss, profit_factor, symbols_traded
        """
        if not positions:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_hold_hours": 0.0,
                "avg_pnl": 0.0,
                "max_win": 0.0,
                "max_loss": 0.0,
                "profit_factor": 0.0,
                "symbols_traded": 0,
            }

        pnls = [p.get("total_pnl", 0) for p in positions]
        wins = [pnl for pnl in pnls if pnl > 0]
        losses = [pnl for pnl in pnls if pnl < 0]

        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0

        hold_hours = [
            p.get("hold_duration", 0) / 3600 for p in positions
            if p.get("hold_duration") is not None
        ]

        symbols = set(p.get("symbol", "") for p in positions if p.get("symbol"))

        return {
            "total_trades": len(positions),
            "win_rate": (len(wins) / len(positions)) * 100 if positions else 0.0,
            "avg_hold_hours": sum(hold_hours) / len(hold_hours) if hold_hours else 0.0,
            "avg_pnl": sum(pnls) / len(pnls) if pnls else 0.0,
            "max_win": max(pnls) if pnls else 0.0,
            "max_loss": min(pnls) if pnls else 0.0,
            "profit_factor": total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0.0,
            "symbols_traded": len(symbols),
        }

    def _compute_hold_duration(self, position: dict) -> Optional[float]:
        """Compute hold duration in seconds."""
        entry_time = position.get("entry_time")
        exit_time = position.get("exit_time")

        if entry_time is None or exit_time is None:
            return None

        if isinstance(entry_time, datetime) and isinstance(exit_time, datetime):
            return (exit_time - entry_time).total_seconds()

        return None

    def _compute_avg_price(self, trades: list[dict]) -> float:
        """Compute volume-weighted average price from trades."""
        if not trades:
            return 0.0

        total_value = 0.0
        total_size = 0.0

        for trade in trades:
            price = trade.get("price", 0)
            size = abs(trade.get("size", 0))
            total_value += price * size
            total_size += size

        return total_value / total_size if total_size > 0 else 0.0
