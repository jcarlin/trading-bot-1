"""Backtest-vs-live performance comparison for strategy drift detection."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from strategy.live_strategy import LiveStrategy, MarketState
from core.types import SignalType

logger = logging.getLogger(__name__)


class BacktestLiveComparator:
    """Replays recent history through strategy logic and compares to actual fills.

    Detects performance decay between backtested and live results.
    Per CLAUDE.md: decay > 50% triggers automatic strategy pause.
    """

    def __init__(self, strategy: LiveStrategy, timescale, redis_store,
                 strategy_name: str, symbol: str = "BTC/USDC"):
        self.strategy = strategy
        self.timescale = timescale
        self.redis_store = redis_store
        self.strategy_name = strategy_name
        self.symbol = symbol

    def compare(self, lookback_hours: int = 168) -> dict:
        """Compare backtested vs live performance.

        Args:
            lookback_hours: Hours of history to replay.

        Returns:
            dict with comparison metrics and pause recommendation.
        """
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=lookback_hours)

        # Fetch historical data
        try:
            candles = self.timescale.query_candles(
                self.symbol, "1h", start, end)
        except Exception:
            logger.exception("Failed to query candles for backtest comparison")
            candles = []

        try:
            funding_rates = self.timescale.query_funding_rates(
                self.symbol, start, end)
        except Exception:
            logger.exception("Failed to query funding rates")
            funding_rates = []

        try:
            live_fills = self.timescale.query_fills_by_strategy(
                self.strategy_name, start, end)
        except Exception:
            logger.exception("Failed to query live fills")
            live_fills = []

        if not candles:
            return self._empty_comparison()

        # Replay strategy on historical data
        simulated_signals = self._replay_strategy(candles, funding_rates)

        # Compute backtested PnL from simulated signals
        backtest_pnl = self._estimate_pnl_from_signals(simulated_signals, candles)
        backtest_trade_count = sum(
            1 for s in simulated_signals
            if s["signal_type"] != SignalType.HOLD.value
        )

        # Compute live PnL from actual fills
        live_pnl = sum(float(f.get("closed_pnl", 0.0)) for f in live_fills)
        live_trade_count = len(live_fills)

        # Compute decay
        if backtest_pnl > 0:
            decay_pct = (1.0 - live_pnl / backtest_pnl) * 100
        elif backtest_pnl < 0 and live_pnl < 0:
            # Both negative: less loss in live = negative decay (good)
            decay_pct = (1.0 - live_pnl / backtest_pnl) * 100
        else:
            decay_pct = 0.0

        # Entry price deviation
        entry_deviation = self._compute_entry_deviation(
            simulated_signals, live_fills)

        should_pause = decay_pct > 50.0

        return {
            "backtest_pnl": round(backtest_pnl, 4),
            "live_pnl": round(live_pnl, 4),
            "decay_pct": round(decay_pct, 2),
            "should_pause": should_pause,
            "backtest_trade_count": backtest_trade_count,
            "live_trade_count": live_trade_count,
            "entry_price_deviation_pct": round(entry_deviation, 4),
        }

    def _replay_strategy(self, candles: list[dict],
                         funding_rates: list[dict]) -> list[dict]:
        """Replay strategy on historical candle data.

        Builds simplified MarketState from each candle and runs on_tick().
        """
        # Build funding rate lookup by time
        funding_lookup = {}
        for fr in funding_rates:
            funding_lookup[fr.get("time")] = fr

        signals = []
        # Save and restore strategy state to avoid side effects
        original_state = self.strategy.get_state()

        try:
            # Reset strategy for clean replay
            self.strategy.set_state({})

            for candle in candles:
                price = float(candle.get("close", 0))
                timestamp = candle.get("time", datetime.now(timezone.utc))

                # Find closest funding rate
                fr = self._find_closest_funding(timestamp, funding_rates)
                funding_rate = float(fr.get("rate", 0)) if fr else 0.0
                premium = float(fr.get("premium", 0)) if fr else 0.0

                market_state = MarketState(
                    mark_price=price,
                    mid_price=price,
                    bid=price * 0.9999,
                    ask=price * 1.0001,
                    funding_rate=funding_rate,
                    premium=premium,
                    open_interest=1000000.0,
                    funding_history=funding_rates,
                    position=None,
                    equity=10000.0,
                    cash=10000.0,
                    drawdown_pct=0.0,
                    recent_candles=candles,
                    spread=price * 0.0002,
                    bid_depth=0.0,
                    ask_depth=0.0,
                    timestamp=timestamp,
                )

                try:
                    signal = self.strategy.on_tick(market_state)
                    signals.append({
                        "signal_type": signal.signal_type.value,
                        "price": signal.price,
                        "timestamp": timestamp,
                        "size": signal.size,
                    })
                except Exception:
                    logger.debug("Strategy error during replay at %s", timestamp)

        finally:
            # Restore original state
            self.strategy.set_state(original_state)

        return signals

    def _estimate_pnl_from_signals(self, signals: list[dict],
                                    candles: list[dict]) -> float:
        """Estimate PnL from simulated signals."""
        pnl = 0.0
        entry_price = None
        entry_side = None

        for signal in signals:
            sig_type = signal["signal_type"]
            price = signal["price"]

            if sig_type in ("enter_long", "enter_short") and entry_price is None:
                entry_price = price
                entry_side = "long" if sig_type == "enter_long" else "short"

            elif sig_type in ("exit_long", "exit_short") and entry_price is not None:
                if entry_side == "long":
                    pnl += price - entry_price
                else:
                    pnl += entry_price - price
                entry_price = None
                entry_side = None

        return pnl

    def _find_closest_funding(self, timestamp, funding_rates: list[dict]):
        """Find the funding rate closest to a given timestamp."""
        if not funding_rates:
            return None

        closest = None
        min_diff = float("inf")
        for fr in funding_rates:
            fr_time = fr.get("time")
            if fr_time is None:
                continue
            try:
                diff = abs((fr_time - timestamp).total_seconds())
                if diff < min_diff:
                    min_diff = diff
                    closest = fr
            except (TypeError, AttributeError):
                continue
        return closest

    def _compute_entry_deviation(self, simulated_signals: list[dict],
                                  live_fills: list[dict]) -> float:
        """Compute average entry price deviation between simulated and live."""
        sim_entries = [
            s for s in simulated_signals
            if s["signal_type"] in ("enter_long", "enter_short")
        ]
        live_entries = [
            f for f in live_fills
            if float(f.get("closed_pnl", 0)) == 0
        ]

        if not sim_entries or not live_entries:
            return 0.0

        # Compare up to min(len(sim), len(live)) entries
        deviations = []
        for i in range(min(len(sim_entries), len(live_entries))):
            sim_price = sim_entries[i]["price"]
            live_price = float(live_entries[i].get("price", 0))
            if sim_price > 0 and live_price > 0:
                dev = abs(sim_price - live_price) / sim_price * 100
                deviations.append(dev)

        return sum(deviations) / len(deviations) if deviations else 0.0

    @staticmethod
    def _empty_comparison() -> dict:
        return {
            "backtest_pnl": 0.0,
            "live_pnl": 0.0,
            "decay_pct": 0.0,
            "should_pause": False,
            "backtest_trade_count": 0,
            "live_trade_count": 0,
            "entry_price_deviation_pct": 0.0,
        }
