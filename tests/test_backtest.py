#!/usr/bin/env python3
"""Integration test: run a backtest on synthetic data to verify the full pipeline."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from core.types import SignalType
from strategy.sma_crossover import SmaCrossoverStrategy
from risk.manager import RiskManager
from backtest.engine import BacktestEngine
from metrics.performance import PerformanceReporter


def generate_synthetic_ohlcv(n_bars: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data with a trending + mean-reverting pattern."""
    rng = np.random.default_rng(seed)

    # Generate a price series with trend + noise
    returns = rng.normal(0.0002, 0.02, n_bars)  # Slight upward drift
    # Add some trending periods
    for i in range(100, 150):
        returns[i] += 0.005  # Uptrend
    for i in range(250, 300):
        returns[i] -= 0.005  # Downtrend
    for i in range(350, 400):
        returns[i] += 0.003  # Another uptrend

    close = 50000.0 * np.cumprod(1 + returns)

    # Generate OHLCV from close
    high = close * (1 + rng.uniform(0, 0.015, n_bars))
    low = close * (1 - rng.uniform(0, 0.015, n_bars))
    open_ = close * (1 + rng.normal(0, 0.005, n_bars))
    volume = rng.uniform(100, 1000, n_bars)

    dates = pd.date_range("2024-01-01", periods=n_bars, freq="h", tz="UTC")

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)
    df.index.name = "timestamp"
    df.attrs["symbol"] = "BTC/USDT"
    return df


def test_strategy_signals():
    """Test that the strategy generates signals on synthetic data."""
    df = generate_synthetic_ohlcv()
    params = {"fast_period": 10, "slow_period": 30, "stop_loss_pct": 0.02, "take_profit_pct": 0.05}
    strategy = SmaCrossoverStrategy(params)
    strategy.setup(df)

    signals = []
    for i in range(len(df)):
        signal = strategy.generate_signal(i, df)
        if signal.signal_type != SignalType.HOLD:
            signals.append((i, signal.signal_type))

    print(f"Strategy generated {len(signals)} non-HOLD signals")
    assert len(signals) > 0, "Strategy should generate at least some signals"

    enters = [s for s in signals if s[1] == SignalType.ENTER_LONG]
    exits = [s for s in signals if s[1] == SignalType.EXIT_LONG]
    print(f"  ENTER_LONG: {len(enters)}, EXIT_LONG: {len(exits)}")


def test_risk_manager():
    """Test position sizing and validation."""
    config = {
        "position_sizing": "percent_equity",
        "max_position_pct": 0.95,
        "max_drawdown_pct": 0.20,
        "commission_pct": 0.001,
    }
    rm = RiskManager(config)

    from core.models import Signal
    signal = Signal(signal_type=SignalType.ENTER_LONG, price=50000, timestamp=pd.Timestamp.now())
    qty = rm.calculate_position_size(signal, equity=10000, current_price=50000)
    print(f"Position size for $10k equity @ $50k: {qty:.6f} BTC")
    assert qty > 0

    commission = rm.apply_commission(qty, 50000)
    print(f"Commission: ${commission:.4f}")
    assert commission > 0

    assert rm.check_drawdown(9000, 10000) is True   # 10% < 20%
    assert rm.check_drawdown(7500, 10000) is False   # 25% > 20%
    print("Risk manager checks passed")


def test_full_backtest():
    """Run a complete backtest and verify results."""
    df = generate_synthetic_ohlcv()
    params = {"fast_period": 10, "slow_period": 30, "stop_loss_pct": 0.02, "take_profit_pct": 0.05}
    strategy = SmaCrossoverStrategy(params)

    risk_config = {
        "position_sizing": "percent_equity",
        "max_position_pct": 0.95,
        "max_drawdown_pct": 0.20,
        "commission_pct": 0.001,
    }
    risk_manager = RiskManager(risk_config)
    engine = BacktestEngine(initial_capital=10000.0, commission_pct=0.001)

    result = engine.run(df, strategy, risk_manager)

    assert result.initial_capital == 10000.0
    assert len(result.equity_curve) == len(df)
    assert len(result.trades) > 0, "Should have at least one trade"

    reporter = PerformanceReporter(result)
    reporter.print_summary()

    summary = reporter.summary()
    assert summary["total_trades"] > 0
    assert 0 <= summary["win_rate"] <= 100
    assert summary["max_drawdown"] >= 0
    print("Full backtest completed successfully!")

    return summary


if __name__ == "__main__":
    print("=" * 60)
    print("  TRADING BOT v0.1 — INTEGRATION TESTS")
    print("=" * 60)

    print("\n--- Test 1: Strategy Signals ---")
    test_strategy_signals()

    print("\n--- Test 2: Risk Manager ---")
    test_risk_manager()

    print("\n--- Test 3: Full Backtest ---")
    summary = test_full_backtest()

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)
