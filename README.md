# Systematic Trading Bot v0.1

A modular, rule-based trading system for backtesting and paper trading crypto.

**This is a learning/experimentation system using play money only.**

## Architecture

```
trading/
├── config/default.yaml        # All configuration in one place
├── core/
│   ├── types.py               # Enums: Side, OrderType, SignalType, etc.
│   └── models.py              # Data models: Bar, Signal, Order, Position, Trade
├── strategy/
│   ├── base.py                # BaseStrategy ABC (implement this for new strategies)
│   └── sma_crossover.py       # Example: SMA crossover with stop loss / take profit
├── risk/
│   └── manager.py             # Position sizing, drawdown checks, signal validation
├── data/
│   └── provider.py            # OHLCV data fetching via ccxt, CSV loading
├── exchange/
│   └── ccxt_exchange.py       # Order execution via ccxt (supports 100+ exchanges)
├── backtest/
│   └── engine.py              # Bar-by-bar backtest engine
├── metrics/
│   └── performance.py         # Sharpe, drawdown, win rate, HTML reports
├── paper/
│   └── trader.py              # Live paper trading loop
├── run_backtest.py            # CLI: run a backtest
├── run_paper.py               # CLI: run paper trading
└── tests/
    └── test_backtest.py       # Integration tests
```

**Key design principle:** The same strategy runs in both backtest and paper trading modes without modification.

## Quick Start

### 1. Install

```bash
cd trading
python3 -m venv .venv
source .venv/bin/activate   # or: .venv/bin/activate.fish
pip install -r requirements.txt
```

### 2. Run a Backtest

```bash
# Uses default config (BTC/USDT, SMA crossover, Binance testnet data)
python run_backtest.py

# With custom config
python run_backtest.py --config config/my_strategy.yaml

# Generate an HTML performance report
python run_backtest.py --report
```

### 3. Run Paper Trading

```bash
# Set exchange API keys (Binance testnet: https://testnet.binance.vision)
export EXCHANGE_API_KEY="your_testnet_key"
export EXCHANGE_API_SECRET="your_testnet_secret"

# Start paper trading (sandbox mode enforced)
python run_paper.py
```

Press `Ctrl+C` to stop — a trade summary will be printed.

### 4. Run Tests

```bash
python tests/test_backtest.py
```

## Configuration

All settings are in `config/default.yaml`:

```yaml
exchange:
  name: binance         # Any ccxt-supported exchange
  sandbox: true         # MUST be true for paper trading

symbol: BTC/USDT
timeframe: 1h

strategy:
  name: sma_crossover
  params:
    fast_period: 20
    slow_period: 50
    stop_loss_pct: 0.02
    take_profit_pct: 0.05

risk:
  position_sizing: percent_equity
  max_position_pct: 0.95
  max_drawdown_pct: 0.20
  commission_pct: 0.001

backtest:
  initial_capital: 10000.0
  start_date: "2024-01-01"
  end_date: "2025-01-01"
```

Environment variable overrides: `EXCHANGE_API_KEY`, `EXCHANGE_API_SECRET`, `TRADING_SYMBOL`, `TRADING_TIMEFRAME`, `BACKTEST_INITIAL_CAPITAL`, `BACKTEST_START_DATE`, `BACKTEST_END_DATE`.

## Adding a New Strategy

1. Create `strategy/my_strategy.py`:

```python
from strategy.base import BaseStrategy
from core.models import Signal
from core.types import SignalType
import pandas as pd

class MyStrategy(BaseStrategy):
    def setup(self, df: pd.DataFrame) -> None:
        # Compute indicators on full DataFrame
        self.indicators["rsi"] = ...  # your indicator

    def generate_signal(self, index: int, df: pd.DataFrame) -> Signal:
        # Generate signal for bar at index (no looking forward!)
        if self.indicators["rsi"].iloc[index] < 30:
            return Signal(
                signal_type=SignalType.ENTER_LONG,
                price=df["close"].iloc[index],
                timestamp=df.index[index],
                stop_loss=df["close"].iloc[index] * 0.98,
                take_profit=df["close"].iloc[index] * 1.06,
            )
        return self.hold_signal(df, index)
```

2. Update config:

```yaml
strategy:
  name: my_strategy   # matches filename, class = MyStrategy + "Strategy" -> not needed
  params:
    rsi_period: 14
```

That's it — the dynamic loader will import `strategy.my_strategy.MyStrategyStrategy`.

**Naming convention:** filename `foo_bar.py` → class `FooBarStrategy`.

## Swapping Exchanges

Change one line in config:

```yaml
exchange:
  name: bybit        # or: kraken, okx, coinbase, alpaca, etc.
  sandbox: true
```

ccxt supports 100+ exchanges. Set the appropriate API keys and sandbox will auto-switch to the exchange's testnet.

## Performance Metrics

The backtest produces:
- **Win rate** — percentage of profitable trades
- **Sharpe ratio** — risk-adjusted returns (annualized, 365 days for crypto)
- **Max drawdown** — largest peak-to-trough decline
- **Profit factor** — gross profit / gross loss
- **Equity curve** — timestamped equity values
- **HTML report** — full tearsheet via quantstats (use `--report` flag)

## Tech Stack

| Component | Library |
|-----------|---------|
| Exchange connectivity | [ccxt](https://github.com/ccxt/ccxt) (108+ exchanges) |
| Data handling | pandas, numpy |
| Performance metrics | [quantstats](https://github.com/ranaroussi/quantstats) |
| Configuration | PyYAML |
| Indicators | pandas rolling (extensible with ta-lib or pandas-ta) |

## Safety

- Paper trading **enforces sandbox mode** — refuses to run with `sandbox: false`
- No real money is ever at risk
- Max drawdown circuit breaker stops trading automatically
- All trades are logged
