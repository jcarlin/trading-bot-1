"""Prometheus metrics definitions for the trading system.

All metrics are module-level singletons so they can be imported and used
from any component without additional setup.
"""

from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# Data Pipeline
# ---------------------------------------------------------------------------

trades_ingested_total = Counter(
    'trades_ingested_total',
    'Total trades ingested',
    ['symbol'],
)

orderbook_updates_total = Counter(
    'orderbook_updates_total',
    'Total orderbook updates',
    ['symbol'],
)

data_latency_seconds = Histogram(
    'data_latency_seconds',
    'Data ingestion latency',
    ['source', 'symbol'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
)

data_quality_status = Gauge(
    'data_quality_status',
    'Data quality status (1=ok, 0=stale)',
    ['source', 'symbol'],
)

ws_connection_status = Gauge(
    'ws_connection_status',
    'WebSocket connection status (1=connected)',
    ['channel'],
)

ws_reconnects_total = Counter(
    'ws_reconnects_total',
    'WebSocket reconnection attempts',
    ['channel'],
)

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

orders_placed_total = Counter(
    'orders_placed_total',
    'Total orders placed',
    ['symbol', 'side', 'type'],
)

orders_filled_total = Counter(
    'orders_filled_total',
    'Total orders filled',
    ['symbol', 'side'],
)

fill_latency_seconds = Histogram(
    'fill_latency_seconds',
    'Order fill latency',
    ['symbol'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
)

slippage_bps = Histogram(
    'slippage_bps',
    'Execution slippage in basis points',
    ['symbol'],
    buckets=[0.5, 1, 2, 5, 10, 25, 50, 100],
)

orders_rejected_total = Counter(
    'orders_rejected_total',
    'Orders rejected by risk controls',
    ['reason'],
)

# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

portfolio_equity_usd = Gauge(
    'portfolio_equity_usd',
    'Total portfolio equity in USD',
)

portfolio_drawdown_pct = Gauge(
    'portfolio_drawdown_pct',
    'Current portfolio drawdown percentage',
)

portfolio_unrealized_pnl_usd = Gauge(
    'portfolio_unrealized_pnl_usd',
    'Unrealized PnL in USD',
)

# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------

risk_check_passed_total = Counter(
    'risk_check_passed_total',
    'Risk checks passed',
    ['check_type'],
)

risk_check_failed_total = Counter(
    'risk_check_failed_total',
    'Risk checks failed',
    ['check_type'],
)

circuit_breaker_status = Gauge(
    'circuit_breaker_status',
    'Circuit breaker status (1=active)',
)

kill_switch_activations_total = Counter(
    'kill_switch_activations_total',
    'Kill switch activations',
)

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

api_requests_total = Counter(
    'api_requests_total',
    'API requests made',
    ['exchange', 'endpoint'],
)

api_rate_limit_remaining = Gauge(
    'api_rate_limit_remaining',
    'API rate limit remaining',
    ['exchange'],
)

api_errors_total = Counter(
    'api_errors_total',
    'API errors',
    ['exchange', 'error_type'],
)
