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

# ---------------------------------------------------------------------------
# Strategy Performance
# ---------------------------------------------------------------------------

strategy_sharpe_ratio = Gauge(
    'strategy_sharpe_ratio',
    'Strategy Sharpe ratio',
    ['strategy_name', 'window'],
)

strategy_pnl_total = Gauge(
    'strategy_pnl_total',
    'Strategy total PnL',
    ['strategy_name'],
)

strategy_max_drawdown = Gauge(
    'strategy_max_drawdown',
    'Strategy max drawdown percentage',
    ['strategy_name', 'window'],
)

strategy_win_rate = Gauge(
    'strategy_win_rate',
    'Strategy win rate percentage',
    ['strategy_name', 'window'],
)

strategy_profit_factor = Gauge(
    'strategy_profit_factor',
    'Strategy profit factor',
    ['strategy_name', 'window'],
)

strategy_signal_count = Counter(
    'strategy_signal_count',
    'Strategy signals generated',
    ['strategy_name', 'signal_type'],
)

# ---------------------------------------------------------------------------
# Evaluation & Self-Assessment (Phase 2)
# ---------------------------------------------------------------------------

strategy_health_score = Gauge(
    'strategy_health_score',
    'Strategy health score (0-100)',
    ['strategy_name'],
)

market_regime_indicator = Gauge(
    'market_regime_indicator',
    'Market regime numeric indicator (1=trending_up, 2=trending_down, 3=ranging, 4=volatile)',
    ['symbol'],
)

backtest_live_decay_pct = Gauge(
    'backtest_live_decay_pct',
    'Backtest vs live performance decay percentage',
    ['strategy_name'],
)

signal_accuracy_pct = Gauge(
    'signal_accuracy_pct',
    'Signal accuracy percentage',
    ['strategy_name'],
)

evaluation_cycle_duration_seconds = Histogram(
    'evaluation_cycle_duration_seconds',
    'Time spent on evaluation cycles',
    ['checkpoint_type'],
    buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)

# ---------------------------------------------------------------------------
# Portfolio & Multi-Strategy (Phase 3)
# ---------------------------------------------------------------------------

portfolio_sharpe_ratio = Gauge(
    'portfolio_sharpe_ratio',
    'Portfolio-level Sharpe ratio',
)

portfolio_total_pnl = Gauge(
    'portfolio_total_pnl',
    'Portfolio total PnL across all strategies',
)

strategy_correlation = Gauge(
    'strategy_correlation',
    'Pairwise correlation between strategy returns',
    ['strategy_a', 'strategy_b'],
)

portfolio_capital_efficiency = Gauge(
    'portfolio_capital_efficiency',
    'Portfolio capital efficiency (return per unit equity)',
)

# ---------------------------------------------------------------------------
# Wallet Intelligence (Phase 3)
# ---------------------------------------------------------------------------

wallet_discovery_count = Gauge(
    'wallet_discovery_count',
    'Number of wallets discovered in latest cycle',
)

wallet_top_score = Gauge(
    'wallet_top_score',
    'Highest wallet score from latest discovery',
)

wallet_analysis_duration_seconds = Histogram(
    'wallet_analysis_duration_seconds',
    'Time spent on wallet analysis',
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0],
)

# ---------------------------------------------------------------------------
# Phase 4: Full Autonomy
# ---------------------------------------------------------------------------

ai_decision_count = Counter(
    'ai_decision_count',
    'AI decision engine decisions made',
    ['action', 'source'],
)

ai_decision_confidence = Histogram(
    'ai_decision_confidence',
    'AI decision confidence distribution',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

allocation_weight = Gauge(
    'allocation_weight',
    'Current allocation weight per strategy',
    ['strategy_name'],
)

shadow_runner_pnl = Gauge(
    'shadow_runner_pnl',
    'Shadow runner cumulative PnL',
    ['shadow_name'],
)

orchestrator_accuracy_pct = Gauge(
    'orchestrator_accuracy_pct',
    'Orchestrator decision accuracy percentage',
)

stress_test_status = Gauge(
    'stress_test_status',
    'Stress test pass status (1=all pass, 0=failures)',
)

# ---------------------------------------------------------------------------
# Phase 5: Execution Algorithms
# ---------------------------------------------------------------------------

algo_execution_count = Counter(
    'algo_execution_count',
    'Execution algorithm invocations',
    ['algo_name', 'symbol'],
)

algo_slippage_bps = Histogram(
    'algo_slippage_improvement_bps',
    'Slippage improvement from execution algorithms in basis points',
    ['algo_name'],
    buckets=[0.5, 1, 2, 5, 10, 25, 50, 100],
)

algo_fill_rate = Gauge(
    'algo_fill_rate_pct',
    'Execution algorithm fill rate percentage',
    ['algo_name'],
)

# ---------------------------------------------------------------------------
# Phase 5: Walk-Forward & Volatility
# ---------------------------------------------------------------------------

walk_forward_decay_pct = Gauge(
    'walk_forward_decay_pct',
    'Walk-forward IS->OOS performance decay percentage',
    ['strategy_name'],
)

walk_forward_is_valid = Gauge(
    'walk_forward_is_valid',
    'Walk-forward validation status (1=valid, 0=invalid)',
    ['strategy_name'],
)

vol_regime_indicator = Gauge(
    'vol_regime_indicator',
    'Volatility regime indicator (1=low, 2=normal, 3=high, 4=expanding, 5=contracting)',
    ['symbol'],
)

# ---------------------------------------------------------------------------
# Phase 5: Notifications & Dashboard
# ---------------------------------------------------------------------------

notifications_sent_total = Counter(
    'notifications_sent_total',
    'Notifications sent',
    ['channel', 'level'],
)

notifications_failed_total = Counter(
    'notifications_failed_total',
    'Notifications failed',
    ['channel'],
)

escalation_triggered_total = Counter(
    'escalation_triggered_total',
    'Escalations triggered',
    ['trigger_type'],
)

api_request_count = Counter(
    'api_request_count',
    'API requests',
    ['endpoint'],
)

# ---------------------------------------------------------------------------
# Phase 6: Meta-Strategy
# ---------------------------------------------------------------------------

meta_strategy_agreement_pct = Gauge(
    'meta_strategy_agreement_pct',
    'Meta-strategy agreement percentage',
    ['strategy_name'],
)

meta_strategy_confidence = Gauge(
    'meta_strategy_confidence',
    'Meta-strategy prediction confidence',
    ['strategy_name'],
)

meta_strategy_sub_signal_count = Gauge(
    'meta_strategy_sub_signal_count',
    'Number of sub-strategy signals available',
    ['strategy_name'],
)

# ---------------------------------------------------------------------------
# Phase 6: ML Regime Classification
# ---------------------------------------------------------------------------

ml_regime_confidence = Gauge(
    'ml_regime_confidence',
    'ML regime classifier confidence',
    ['regime'],
)

ml_regime_agreement = Gauge(
    'ml_regime_agreement',
    'Rule-ML regime classifier agreement (1=agree, 0=disagree)',
    [],
)

# ---------------------------------------------------------------------------
# Phase 6: Wallet Monitoring & Smart Money
# ---------------------------------------------------------------------------

wallet_monitor_active_count = Gauge(
    'wallet_monitor_active_count',
    'Number of actively monitored wallets',
)

wallet_signal_detected_total = Counter(
    'wallet_signal_detected_total',
    'Total wallet signals detected',
    ['signal_type'],
)

smart_money_agreement_count = Gauge(
    'smart_money_agreement_count',
    'Number of wallets agreeing on direction',
    ['direction'],
)
