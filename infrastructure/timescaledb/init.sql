-- TimescaleDB initialization for trading bot
-- Creates all required tables, hypertables, retention policies, and indexes.

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============================================================
-- Candles
-- ============================================================
CREATE TABLE IF NOT EXISTS candles (
    time          TIMESTAMPTZ      NOT NULL,
    symbol        TEXT             NOT NULL,
    timeframe     TEXT             NOT NULL,
    open          DOUBLE PRECISION,
    high          DOUBLE PRECISION,
    low           DOUBLE PRECISION,
    close         DOUBLE PRECISION,
    volume        DOUBLE PRECISION,
    exchange_ts   TIMESTAMPTZ,
    receipt_ts    TIMESTAMPTZ,
    seq_num       BIGINT,
    UNIQUE (time, symbol, timeframe)
);

SELECT create_hypertable('candles', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_candles_symbol ON candles (symbol, timeframe, time DESC);

-- ============================================================
-- Trades (tape)
-- ============================================================
CREATE TABLE IF NOT EXISTS trades (
    time          TIMESTAMPTZ      NOT NULL,
    symbol        TEXT             NOT NULL,
    price         DOUBLE PRECISION,
    size          DOUBLE PRECISION,
    side          TEXT,
    trade_id      TEXT,
    exchange_ts   TIMESTAMPTZ,
    receipt_ts    TIMESTAMPTZ,
    seq_num       BIGINT
);

SELECT create_hypertable('trades', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades (symbol, time DESC);

-- 90-day retention policy
SELECT add_retention_policy('trades', INTERVAL '90 days', if_not_exists => TRUE);

-- ============================================================
-- Order book snapshots
-- ============================================================
CREATE TABLE IF NOT EXISTS orderbook_snapshots (
    time          TIMESTAMPTZ      NOT NULL,
    symbol        TEXT             NOT NULL,
    bids          JSONB,
    asks          JSONB,
    spread        DOUBLE PRECISION,
    mid_price     DOUBLE PRECISION
);

SELECT create_hypertable('orderbook_snapshots', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_ob_symbol ON orderbook_snapshots (symbol, time DESC);

-- 30-day retention policy
SELECT add_retention_policy('orderbook_snapshots', INTERVAL '30 days', if_not_exists => TRUE);

-- ============================================================
-- Funding rates
-- ============================================================
CREATE TABLE IF NOT EXISTS funding_rates (
    time           TIMESTAMPTZ      NOT NULL,
    symbol         TEXT             NOT NULL,
    rate           DOUBLE PRECISION,
    premium        DOUBLE PRECISION,
    mark_price     DOUBLE PRECISION,
    oracle_price   DOUBLE PRECISION,
    open_interest  DOUBLE PRECISION
);

SELECT create_hypertable('funding_rates', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_funding_symbol ON funding_rates (symbol, time DESC);

-- ============================================================
-- Orders (regular table — SERIAL id)
-- ============================================================
CREATE TABLE IF NOT EXISTS orders (
    id             SERIAL PRIMARY KEY,
    created_at     TIMESTAMPTZ      DEFAULT NOW(),
    order_id       TEXT             UNIQUE,
    symbol         TEXT,
    side           TEXT,
    type           TEXT,
    quantity       DOUBLE PRECISION,
    price          DOUBLE PRECISION,
    stop_loss      DOUBLE PRECISION,
    take_profit    DOUBLE PRECISION,
    status         TEXT             DEFAULT 'pending',
    strategy_name  TEXT
);

CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders (symbol);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders (status);

-- ============================================================
-- Fills
-- ============================================================
CREATE TABLE IF NOT EXISTS fills (
    time          TIMESTAMPTZ      NOT NULL,
    fill_id       TEXT,
    order_id      TEXT             REFERENCES orders(order_id),
    symbol        TEXT,
    side          TEXT,
    quantity      DOUBLE PRECISION,
    price         DOUBLE PRECISION,
    commission    DOUBLE PRECISION,
    closed_pnl    DOUBLE PRECISION
);

SELECT create_hypertable('fills', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_fills_order ON fills (order_id);
CREATE INDEX IF NOT EXISTS idx_fills_symbol ON fills (symbol, time DESC);

-- ============================================================
-- Decision log
-- ============================================================
CREATE TABLE IF NOT EXISTS decision_log (
    time           TIMESTAMPTZ      NOT NULL,
    decision_type  TEXT,
    strategy       TEXT,
    context        JSONB,
    hypothesis     TEXT,
    action         JSONB,
    alternatives   JSONB,
    confidence     DOUBLE PRECISION,
    outcome        JSONB
);

SELECT create_hypertable('decision_log', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_decision_strategy ON decision_log (strategy, time DESC);

-- ============================================================
-- System events
-- ============================================================
CREATE TABLE IF NOT EXISTS system_events (
    time          TIMESTAMPTZ      NOT NULL,
    event_type    TEXT,
    severity      TEXT,
    component     TEXT,
    message       TEXT,
    details       JSONB
);

SELECT create_hypertable('system_events', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_events_type ON system_events (event_type, time DESC);
CREATE INDEX IF NOT EXISTS idx_events_severity ON system_events (severity, time DESC);

-- ============================================================
-- Equity snapshots
-- ============================================================
CREATE TABLE IF NOT EXISTS equity_snapshots (
    time            TIMESTAMPTZ      NOT NULL,
    total_equity    DOUBLE PRECISION,
    cash            DOUBLE PRECISION,
    position_value  DOUBLE PRECISION,
    unrealized_pnl  DOUBLE PRECISION,
    realized_pnl    DOUBLE PRECISION,
    peak_equity     DOUBLE PRECISION,
    drawdown_pct    DOUBLE PRECISION,
    positions       JSONB
);

SELECT create_hypertable('equity_snapshots', 'time', if_not_exists => TRUE);

-- ============================================================
-- Continuous aggregate: 1-minute candles from trades
-- ============================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS candles_1m
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', time) AS bucket,
    symbol,
    first(price, time)            AS open,
    max(price)                    AS high,
    min(price)                    AS low,
    last(price, time)             AS close,
    sum(size)                     AS volume
FROM trades
GROUP BY bucket, symbol
WITH NO DATA;

SELECT add_continuous_aggregate_policy('candles_1m',
    start_offset  => INTERVAL '1 hour',
    end_offset    => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists => TRUE
);
