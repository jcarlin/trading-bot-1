"""TimescaleDB storage backend using psycopg2 connection pool."""

import json
import logging
from datetime import datetime
from typing import Optional

import psycopg2
import psycopg2.extras
from psycopg2 import pool

logger = logging.getLogger(__name__)


class TimescaleStore:
    """Persistent storage for time-series and relational trading data.

    Uses a threaded connection pool for safe concurrent access.
    """

    def __init__(self, config: dict):
        self._pool = pool.ThreadedConnectionPool(
            minconn=config.get("minconn", 2),
            maxconn=config.get("maxconn", 10),
            host=config.get("host", "localhost"),
            port=config.get("port", 5432),
            dbname=config.get("dbname", "trading_bot"),
            user=config.get("user", "trading"),
            password=config.get("password", ""),
        )
        logger.info("TimescaleStore connected to %s:%s/%s",
                     config.get("host"), config.get("port"), config.get("dbname"))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_conn(self):
        return self._pool.getconn()

    def _put_conn(self, conn):
        self._pool.putconn(conn)

    def _execute(self, sql: str, params: tuple = ()) -> None:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
            conn.commit()
        except psycopg2.Error:
            conn.rollback()
            logger.exception("DB execute error")
            raise
        finally:
            self._put_conn(conn)

    def _execute_returning(self, sql: str, params: tuple = ()):
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                result = cur.fetchone()
            conn.commit()
            return result
        except psycopg2.Error:
            conn.rollback()
            logger.exception("DB execute-returning error")
            raise
        finally:
            self._put_conn(conn)

    def _query(self, sql: str, params: tuple = ()) -> list[dict]:
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
            return [dict(r) for r in rows]
        except psycopg2.Error:
            logger.exception("DB query error")
            raise
        finally:
            self._put_conn(conn)

    def _query_one(self, sql: str, params: tuple = ()) -> Optional[dict]:
        rows = self._query(sql, params)
        return rows[0] if rows else None

    # ------------------------------------------------------------------
    # Candles
    # ------------------------------------------------------------------

    def insert_candle(self, candle: dict) -> None:
        sql = """
            INSERT INTO candles (time, symbol, timeframe, open, high, low, close,
                                 volume, exchange_ts, receipt_ts, seq_num)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (time, symbol, timeframe) DO UPDATE SET
                open = EXCLUDED.open, high = EXCLUDED.high,
                low = EXCLUDED.low, close = EXCLUDED.close,
                volume = EXCLUDED.volume
        """
        self._execute(sql, (
            candle["time"], candle["symbol"], candle["timeframe"],
            candle["open"], candle["high"], candle["low"], candle["close"],
            candle.get("volume"), candle.get("exchange_ts"),
            candle.get("receipt_ts"), candle.get("seq_num"),
        ))

    def query_candles(self, symbol: str, timeframe: str,
                      start: datetime, end: datetime) -> list[dict]:
        sql = """
            SELECT * FROM candles
            WHERE symbol = %s AND timeframe = %s AND time >= %s AND time <= %s
            ORDER BY time ASC
        """
        return self._query(sql, (symbol, timeframe, start, end))

    def get_latest_candle(self, symbol: str, timeframe: str) -> Optional[dict]:
        sql = """
            SELECT * FROM candles
            WHERE symbol = %s AND timeframe = %s
            ORDER BY time DESC LIMIT 1
        """
        return self._query_one(sql, (symbol, timeframe))

    # ------------------------------------------------------------------
    # Trades
    # ------------------------------------------------------------------

    def insert_trade(self, trade: dict) -> None:
        sql = """
            INSERT INTO trades (time, symbol, price, size, side, trade_id,
                                exchange_ts, receipt_ts, seq_num)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        self._execute(sql, (
            trade["time"], trade["symbol"], trade["price"], trade["size"],
            trade.get("side"), trade.get("trade_id"),
            trade.get("exchange_ts"), trade.get("receipt_ts"),
            trade.get("seq_num"),
        ))

    def insert_trades_batch(self, trades: list[dict]) -> None:
        sql = """
            INSERT INTO trades (time, symbol, price, size, side, trade_id,
                                exchange_ts, receipt_ts, seq_num)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                psycopg2.extras.execute_batch(cur, sql, [
                    (t["time"], t["symbol"], t["price"], t["size"],
                     t.get("side"), t.get("trade_id"),
                     t.get("exchange_ts"), t.get("receipt_ts"),
                     t.get("seq_num"))
                    for t in trades
                ])
            conn.commit()
        except psycopg2.Error:
            conn.rollback()
            logger.exception("DB batch insert error")
            raise
        finally:
            self._put_conn(conn)

    def query_trades(self, symbol: str, start: datetime,
                     end: datetime) -> list[dict]:
        sql = """
            SELECT * FROM trades
            WHERE symbol = %s AND time >= %s AND time <= %s
            ORDER BY time ASC
        """
        return self._query(sql, (symbol, start, end))

    # ------------------------------------------------------------------
    # Order book snapshots
    # ------------------------------------------------------------------

    def insert_orderbook_snapshot(self, snapshot: dict) -> None:
        sql = """
            INSERT INTO orderbook_snapshots (time, symbol, bids, asks, spread, mid_price)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        self._execute(sql, (
            snapshot["time"], snapshot["symbol"],
            json.dumps(snapshot.get("bids", [])),
            json.dumps(snapshot.get("asks", [])),
            snapshot.get("spread"), snapshot.get("mid_price"),
        ))

    # ------------------------------------------------------------------
    # Funding rates
    # ------------------------------------------------------------------

    def insert_funding_rate(self, rate: dict) -> None:
        sql = """
            INSERT INTO funding_rates (time, symbol, rate, premium, mark_price,
                                       oracle_price, open_interest)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        self._execute(sql, (
            rate["time"], rate["symbol"], rate["rate"],
            rate.get("premium"), rate.get("mark_price"),
            rate.get("oracle_price"), rate.get("open_interest"),
        ))

    def query_funding_rates(self, symbol: str, start: datetime, end: datetime) -> list[dict]:
        sql = """
            SELECT * FROM funding_rates
            WHERE symbol = %s AND time >= %s AND time <= %s
            ORDER BY time ASC
        """
        return self._query(sql, (symbol, start, end))

    def get_latest_funding_rate(self, symbol: str) -> Optional[dict]:
        sql = """
            SELECT * FROM funding_rates
            WHERE symbol = %s
            ORDER BY time DESC LIMIT 1
        """
        return self._query_one(sql, (symbol,))

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def insert_order(self, order: dict) -> int:
        sql = """
            INSERT INTO orders (order_id, symbol, side, type, quantity, price,
                                stop_loss, take_profit, status, strategy_name)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        row = self._execute_returning(sql, (
            order["order_id"], order["symbol"], order["side"], order["type"],
            order["quantity"], order.get("price"),
            order.get("stop_loss"), order.get("take_profit"),
            order.get("status", "pending"), order.get("strategy_name"),
        ))
        return row[0]

    def update_order_status(self, order_id: str, status: str) -> None:
        sql = "UPDATE orders SET status = %s WHERE order_id = %s"
        self._execute(sql, (status, order_id))

    def query_orders_by_strategy(self, strategy_name: str, start: datetime, end: datetime) -> list[dict]:
        sql = """
            SELECT * FROM orders
            WHERE strategy_name = %s
            ORDER BY order_id ASC
        """
        return self._query(sql, (strategy_name,))

    # ------------------------------------------------------------------
    # Fills
    # ------------------------------------------------------------------

    def insert_fill(self, fill: dict) -> None:
        sql = """
            INSERT INTO fills (time, fill_id, order_id, symbol, side,
                               quantity, price, commission, closed_pnl)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        self._execute(sql, (
            fill["time"], fill.get("fill_id"), fill.get("order_id"),
            fill["symbol"], fill["side"], fill["quantity"], fill["price"],
            fill.get("commission", 0.0), fill.get("closed_pnl", 0.0),
        ))

    def query_fills_by_strategy(self, strategy_name: str, start: datetime, end: datetime) -> list[dict]:
        sql = """
            SELECT f.* FROM fills f
            JOIN orders o ON f.order_id = o.order_id
            WHERE o.strategy_name = %s AND f.time >= %s AND f.time <= %s
            ORDER BY f.time ASC
        """
        return self._query(sql, (strategy_name, start, end))

    # ------------------------------------------------------------------
    # Decision log
    # ------------------------------------------------------------------

    def insert_decision(self, decision: dict) -> None:
        sql = """
            INSERT INTO decision_log (time, decision_type, strategy, context,
                                      hypothesis, action, alternatives,
                                      confidence, outcome)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        self._execute(sql, (
            decision["time"], decision.get("decision_type"),
            decision.get("strategy"),
            json.dumps(decision.get("context", {})),
            decision.get("hypothesis"),
            json.dumps(decision.get("action", {})),
            json.dumps(decision.get("alternatives", [])),
            decision.get("confidence"),
            json.dumps(decision.get("outcome", {})),
        ))

    # ------------------------------------------------------------------
    # System events
    # ------------------------------------------------------------------

    def insert_system_event(self, event: dict) -> None:
        sql = """
            INSERT INTO system_events (time, event_type, severity, component,
                                       message, details)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        self._execute(sql, (
            event["time"], event.get("event_type"), event.get("severity"),
            event.get("component"), event.get("message"),
            json.dumps(event.get("details", {})),
        ))

    # ------------------------------------------------------------------
    # Equity snapshots
    # ------------------------------------------------------------------

    def insert_equity_snapshot(self, snapshot: dict) -> None:
        sql = """
            INSERT INTO equity_snapshots (time, total_equity, cash, position_value,
                                          unrealized_pnl, realized_pnl,
                                          peak_equity, drawdown_pct, positions)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        self._execute(sql, (
            snapshot["time"], snapshot["total_equity"], snapshot["cash"],
            snapshot["position_value"],
            snapshot.get("unrealized_pnl", 0.0),
            snapshot.get("realized_pnl", 0.0),
            snapshot.get("peak_equity", 0.0),
            snapshot.get("drawdown_pct", 0.0),
            json.dumps(snapshot.get("positions", {})),
        ))

    def query_equity_snapshots(self, start: datetime,
                               end: datetime) -> list[dict]:
        sql = """
            SELECT * FROM equity_snapshots
            WHERE time >= %s AND time <= %s
            ORDER BY time ASC
        """
        return self._query(sql, (start, end))

    # ------------------------------------------------------------------
    # Market regime (stored in system_events)
    # ------------------------------------------------------------------

    def insert_market_regime(self, symbol: str, timeframe: str, regime: str,
                             confidence: float, indicators: dict,
                             timestamp: datetime) -> None:
        """Store a market regime classification as a system event."""
        self.insert_system_event({
            "time": timestamp,
            "event_type": "market_regime",
            "severity": "info",
            "component": "regime_classifier",
            "message": f"{symbol} {timeframe}: {regime} (conf={confidence:.2f})",
            "details": {
                "symbol": symbol,
                "timeframe": timeframe,
                "regime": regime,
                "confidence": confidence,
                "indicators": indicators,
            },
        })

    def query_market_regimes(self, symbol: str, start: datetime,
                             end: datetime) -> list[dict]:
        """Query market regime classifications from system events."""
        sql = """
            SELECT time, details FROM system_events
            WHERE event_type = 'market_regime'
              AND time >= %s AND time <= %s
            ORDER BY time ASC
        """
        rows = self._query(sql, (start, end))
        results = []
        for row in rows:
            details = row.get("details", {})
            if isinstance(details, str):
                import json
                details = json.loads(details)
            if details.get("symbol") == symbol:
                results.append({
                    "time": row["time"],
                    **details,
                })
        return results

    # ------------------------------------------------------------------
    # Health scores (stored in system_events)
    # ------------------------------------------------------------------

    def insert_health_score(self, strategy_name: str, score: float,
                            grade: str, components: dict,
                            timestamp: datetime) -> None:
        """Store a strategy health score as a system event."""
        self.insert_system_event({
            "time": timestamp,
            "event_type": "health_score",
            "severity": "info",
            "component": "health_scorer",
            "message": f"{strategy_name}: score={score:.1f} grade={grade}",
            "details": {
                "strategy_name": strategy_name,
                "score": score,
                "grade": grade,
                "components": components,
            },
        })

    def query_health_scores(self, strategy_name: str, start: datetime,
                            end: datetime) -> list[dict]:
        """Query strategy health scores from system events."""
        sql = """
            SELECT time, details FROM system_events
            WHERE event_type = 'health_score'
              AND time >= %s AND time <= %s
            ORDER BY time ASC
        """
        rows = self._query(sql, (start, end))
        results = []
        for row in rows:
            details = row.get("details", {})
            if isinstance(details, str):
                import json
                details = json.loads(details)
            if details.get("strategy_name") == strategy_name:
                results.append({
                    "time": row["time"],
                    **details,
                })
        return results

    # ------------------------------------------------------------------
    # Decision queries
    # ------------------------------------------------------------------

    def query_decisions(self, strategy_name: str, start: datetime,
                        end: datetime) -> list[dict]:
        """Query decision log entries for a strategy."""
        sql = """
            SELECT * FROM decision_log
            WHERE strategy = %s AND time >= %s AND time <= %s
            ORDER BY time ASC
        """
        return self._query(sql, (strategy_name, start, end))


    # ------------------------------------------------------------------
    # Correlation snapshots (stored in system_events)
    # ------------------------------------------------------------------

    def insert_correlation_snapshot(self, strategies: list[str],
                                    matrix: list[list[float]],
                                    timestamp: datetime) -> None:
        """Store a correlation matrix snapshot as a system event."""
        self.insert_system_event({
            "time": timestamp,
            "event_type": "correlation_snapshot",
            "severity": "info",
            "component": "correlation_analyzer",
            "message": f"Correlation snapshot for {len(strategies)} strategies",
            "details": {
                "strategies": strategies,
                "matrix": matrix,
            },
        })

    def query_correlation_snapshots(self, start: datetime,
                                     end: datetime) -> list[dict]:
        """Query correlation snapshots from system events."""
        sql = """
            SELECT time, details FROM system_events
            WHERE event_type = 'correlation_snapshot'
              AND time >= %s AND time <= %s
            ORDER BY time ASC
        """
        rows = self._query(sql, (start, end))
        results = []
        for row in rows:
            details = row.get("details", {})
            if isinstance(details, str):
                import json
                details = json.loads(details)
            results.append({
                "time": row["time"],
                **details,
            })
        return results

    # ------------------------------------------------------------------
    # Wallet intelligence
    # ------------------------------------------------------------------

    def insert_wallet_score(self, address: str, score: float,
                            components: dict, timestamp: datetime) -> None:
        """Store a wallet score as a system event."""
        self.insert_system_event({
            "time": timestamp,
            "event_type": "wallet_score",
            "severity": "info",
            "component": "wallet_scorer",
            "message": f"Wallet {address[:10]}...: score={score:.1f}",
            "details": {
                "address": address,
                "score": score,
                "components": components,
            },
        })

    def query_wallet_scores(self, min_score: float = 0.0,
                            limit: int = 50) -> list[dict]:
        """Query wallet scores from system events."""
        sql = """
            SELECT time, details FROM system_events
            WHERE event_type = 'wallet_score'
            ORDER BY time DESC
            LIMIT %s
        """
        rows = self._query(sql, (limit,))
        results = []
        for row in rows:
            details = row.get("details", {})
            if isinstance(details, str):
                import json
                details = json.loads(details)
            if details.get("score", 0) >= min_score:
                results.append({
                    "time": row["time"],
                    **details,
                })
        return results

    def insert_wallet_analysis(self, address: str, analysis: dict,
                                timestamp: datetime) -> None:
        """Store a wallet analysis as a system event."""
        self.insert_system_event({
            "time": timestamp,
            "event_type": "wallet_analysis",
            "severity": "info",
            "component": "pattern_analyzer",
            "message": f"Analysis for {address[:10]}...",
            "details": {
                "address": address,
                "analysis": analysis,
            },
        })

    def query_wallet_analyses(self, address: str, start: datetime,
                               end: datetime) -> list[dict]:
        """Query wallet analyses from system events."""
        sql = """
            SELECT time, details FROM system_events
            WHERE event_type = 'wallet_analysis'
              AND time >= %s AND time <= %s
            ORDER BY time ASC
        """
        rows = self._query(sql, (start, end))
        results = []
        for row in rows:
            details = row.get("details", {})
            if isinstance(details, str):
                import json
                details = json.loads(details)
            if details.get("address") == address:
                results.append({
                    "time": row["time"],
                    **details,
                })
        return results

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._pool:
            self._pool.closeall()
            logger.info("TimescaleStore connection pool closed")
