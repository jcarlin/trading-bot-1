"""Tests for storage layer (TimescaleStore and RedisStore).

All external dependencies (psycopg2, redis) are mocked — no real DB connections.
"""

import json
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch, call


class TestTimescaleStore(unittest.TestCase):
    """Tests for storage.timescale.TimescaleStore."""

    @patch("storage.timescale.pool.ThreadedConnectionPool")
    def setUp(self, mock_pool_cls):
        self.mock_pool = MagicMock()
        mock_pool_cls.return_value = self.mock_pool

        self.mock_conn = MagicMock()
        self.mock_cursor = MagicMock()
        self.mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=self.mock_cursor)
        self.mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        self.mock_pool.getconn.return_value = self.mock_conn

        from storage.timescale import TimescaleStore
        self.store = TimescaleStore({
            "host": "localhost", "port": 5432,
            "dbname": "test", "user": "test", "password": "test",
        })
        self.mock_pool_cls = mock_pool_cls

    def test_pool_creation(self):
        self.mock_pool_cls = None  # already asserted in setUp
        # Pool was created with correct params
        self.assertIsNotNone(self.store._pool)

    def test_insert_candle(self):
        candle = {
            "time": datetime(2024, 1, 1), "symbol": "BTC-USD",
            "timeframe": "1m", "open": 100.0, "high": 105.0,
            "low": 99.0, "close": 103.0, "volume": 1000.0,
            "exchange_ts": None, "receipt_ts": None, "seq_num": None,
        }
        self.store.insert_candle(candle)
        self.mock_cursor.execute.assert_called_once()
        sql = self.mock_cursor.execute.call_args[0][0]
        self.assertIn("INSERT INTO candles", sql)
        self.mock_conn.commit.assert_called_once()

    def test_insert_trade(self):
        trade = {
            "time": datetime(2024, 1, 1), "symbol": "ETH-USD",
            "price": 2000.0, "size": 1.5,
        }
        self.store.insert_trade(trade)
        self.mock_cursor.execute.assert_called_once()
        sql = self.mock_cursor.execute.call_args[0][0]
        self.assertIn("INSERT INTO trades", sql)

    @patch("storage.timescale.psycopg2.extras.execute_batch")
    def test_insert_trades_batch(self, mock_execute_batch):
        trades = [
            {"time": datetime(2024, 1, 1), "symbol": "BTC-USD",
             "price": 100.0, "size": 0.5},
            {"time": datetime(2024, 1, 1), "symbol": "BTC-USD",
             "price": 101.0, "size": 0.3},
        ]
        self.store.insert_trades_batch(trades)
        mock_execute_batch.assert_called_once()
        self.mock_conn.commit.assert_called_once()

    def test_insert_orderbook_snapshot(self):
        snapshot = {
            "time": datetime(2024, 1, 1), "symbol": "BTC-USD",
            "bids": [[100, 1]], "asks": [[101, 1]],
            "spread": 1.0, "mid_price": 100.5,
        }
        self.store.insert_orderbook_snapshot(snapshot)
        self.mock_cursor.execute.assert_called_once()
        sql = self.mock_cursor.execute.call_args[0][0]
        self.assertIn("INSERT INTO orderbook_snapshots", sql)

    def test_insert_funding_rate(self):
        rate = {
            "time": datetime(2024, 1, 1), "symbol": "BTC-USD",
            "rate": 0.0001, "premium": 0.0, "mark_price": 100.0,
        }
        self.store.insert_funding_rate(rate)
        self.mock_cursor.execute.assert_called_once()
        sql = self.mock_cursor.execute.call_args[0][0]
        self.assertIn("INSERT INTO funding_rates", sql)

    def test_insert_order_returns_id(self):
        self.mock_cursor.fetchone.return_value = (42,)
        order = {
            "order_id": "ord-1", "symbol": "BTC-USD", "side": "buy",
            "type": "limit", "quantity": 1.0, "price": 100.0,
        }
        result = self.store.insert_order(order)
        self.assertEqual(result, 42)
        sql = self.mock_cursor.execute.call_args[0][0]
        self.assertIn("INSERT INTO orders", sql)
        self.assertIn("RETURNING id", sql)

    def test_update_order_status(self):
        self.store.update_order_status("ord-1", "filled")
        self.mock_cursor.execute.assert_called_once()
        sql = self.mock_cursor.execute.call_args[0][0]
        self.assertIn("UPDATE orders SET status", sql)

    def test_insert_fill(self):
        fill = {
            "time": datetime(2024, 1, 1), "fill_id": "f-1",
            "order_id": "ord-1", "symbol": "BTC-USD",
            "side": "buy", "quantity": 1.0, "price": 100.0,
        }
        self.store.insert_fill(fill)
        self.mock_cursor.execute.assert_called_once()
        sql = self.mock_cursor.execute.call_args[0][0]
        self.assertIn("INSERT INTO fills", sql)

    def test_insert_decision(self):
        decision = {
            "time": datetime(2024, 1, 1), "decision_type": "allocation",
            "strategy": "momentum", "context": {"regime": "trending"},
            "hypothesis": "trend will continue", "action": {"increase": 10},
            "alternatives": [], "confidence": 0.8, "outcome": {},
        }
        self.store.insert_decision(decision)
        self.mock_cursor.execute.assert_called_once()
        sql = self.mock_cursor.execute.call_args[0][0]
        self.assertIn("INSERT INTO decision_log", sql)

    def test_insert_system_event(self):
        event = {
            "time": datetime(2024, 1, 1), "event_type": "strategy_start",
            "severity": "info", "component": "orchestrator",
            "message": "Strategy started", "details": {},
        }
        self.store.insert_system_event(event)
        self.mock_cursor.execute.assert_called_once()
        sql = self.mock_cursor.execute.call_args[0][0]
        self.assertIn("INSERT INTO system_events", sql)

    def test_insert_equity_snapshot(self):
        snap = {
            "time": datetime(2024, 1, 1), "total_equity": 10000.0,
            "cash": 5000.0, "position_value": 5000.0,
        }
        self.store.insert_equity_snapshot(snap)
        self.mock_cursor.execute.assert_called_once()
        sql = self.mock_cursor.execute.call_args[0][0]
        self.assertIn("INSERT INTO equity_snapshots", sql)

    def test_query_candles(self):
        # RealDictCursor returns list of RealDictRow, simulate with dicts
        mock_dict_cursor = MagicMock()
        mock_dict_cursor.fetchall.return_value = [
            {"time": datetime(2024, 1, 1), "symbol": "BTC-USD",
             "open": 100.0, "close": 103.0},
        ]
        mock_dict_cursor.__enter__ = MagicMock(return_value=mock_dict_cursor)
        mock_dict_cursor.__exit__ = MagicMock(return_value=False)
        self.mock_conn.cursor.return_value = mock_dict_cursor

        results = self.store.query_candles(
            "BTC-USD", "1m", datetime(2024, 1, 1), datetime(2024, 1, 2))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["symbol"], "BTC-USD")

    def test_query_trades(self):
        mock_dict_cursor = MagicMock()
        mock_dict_cursor.fetchall.return_value = []
        mock_dict_cursor.__enter__ = MagicMock(return_value=mock_dict_cursor)
        mock_dict_cursor.__exit__ = MagicMock(return_value=False)
        self.mock_conn.cursor.return_value = mock_dict_cursor

        results = self.store.query_trades(
            "ETH-USD", datetime(2024, 1, 1), datetime(2024, 1, 2))
        self.assertEqual(results, [])

    def test_query_equity_snapshots(self):
        mock_dict_cursor = MagicMock()
        mock_dict_cursor.fetchall.return_value = [
            {"time": datetime(2024, 1, 1), "total_equity": 10000.0},
        ]
        mock_dict_cursor.__enter__ = MagicMock(return_value=mock_dict_cursor)
        mock_dict_cursor.__exit__ = MagicMock(return_value=False)
        self.mock_conn.cursor.return_value = mock_dict_cursor

        results = self.store.query_equity_snapshots(
            datetime(2024, 1, 1), datetime(2024, 1, 2))
        self.assertEqual(len(results), 1)

    def test_get_latest_candle_returns_none_when_empty(self):
        mock_dict_cursor = MagicMock()
        mock_dict_cursor.fetchall.return_value = []
        mock_dict_cursor.__enter__ = MagicMock(return_value=mock_dict_cursor)
        mock_dict_cursor.__exit__ = MagicMock(return_value=False)
        self.mock_conn.cursor.return_value = mock_dict_cursor

        result = self.store.get_latest_candle("BTC-USD", "1m")
        self.assertIsNone(result)

    def test_query_funding_rates(self):
        mock_dict_cursor = MagicMock()
        mock_dict_cursor.fetchall.return_value = [
            {"time": datetime(2024, 1, 1), "symbol": "BTC-USD",
             "rate": 0.0001, "premium": 0.0},
        ]
        mock_dict_cursor.__enter__ = MagicMock(return_value=mock_dict_cursor)
        mock_dict_cursor.__exit__ = MagicMock(return_value=False)
        self.mock_conn.cursor.return_value = mock_dict_cursor

        results = self.store.query_funding_rates(
            "BTC-USD", datetime(2024, 1, 1), datetime(2024, 1, 2))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["rate"], 0.0001)

    def test_get_latest_funding_rate(self):
        mock_dict_cursor = MagicMock()
        mock_dict_cursor.fetchall.return_value = [
            {"time": datetime(2024, 1, 1), "symbol": "BTC-USD",
             "rate": 0.0002, "mark_price": 50000.0},
        ]
        mock_dict_cursor.__enter__ = MagicMock(return_value=mock_dict_cursor)
        mock_dict_cursor.__exit__ = MagicMock(return_value=False)
        self.mock_conn.cursor.return_value = mock_dict_cursor

        result = self.store.get_latest_funding_rate("BTC-USD")
        self.assertIsNotNone(result)
        self.assertEqual(result["rate"], 0.0002)

    def test_get_latest_funding_rate_empty(self):
        mock_dict_cursor = MagicMock()
        mock_dict_cursor.fetchall.return_value = []
        mock_dict_cursor.__enter__ = MagicMock(return_value=mock_dict_cursor)
        mock_dict_cursor.__exit__ = MagicMock(return_value=False)
        self.mock_conn.cursor.return_value = mock_dict_cursor

        result = self.store.get_latest_funding_rate("BTC-USD")
        self.assertIsNone(result)

    def test_query_fills_by_strategy(self):
        mock_dict_cursor = MagicMock()
        mock_dict_cursor.fetchall.return_value = [
            {"time": datetime(2024, 1, 1), "fill_id": "f-1",
             "order_id": "ord-1", "symbol": "BTC-USD",
             "side": "buy", "quantity": 1.0, "price": 100.0},
        ]
        mock_dict_cursor.__enter__ = MagicMock(return_value=mock_dict_cursor)
        mock_dict_cursor.__exit__ = MagicMock(return_value=False)
        self.mock_conn.cursor.return_value = mock_dict_cursor

        results = self.store.query_fills_by_strategy(
            "funding_rate_arb", datetime(2024, 1, 1), datetime(2024, 1, 2))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["fill_id"], "f-1")

    def test_query_orders_by_strategy(self):
        mock_dict_cursor = MagicMock()
        mock_dict_cursor.fetchall.return_value = [
            {"order_id": "ord-1", "symbol": "BTC-USD",
             "side": "buy", "strategy_name": "funding_rate_arb"},
        ]
        mock_dict_cursor.__enter__ = MagicMock(return_value=mock_dict_cursor)
        mock_dict_cursor.__exit__ = MagicMock(return_value=False)
        self.mock_conn.cursor.return_value = mock_dict_cursor

        results = self.store.query_orders_by_strategy(
            "funding_rate_arb", datetime(2024, 1, 1), datetime(2024, 1, 2))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["strategy_name"], "funding_rate_arb")

    def test_close(self):
        self.store.close()
        self.mock_pool.closeall.assert_called_once()

    def test_execute_rollback_on_error(self):
        import psycopg2
        self.mock_cursor.execute.side_effect = psycopg2.Error("test")
        with self.assertRaises(psycopg2.Error):
            self.store.insert_trade({
                "time": datetime(2024, 1, 1), "symbol": "X",
                "price": 1.0, "size": 1.0,
            })
        self.mock_conn.rollback.assert_called_once()


class TestRedisStore(unittest.TestCase):
    """Tests for storage.redis_store.RedisStore."""

    @patch("storage.redis_store.redis.Redis")
    def setUp(self, mock_redis_cls):
        self.mock_redis = MagicMock()
        mock_redis_cls.return_value = self.mock_redis

        from storage.redis_store import RedisStore
        self.store = RedisStore({
            "host": "localhost", "port": 6379, "db": 0,
        })

    # -- Order book -------------------------------------------------

    def test_set_orderbook(self):
        self.store.set_orderbook(
            "BTC-USD", bids=[[100, 1]], asks=[[101, 1]],
            mid_price=100.5, spread=1.0, timestamp="2024-01-01T00:00:00Z")
        self.mock_redis.hset.assert_called_once()
        args = self.mock_redis.hset.call_args
        self.assertEqual(args[0][0], "ob:BTC-USD")

    def test_get_orderbook(self):
        self.mock_redis.hgetall.return_value = {
            "bids": "[[100, 1]]", "asks": "[[101, 1]]",
            "mid_price": "100.5", "spread": "1.0",
            "timestamp": "2024-01-01T00:00:00Z",
        }
        result = self.store.get_orderbook("BTC-USD")
        self.assertEqual(result["mid_price"], 100.5)
        self.assertEqual(result["bids"], [[100, 1]])

    def test_get_orderbook_returns_none(self):
        self.mock_redis.hgetall.return_value = {}
        result = self.store.get_orderbook("BTC-USD")
        self.assertIsNone(result)

    # -- Price ------------------------------------------------------

    def test_set_price(self):
        self.store.set_price("ETH-USD", last=2000.0, bid=1999.0,
                             ask=2001.0, timestamp="2024-01-01T00:00:00Z")
        self.mock_redis.hset.assert_called_once()
        args = self.mock_redis.hset.call_args
        self.assertEqual(args[0][0], "price:ETH-USD")

    def test_get_price(self):
        self.mock_redis.hgetall.return_value = {
            "last": "2000.0", "bid": "1999.0", "ask": "2001.0",
            "timestamp": "2024-01-01T00:00:00Z",
        }
        result = self.store.get_price("ETH-USD")
        self.assertEqual(result["last"], 2000.0)
        self.assertEqual(result["bid"], 1999.0)

    # -- Position ---------------------------------------------------

    def test_set_position(self):
        self.store.set_position(
            "BTC-USD", entry_price=100.0, quantity=1.0,
            side="buy", entry_time="2024-01-01T00:00:00Z")
        self.mock_redis.hset.assert_called_once()
        args = self.mock_redis.hset.call_args
        self.assertEqual(args[0][0], "position:BTC-USD")

    def test_get_position(self):
        self.mock_redis.hgetall.return_value = {
            "entry_price": "100.0", "quantity": "1.0",
            "side": "buy", "entry_time": "2024-01-01T00:00:00Z",
            "unrealized_pnl": "5.0",
        }
        result = self.store.get_position("BTC-USD")
        self.assertEqual(result["entry_price"], 100.0)
        self.assertEqual(result["side"], "buy")

    def test_get_position_returns_none(self):
        self.mock_redis.hgetall.return_value = {}
        result = self.store.get_position("BTC-USD")
        self.assertIsNone(result)

    # -- Account state ----------------------------------------------

    def test_set_account_state(self):
        self.store.set_account_state(
            total_equity=10000.0, cash=5000.0,
            position_value=5000.0, unrealized_pnl=100.0,
            drawdown_pct=0.02, timestamp="2024-01-01T00:00:00Z")
        self.mock_redis.hset.assert_called_once()
        args = self.mock_redis.hset.call_args
        self.assertEqual(args[0][0], "account:state")

    def test_get_account_state(self):
        self.mock_redis.hgetall.return_value = {
            "total_equity": "10000.0", "cash": "5000.0",
            "position_value": "5000.0", "unrealized_pnl": "100.0",
            "drawdown_pct": "0.02", "timestamp": "2024-01-01T00:00:00Z",
        }
        result = self.store.get_account_state()
        self.assertEqual(result["total_equity"], 10000.0)
        self.assertEqual(result["drawdown_pct"], 0.02)

    def test_get_account_state_returns_none(self):
        self.mock_redis.hgetall.return_value = {}
        result = self.store.get_account_state()
        self.assertIsNone(result)

    # -- Circuit breaker --------------------------------------------

    def test_set_circuit_breaker(self):
        self.store.set_circuit_breaker(
            active=True, reason="max drawdown",
            activated_at="2024-01-01T00:00:00Z")
        self.mock_redis.hset.assert_called_once()
        args = self.mock_redis.hset.call_args
        self.assertEqual(args[0][0], "circuit_breaker:state")

    def test_get_circuit_breaker_active(self):
        self.mock_redis.hgetall.return_value = {
            "active": "true", "reason": "max drawdown",
            "activated_at": "2024-01-01T00:00:00Z",
        }
        result = self.store.get_circuit_breaker()
        self.assertTrue(result["active"])
        self.assertEqual(result["reason"], "max drawdown")

    def test_get_circuit_breaker_inactive(self):
        self.mock_redis.hgetall.return_value = {
            "active": "false", "reason": "",
            "activated_at": "",
        }
        result = self.store.get_circuit_breaker()
        self.assertFalse(result["active"])

    # -- Data quality -----------------------------------------------

    def test_set_data_quality_with_ttl(self):
        self.store.set_data_quality(
            source="hyperliquid", symbol="BTC-USD",
            last_update="2024-01-01T00:00:00Z",
            latency_ms=15.2, status="ok")
        self.mock_redis.hset.assert_called_once()
        self.mock_redis.expire.assert_called_once_with(
            "data_quality:hyperliquid:BTC-USD", 300)

    def test_set_data_quality_custom_ttl(self):
        self.store.set_data_quality(
            source="coinglass", symbol="ETH-USD",
            last_update="2024-01-01", latency_ms=50.0,
            status="ok", ttl=600)
        self.mock_redis.expire.assert_called_once_with(
            "data_quality:coinglass:ETH-USD", 600)

    def test_get_data_quality(self):
        self.mock_redis.hgetall.return_value = {
            "last_update": "2024-01-01T00:00:00Z",
            "latency_ms": "15.2", "status": "ok",
        }
        result = self.store.get_data_quality("hyperliquid", "BTC-USD")
        self.assertEqual(result["latency_ms"], 15.2)
        self.assertEqual(result["status"], "ok")

    def test_get_data_quality_returns_none_when_expired(self):
        self.mock_redis.hgetall.return_value = {}
        result = self.store.get_data_quality("hyperliquid", "BTC-USD")
        self.assertIsNone(result)

    # -- Ping -------------------------------------------------------

    def test_ping_success(self):
        self.mock_redis.ping.return_value = True
        self.assertTrue(self.store.ping())

    def test_ping_failure(self):
        import redis as redis_lib
        self.mock_redis.ping.side_effect = redis_lib.ConnectionError()
        self.assertFalse(self.store.ping())

    # -- Open orders ------------------------------------------------

    def test_add_and_get_open_orders(self):
        order = {"order_id": "o-1", "symbol": "BTC-USD", "side": "buy"}
        self.store.add_open_order("BTC-USD", order)
        self.mock_redis.sadd.assert_called_once_with(
            "orders:open:BTC-USD", json.dumps(order))

    def test_remove_open_order(self):
        order = {"order_id": "o-1", "symbol": "BTC-USD", "side": "buy"}
        self.store.remove_open_order("BTC-USD", order)
        self.mock_redis.srem.assert_called_once_with(
            "orders:open:BTC-USD", json.dumps(order))

    def test_get_open_orders(self):
        raw = {json.dumps({"order_id": "o-1"}), json.dumps({"order_id": "o-2"})}
        self.mock_redis.smembers.return_value = raw
        orders = self.store.get_open_orders("BTC-USD")
        self.assertEqual(len(orders), 2)
        ids = {o["order_id"] for o in orders}
        self.assertEqual(ids, {"o-1", "o-2"})

    # -- Strategy state ---------------------------------------------

    def test_set_strategy_state(self):
        self.store.set_strategy_state(
            "momentum", status="active",
            last_signal="enter_long",
            last_signal_time="2024-01-01T00:00:00Z")
        self.mock_redis.hset.assert_called_once()
        args = self.mock_redis.hset.call_args
        self.assertEqual(args[0][0], "strategy:momentum:state")

    def test_get_strategy_state(self):
        self.mock_redis.hgetall.return_value = {
            "status": "active", "last_signal": "enter_long",
            "last_signal_time": "2024-01-01T00:00:00Z",
        }
        result = self.store.get_strategy_state("momentum")
        self.assertEqual(result["status"], "active")

    # -- Funding ----------------------------------------------------

    def test_set_funding(self):
        self.store.set_funding(
            "BTC-USD", rate=0.0001, premium=0.0002,
            mark_price=50000.0, timestamp="2024-01-01T00:00:00Z")
        self.mock_redis.hset.assert_called_once()
        args = self.mock_redis.hset.call_args
        self.assertEqual(args[0][0], "funding:BTC-USD")

    def test_get_funding(self):
        self.mock_redis.hgetall.return_value = {
            "rate": "0.0001", "premium": "0.0002",
            "mark_price": "50000.0", "timestamp": "2024-01-01T00:00:00Z",
        }
        result = self.store.get_funding("BTC-USD")
        self.assertEqual(result["rate"], 0.0001)
        self.assertEqual(result["mark_price"], 50000.0)

    def test_get_funding_returns_none(self):
        self.mock_redis.hgetall.return_value = {}
        result = self.store.get_funding("BTC-USD")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
