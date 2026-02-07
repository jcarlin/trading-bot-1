"""Tests for HyperliquidNormalizer."""

import unittest
from datetime import datetime, timezone

from core.models import Fill, Order, Position
from core.types import OrderType, Side
from exchange.hyperliquid.normalizer import HyperliquidNormalizer


class TestNormalizePositions(unittest.TestCase):
    """Test HyperliquidNormalizer.normalize_positions."""

    def test_normalize_positions_long(self):
        user_state = {
            "assetPositions": [
                {
                    "type": "oneWay",
                    "position": {
                        "coin": "BTC",
                        "szi": "0.5",
                        "entryPx": "50000.0",
                        "unrealizedPnl": "250.0",
                        "leverage": {"type": "cross", "value": 5},
                    },
                },
            ],
            "marginSummary": {"accountValue": "100000"},
        }
        positions = HyperliquidNormalizer.normalize_positions(user_state)

        self.assertEqual(len(positions), 1)
        pos = positions[0]
        self.assertIsInstance(pos, Position)
        self.assertEqual(pos.symbol, "BTC/USDC")
        self.assertEqual(pos.side, Side.BUY)
        self.assertAlmostEqual(pos.entry_price, 50000.0)
        self.assertAlmostEqual(pos.quantity, 0.5)
        self.assertAlmostEqual(pos.unrealized_pnl, 250.0)

    def test_normalize_positions_short(self):
        user_state = {
            "assetPositions": [
                {
                    "type": "oneWay",
                    "position": {
                        "coin": "ETH",
                        "szi": "-2.0",
                        "entryPx": "3000.0",
                        "unrealizedPnl": "-50.0",
                    },
                },
            ],
        }
        positions = HyperliquidNormalizer.normalize_positions(user_state)

        self.assertEqual(len(positions), 1)
        pos = positions[0]
        self.assertEqual(pos.side, Side.SELL)
        self.assertAlmostEqual(pos.quantity, 2.0)
        self.assertEqual(pos.symbol, "ETH/USDC")
        self.assertAlmostEqual(pos.unrealized_pnl, -50.0)

    def test_normalize_positions_empty(self):
        user_state = {"assetPositions": []}
        self.assertEqual(HyperliquidNormalizer.normalize_positions(user_state), [])

    def test_normalize_positions_missing_key(self):
        # No assetPositions key at all
        self.assertEqual(HyperliquidNormalizer.normalize_positions({}), [])

    def test_normalize_positions_zero_size_skipped(self):
        user_state = {
            "assetPositions": [
                {
                    "type": "oneWay",
                    "position": {
                        "coin": "BTC",
                        "szi": "0",
                        "entryPx": "50000.0",
                        "unrealizedPnl": "0",
                    },
                },
            ],
        }
        self.assertEqual(HyperliquidNormalizer.normalize_positions(user_state), [])

    def test_normalize_positions_multiple(self):
        user_state = {
            "assetPositions": [
                {
                    "type": "oneWay",
                    "position": {
                        "coin": "BTC",
                        "szi": "1.0",
                        "entryPx": "50000.0",
                        "unrealizedPnl": "100.0",
                    },
                },
                {
                    "type": "oneWay",
                    "position": {
                        "coin": "ETH",
                        "szi": "-5.0",
                        "entryPx": "3000.0",
                        "unrealizedPnl": "-20.0",
                    },
                },
            ],
        }
        positions = HyperliquidNormalizer.normalize_positions(user_state)
        self.assertEqual(len(positions), 2)
        self.assertEqual(positions[0].side, Side.BUY)
        self.assertEqual(positions[1].side, Side.SELL)


class TestNormalizeFill(unittest.TestCase):
    """Test HyperliquidNormalizer.normalize_fill."""

    def _make_order(self, **kwargs):
        defaults = {
            "symbol": "BTC/USDC",
            "side": Side.BUY,
            "order_type": OrderType.MARKET,
            "quantity": 0.5,
            "price": None,
            "order_id": "local-123",
        }
        defaults.update(kwargs)
        return Order(**defaults)

    def test_normalize_fill_filled(self):
        response = {
            "status": "ok",
            "response": {
                "type": "order",
                "data": {
                    "statuses": [
                        {
                            "filled": {
                                "totalSz": "0.5",
                                "avgPx": "50100.0",
                                "oid": 99999,
                            }
                        }
                    ]
                },
            },
        }
        order = self._make_order()
        fill = HyperliquidNormalizer.normalize_fill(response, order)

        self.assertIsInstance(fill, Fill)
        self.assertEqual(fill.order_id, "99999")
        self.assertEqual(fill.symbol, "BTC/USDC")
        self.assertEqual(fill.side, Side.BUY)
        self.assertAlmostEqual(fill.quantity, 0.5)
        self.assertAlmostEqual(fill.fill_price, 50100.0)

    def test_normalize_fill_resting(self):
        response = {
            "status": "ok",
            "response": {
                "type": "order",
                "data": {
                    "statuses": [{"resting": {"oid": 12345}}]
                },
            },
        }
        order = self._make_order(
            order_type=OrderType.LIMIT, price=49900.0
        )
        fill = HyperliquidNormalizer.normalize_fill(response, order)

        self.assertEqual(fill.order_id, "12345")
        self.assertAlmostEqual(fill.quantity, 0.0)  # not filled yet
        self.assertAlmostEqual(fill.fill_price, 49900.0)  # original limit price

    def test_normalize_fill_empty_statuses(self):
        response = {
            "status": "ok",
            "response": {"type": "order", "data": {"statuses": []}},
        }
        order = self._make_order()
        fill = HyperliquidNormalizer.normalize_fill(response, order)
        self.assertEqual(fill.order_id, "local-123")
        self.assertAlmostEqual(fill.quantity, 0.0)


class TestNormalizeTrade(unittest.TestCase):
    """Test HyperliquidNormalizer.normalize_trade."""

    def test_normalize_trade(self):
        ws_trade = {
            "coin": "BTC",
            "side": "Buy",
            "px": "50200.5",
            "sz": "0.3",
            "time": 1700000000000,
            "hash": "0xabc123",
        }
        result = HyperliquidNormalizer.normalize_trade(ws_trade)

        self.assertEqual(result["symbol"], "BTC/USDC")
        self.assertEqual(result["side"], "buy")
        self.assertAlmostEqual(result["price"], 50200.5)
        self.assertAlmostEqual(result["size"], 0.3)
        self.assertEqual(result["timestamp"], 1700000000000)
        self.assertEqual(result["trade_id"], "0xabc123")

    def test_normalize_trade_sell(self):
        ws_trade = {
            "coin": "ETH",
            "side": "Sell",
            "px": "3000",
            "sz": "1.5",
            "time": 1700000001000,
            "hash": "0xdef456",
        }
        result = HyperliquidNormalizer.normalize_trade(ws_trade)
        self.assertEqual(result["side"], "sell")
        self.assertEqual(result["symbol"], "ETH/USDC")


class TestNormalizeOrderbook(unittest.TestCase):
    """Test HyperliquidNormalizer.normalize_orderbook."""

    def test_normalize_orderbook(self):
        l2_data = {
            "coin": "BTC",
            "levels": [
                [
                    {"px": "50000", "sz": "1.5", "n": 3},
                    {"px": "49999", "sz": "2.0", "n": 5},
                ],
                [
                    {"px": "50001", "sz": "0.8", "n": 2},
                    {"px": "50002", "sz": "1.0", "n": 4},
                ],
            ],
        }
        result = HyperliquidNormalizer.normalize_orderbook(l2_data)

        self.assertEqual(result["symbol"], "BTC/USDC")
        self.assertEqual(len(result["bids"]), 2)
        self.assertEqual(len(result["asks"]), 2)
        self.assertAlmostEqual(result["bids"][0][0], 50000.0)
        self.assertAlmostEqual(result["bids"][0][1], 1.5)
        self.assertAlmostEqual(result["asks"][0][0], 50001.0)
        self.assertAlmostEqual(result["asks"][0][1], 0.8)

    def test_normalize_orderbook_empty(self):
        l2_data = {"coin": "ETH", "levels": [[], []]}
        result = HyperliquidNormalizer.normalize_orderbook(l2_data)
        self.assertEqual(result["bids"], [])
        self.assertEqual(result["asks"], [])
        self.assertEqual(result["symbol"], "ETH/USDC")


class TestNormalizeCandle(unittest.TestCase):
    """Test HyperliquidNormalizer.normalize_candle."""

    def test_normalize_candle(self):
        candle_data = {
            "t": 1700000000000,
            "T": 1700000060000,
            "s": "BTC",
            "i": "1m",
            "o": "50000.0",
            "c": "50100.0",
            "h": "50150.0",
            "l": "49950.0",
            "v": "123.456",
        }
        result = HyperliquidNormalizer.normalize_candle(candle_data)

        self.assertEqual(result["timestamp"], 1700000000000)
        self.assertEqual(result["close_timestamp"], 1700000060000)
        self.assertEqual(result["symbol"], "BTC/USDC")
        self.assertEqual(result["interval"], "1m")
        self.assertAlmostEqual(result["open"], 50000.0)
        self.assertAlmostEqual(result["high"], 50150.0)
        self.assertAlmostEqual(result["low"], 49950.0)
        self.assertAlmostEqual(result["close"], 50100.0)
        self.assertAlmostEqual(result["volume"], 123.456)


class TestNormalizeFunding(unittest.TestCase):
    """Test HyperliquidNormalizer.normalize_funding."""

    def test_normalize_funding(self):
        funding_data = {
            "funding": "0.0001",
            "openInterest": "1234.5",
            "prevDayPx": "50000",
            "dayNtlVlm": "123456789",
            "premium": "0.00005",
            "oraclePx": "50050",
            "markPx": "50045",
        }
        result = HyperliquidNormalizer.normalize_funding(funding_data)

        self.assertAlmostEqual(result["funding_rate"], 0.0001)
        self.assertAlmostEqual(result["open_interest"], 1234.5)
        self.assertAlmostEqual(result["premium"], 0.00005)
        self.assertAlmostEqual(result["oracle_price"], 50050.0)
        self.assertAlmostEqual(result["mark_price"], 50045.0)
        self.assertAlmostEqual(result["day_volume"], 123456789.0)
        self.assertAlmostEqual(result["prev_day_price"], 50000.0)


class TestSymbolConversion(unittest.TestCase):
    """Test coin_to_symbol and symbol_to_coin."""

    def test_coin_to_symbol(self):
        self.assertEqual(HyperliquidNormalizer.coin_to_symbol("BTC"), "BTC/USDC")
        self.assertEqual(HyperliquidNormalizer.coin_to_symbol("ETH"), "ETH/USDC")

    def test_coin_to_symbol_empty(self):
        self.assertEqual(HyperliquidNormalizer.coin_to_symbol(""), "")

    def test_symbol_to_coin(self):
        self.assertEqual(HyperliquidNormalizer.symbol_to_coin("BTC/USDC"), "BTC")
        self.assertEqual(HyperliquidNormalizer.symbol_to_coin("ETH/USDC"), "ETH")

    def test_symbol_to_coin_no_slash(self):
        self.assertEqual(HyperliquidNormalizer.symbol_to_coin("BTC"), "BTC")


if __name__ == "__main__":
    unittest.main()
