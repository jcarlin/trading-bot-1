"""Tests for HyperliquidClient with mocked SDK dependencies."""

import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from core.models import Fill, Order, Position
from core.types import OrderType, Side


class TestHyperliquidClient(unittest.TestCase):
    """Test HyperliquidClient methods with mocked hyperliquid SDK."""

    def setUp(self):
        """Patch SDK imports and create a client instance."""
        # Mock the SDK modules
        self.mock_info_cls = MagicMock()
        self.mock_exchange_cls = MagicMock()
        self.mock_info = MagicMock()
        self.mock_exchange = MagicMock()
        self.mock_info_cls.return_value = self.mock_info
        self.mock_exchange_cls.return_value = self.mock_exchange

        # Mock eth_account
        self.mock_account = MagicMock()
        self.mock_account.address = "0xTestAddress"

        patches = {
            "hyperliquid.info.Info": self.mock_info_cls,
            "hyperliquid.exchange.Exchange": self.mock_exchange_cls,
        }

        self.patchers = []
        for target, mock_obj in patches.items():
            p = patch.dict("sys.modules", {
                "hyperliquid": MagicMock(),
                "hyperliquid.info": MagicMock(Info=self.mock_info_cls),
                "hyperliquid.exchange": MagicMock(Exchange=self.mock_exchange_cls),
            })
            p.start()
            self.patchers.append(p)

        # Patch eth_account in auth module
        self.eth_patcher = patch(
            "exchange.hyperliquid.auth._HAS_ETH_ACCOUNT", True
        )
        self.eth_patcher.start()
        self.eth_account_patcher = patch(
            "exchange.hyperliquid.auth._EthAccount"
        )
        mock_eth = self.eth_account_patcher.start()
        mock_eth.from_key.return_value = self.mock_account

        from exchange.hyperliquid.client import HyperliquidClient

        self.client = HyperliquidClient(
            account_address="0xTestAddress",
            private_key="0x" + "ab" * 32,
            testnet=True,
        )
        # Replace SDK instances with our mocks
        self.client._info = self.mock_info
        self.client._exchange = self.mock_exchange

    def tearDown(self):
        for p in self.patchers:
            p.stop()
        self.eth_patcher.stop()
        self.eth_account_patcher.stop()

    # ------------------------------------------------------------------
    # place_order
    # ------------------------------------------------------------------

    def test_place_order_market(self):
        """Market order uses IOC with aggressive price."""
        self.mock_info.all_mids.return_value = {"BTC": "50000.0"}
        self.mock_exchange.order.return_value = {
            "status": "ok",
            "response": {
                "type": "order",
                "data": {
                    "statuses": [
                        {
                            "filled": {
                                "totalSz": "0.5",
                                "avgPx": "50100.0",
                                "oid": 111,
                            }
                        }
                    ]
                },
            },
        }

        order = Order(
            symbol="BTC/USDC",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=0.5,
        )
        fill = self.client.place_order(order)

        self.assertIsInstance(fill, Fill)
        self.assertEqual(fill.order_id, "111")
        self.assertAlmostEqual(fill.quantity, 0.5)
        self.assertAlmostEqual(fill.fill_price, 50100.0)
        self.assertEqual(fill.side, Side.BUY)

        # Verify SDK was called with IOC
        call_args = self.mock_exchange.order.call_args
        self.assertEqual(call_args[0][0], "BTC")  # coin
        self.assertTrue(call_args[0][1])  # is_buy
        self.assertEqual(call_args[0][4], {"limit": {"tif": "Ioc"}})

    def test_place_order_limit(self):
        """Limit order uses GTC."""
        self.mock_exchange.order.return_value = {
            "status": "ok",
            "response": {
                "type": "order",
                "data": {
                    "statuses": [{"resting": {"oid": 222}}]
                },
            },
        }

        order = Order(
            symbol="ETH/USDC",
            side=Side.SELL,
            order_type=OrderType.LIMIT,
            quantity=2.0,
            price=3100.0,
        )
        fill = self.client.place_order(order)

        self.assertIsInstance(fill, Fill)
        self.assertEqual(fill.order_id, "222")
        self.assertAlmostEqual(fill.quantity, 0.0)  # resting, not filled

        call_args = self.mock_exchange.order.call_args
        self.assertEqual(call_args[0][0], "ETH")
        self.assertFalse(call_args[0][1])  # is_buy = False
        self.assertAlmostEqual(call_args[0][2], 2.0)  # size
        self.assertAlmostEqual(call_args[0][3], 3100.0)  # price
        self.assertEqual(call_args[0][4], {"limit": {"tif": "Gtc"}})

    # ------------------------------------------------------------------
    # cancel_order
    # ------------------------------------------------------------------

    def test_cancel_order(self):
        self.mock_exchange.cancel.return_value = {"status": "ok"}
        result = self.client.cancel_order("12345", "BTC/USDC")

        self.assertTrue(result)
        self.mock_exchange.cancel.assert_called_once_with("BTC", 12345)

    def test_cancel_order_failure(self):
        self.mock_exchange.cancel.side_effect = Exception("Not found")
        result = self.client.cancel_order("99999", "BTC/USDC")
        self.assertFalse(result)

    # ------------------------------------------------------------------
    # get_balance
    # ------------------------------------------------------------------

    def test_get_balance(self):
        self.mock_info.user_state.return_value = {
            "marginSummary": {
                "accountValue": "100000.50",
                "totalMarginUsed": "25000.0",
            },
            "assetPositions": [],
        }
        balance = self.client.get_balance()
        self.assertAlmostEqual(balance, 100000.50)
        self.mock_info.user_state.assert_called_once_with("0xTestAddress")

    # ------------------------------------------------------------------
    # get_ticker
    # ------------------------------------------------------------------

    def test_get_ticker(self):
        self.mock_info.all_mids.return_value = {"BTC": "50000.0"}
        self.mock_info.l2_snapshot.return_value = {
            "coin": "BTC",
            "levels": [
                [{"px": "49999.0", "sz": "1.0", "n": 1}],
                [{"px": "50001.0", "sz": "0.5", "n": 1}],
            ],
        }

        ticker = self.client.get_ticker("BTC/USDC")
        self.assertEqual(ticker["symbol"], "BTC/USDC")
        self.assertAlmostEqual(ticker["mid"], 50000.0)
        self.assertAlmostEqual(ticker["bid"], 49999.0)
        self.assertAlmostEqual(ticker["ask"], 50001.0)
        self.assertAlmostEqual(ticker["last"], 50000.0)

    # ------------------------------------------------------------------
    # get_positions
    # ------------------------------------------------------------------

    def test_get_positions(self):
        self.mock_info.user_state.return_value = {
            "assetPositions": [
                {
                    "type": "oneWay",
                    "position": {
                        "coin": "BTC",
                        "szi": "0.5",
                        "entryPx": "50000.0",
                        "unrealizedPnl": "250.0",
                    },
                },
                {
                    "type": "oneWay",
                    "position": {
                        "coin": "ETH",
                        "szi": "-3.0",
                        "entryPx": "3000.0",
                        "unrealizedPnl": "-100.0",
                    },
                },
            ],
            "marginSummary": {"accountValue": "100000"},
        }
        positions = self.client.get_positions()
        self.assertEqual(len(positions), 2)
        self.assertIsInstance(positions[0], Position)
        self.assertEqual(positions[0].side, Side.BUY)
        self.assertEqual(positions[1].side, Side.SELL)

    def test_get_positions_empty(self):
        self.mock_info.user_state.return_value = {
            "assetPositions": [],
            "marginSummary": {"accountValue": "100000"},
        }
        self.assertEqual(self.client.get_positions(), [])

    # ------------------------------------------------------------------
    # get_account_state
    # ------------------------------------------------------------------

    def test_get_account_state(self):
        self.mock_info.user_state.return_value = {
            "marginSummary": {
                "accountValue": "100000.0",
                "totalMarginUsed": "25000.0",
                "totalNtlPos": "50000.0",
                "totalRawUsd": "75000.0",
            },
            "withdrawable": "50000.0",
            "assetPositions": [{"position": {"coin": "BTC", "szi": "1"}}],
        }
        state = self.client.get_account_state()

        self.assertAlmostEqual(state["account_value"], 100000.0)
        self.assertAlmostEqual(state["total_margin_used"], 25000.0)
        self.assertAlmostEqual(state["total_ntl_pos"], 50000.0)
        self.assertAlmostEqual(state["total_raw_usd"], 75000.0)
        self.assertAlmostEqual(state["withdrawable"], 50000.0)
        self.assertEqual(state["positions"], 1)
        self.assertIn("timestamp", state)

    # ------------------------------------------------------------------
    # get_open_orders
    # ------------------------------------------------------------------

    def test_get_open_orders(self):
        self.mock_info.open_orders.return_value = [
            {"coin": "BTC", "oid": 1, "side": "B", "limitPx": "49000"},
            {"coin": "ETH", "oid": 2, "side": "A", "limitPx": "2900"},
            {"coin": "BTC", "oid": 3, "side": "A", "limitPx": "51000"},
        ]
        orders = self.client.get_open_orders("BTC/USDC")
        # Should only return BTC orders
        self.assertEqual(len(orders), 2)
        self.assertEqual(orders[0]["oid"], 1)
        self.assertEqual(orders[1]["oid"], 3)


if __name__ == "__main__":
    unittest.main()
