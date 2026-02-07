"""Normalise Hyperliquid API / WebSocket responses to internal models."""

import logging
from datetime import datetime, timezone
from typing import Optional

from core.models import Fill, Order, Position
from core.types import Side

logger = logging.getLogger(__name__)


class HyperliquidNormalizer:
    """Static helpers that convert Hyperliquid data structures into the
    canonical models defined in ``core.models``.
    """

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_positions(user_state: dict) -> list[Position]:
        """Convert Hyperliquid ``user_state`` to a list of Position objects.

        HL ``assetPositions`` entries look like::

            {
                "type": "oneWay",
                "position": {
                    "coin": "BTC",
                    "szi": "0.5",        # signed size (negative = short)
                    "entryPx": "50000.0",
                    "unrealizedPnl": "250.0",
                    ...
                }
            }
        """
        positions: list[Position] = []
        asset_positions = user_state.get("assetPositions", [])

        for ap in asset_positions:
            pos = ap.get("position", {})
            szi = float(pos.get("szi", 0))
            if szi == 0:
                continue

            side = Side.BUY if szi > 0 else Side.SELL
            quantity = abs(szi)
            entry_px = float(pos.get("entryPx", 0))
            unrealized_pnl = float(pos.get("unrealizedPnl", 0))
            coin = pos.get("coin", "")
            symbol = HyperliquidNormalizer.coin_to_symbol(coin)

            positions.append(
                Position(
                    symbol=symbol,
                    side=side,
                    entry_price=entry_px,
                    quantity=quantity,
                    entry_time=datetime.now(timezone.utc),
                    unrealized_pnl=unrealized_pnl,
                )
            )

        return positions

    # ------------------------------------------------------------------
    # Fills
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_fill(order_response: dict, original_order: Order) -> Fill:
        """Convert an HL order response into a ``Fill``.

        HL responses have this shape::

            {
                "status": "ok",
                "response": {
                    "type": "order",
                    "data": {
                        "statuses": [
                            {"filled": {"totalSz": "0.5", "avgPx": "50100.0"}}
                            # OR
                            {"resting": {"oid": 123456}}
                        ]
                    }
                }
            }
        """
        response = order_response.get("response", {})
        data = response.get("data", {})
        statuses = data.get("statuses", [])

        order_id = original_order.order_id or ""
        fill_price = 0.0
        filled_qty = 0.0

        if statuses:
            status_entry = statuses[0]

            if "filled" in status_entry:
                filled = status_entry["filled"]
                filled_qty = float(filled.get("totalSz", 0))
                fill_price = float(filled.get("avgPx", 0))
                order_id = str(filled.get("oid", order_id))
            elif "resting" in status_entry:
                resting = status_entry["resting"]
                order_id = str(resting.get("oid", order_id))
                # For resting orders, the fill hasn't happened yet.
                fill_price = original_order.price or 0.0
                filled_qty = 0.0

        return Fill(
            order_id=order_id,
            symbol=original_order.symbol,
            side=original_order.side,
            quantity=filled_qty,
            fill_price=fill_price,
            timestamp=datetime.now(timezone.utc),
        )

    # ------------------------------------------------------------------
    # WebSocket trade
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_trade(ws_trade: dict) -> dict:
        """Normalise a Hyperliquid WebSocket trade message.

        Input keys: ``coin``, ``side``, ``px``, ``sz``, ``time``, ``hash``.
        """
        return {
            "symbol": HyperliquidNormalizer.coin_to_symbol(ws_trade.get("coin", "")),
            "side": ws_trade.get("side", "").lower(),
            "price": float(ws_trade.get("px", 0)),
            "size": float(ws_trade.get("sz", 0)),
            "timestamp": ws_trade.get("time", 0),
            "trade_id": ws_trade.get("hash", ""),
        }

    # ------------------------------------------------------------------
    # Order book
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_orderbook(l2_data: dict) -> dict:
        """Normalise Hyperliquid L2 order book snapshot.

        HL format::

            {
                "coin": "BTC",
                "levels": [
                    [{"px": "50000", "sz": "1.5", "n": 3}, ...],  # bids
                    [{"px": "49999", "sz": "0.8", "n": 2}, ...]   # asks
                ]
            }

        Returns a dict with ``symbol``, ``bids``, and ``asks`` where each
        side is a list of ``[price, size]`` pairs.
        """
        coin = l2_data.get("coin", "")
        levels = l2_data.get("levels", [[], []])
        bids_raw = levels[0] if len(levels) > 0 else []
        asks_raw = levels[1] if len(levels) > 1 else []

        bids = [[float(b["px"]), float(b["sz"])] for b in bids_raw]
        asks = [[float(a["px"]), float(a["sz"])] for a in asks_raw]

        return {
            "symbol": HyperliquidNormalizer.coin_to_symbol(coin),
            "bids": bids,
            "asks": asks,
        }

    # ------------------------------------------------------------------
    # Candles
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_candle(candle_data: dict) -> dict:
        """Normalise a Hyperliquid candle.

        Input keys: ``t`` (open ts ms), ``T`` (close ts ms), ``s`` (coin),
        ``i`` (interval), ``o``, ``c``, ``h``, ``l``, ``v``.
        """
        return {
            "timestamp": candle_data.get("t", 0),
            "close_timestamp": candle_data.get("T", 0),
            "symbol": HyperliquidNormalizer.coin_to_symbol(candle_data.get("s", "")),
            "interval": candle_data.get("i", ""),
            "open": float(candle_data.get("o", 0)),
            "high": float(candle_data.get("h", 0)),
            "low": float(candle_data.get("l", 0)),
            "close": float(candle_data.get("c", 0)),
            "volume": float(candle_data.get("v", 0)),
        }

    # ------------------------------------------------------------------
    # Funding
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_funding(funding_data: dict) -> dict:
        """Normalise funding data from the ``metaAndAssetCtxs`` response.

        ``funding_data`` is one element of the ``assetCtxs`` list and looks
        like::

            {
                "funding": "0.0001",
                "openInterest": "1234.5",
                "prevDayPx": "50000",
                "dayNtlVlm": "123456789",
                "premium": "0.00005",
                "oraclePx": "50050",
                "markPx": "50045",
                ...
            }
        """
        return {
            "funding_rate": float(funding_data.get("funding", 0)),
            "open_interest": float(funding_data.get("openInterest", 0)),
            "premium": float(funding_data.get("premium", 0)),
            "oracle_price": float(funding_data.get("oraclePx", 0)),
            "mark_price": float(funding_data.get("markPx", 0)),
            "day_volume": float(funding_data.get("dayNtlVlm", 0)),
            "prev_day_price": float(funding_data.get("prevDayPx", 0)),
        }

    # ------------------------------------------------------------------
    # Symbol <-> coin helpers
    # ------------------------------------------------------------------

    @staticmethod
    def coin_to_symbol(coin: str) -> str:
        """Convert Hyperliquid coin name to canonical symbol.

        ``"BTC"`` -> ``"BTC/USDC"``
        """
        if not coin:
            return ""
        return f"{coin}/USDC"

    @staticmethod
    def symbol_to_coin(symbol: str) -> str:
        """Convert canonical symbol to Hyperliquid coin name.

        ``"BTC/USDC"`` -> ``"BTC"``
        """
        if "/" in symbol:
            return symbol.split("/")[0]
        return symbol
