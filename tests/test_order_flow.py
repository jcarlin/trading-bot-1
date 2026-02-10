"""Tests for Phase 7 Workstream B: Multi-exchange data, liquidation aggregator,
order flow analyzer, and order flow imbalance strategy."""

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.multi_exchange import (
    ExchangeDataSource,
    HyperliquidDataSource,
    BinanceDataSource,
    BybitDataSource,
    MockExchangeDataSource,
    MultiExchangeAggregator,
)
from data.liquidation_aggregator import LiquidationAggregator
from data.order_flow import OrderFlowAnalyzer
from strategy.order_flow_imbalance import (
    OrderFlowImbalanceStrategy,
    OrderFlowImbalanceBacktestAdapter,
)
from core.types import SignalType
from strategy.live_strategy import MarketState


# ---------------------------------------------------------------------------
# ExchangeDataSource tests
# ---------------------------------------------------------------------------

class TestExchangeDataSourceABC(unittest.TestCase):
    def test_cannot_instantiate_abc(self):
        with self.assertRaises(TypeError):
            ExchangeDataSource()

    def test_hyperliquid_returns_empty(self):
        src = HyperliquidDataSource({})
        self.assertEqual(src.get_liquidations("BTC"), [])
        self.assertIsNone(src.get_funding_rate("BTC"))
        self.assertEqual(src.get_recent_trades("BTC"), [])
        self.assertEqual(src.get_exchange_name(), "hyperliquid")

    def test_binance_returns_empty(self):
        src = BinanceDataSource({})
        self.assertEqual(src.get_liquidations("BTC"), [])
        self.assertEqual(src.get_exchange_name(), "binance")

    def test_bybit_returns_empty(self):
        src = BybitDataSource({})
        self.assertEqual(src.get_liquidations("BTC"), [])
        self.assertEqual(src.get_exchange_name(), "bybit")


class TestMockExchangeDataSource(unittest.TestCase):
    def test_returns_stored_liquidations(self):
        liqs = [{"side": "long", "size": 100}]
        src = MockExchangeDataSource("test", liquidations=liqs)
        self.assertEqual(src.get_liquidations("BTC"), liqs)
        self.assertEqual(src.get_exchange_name(), "test")

    def test_returns_stored_funding(self):
        f = {"rate": 0.001, "symbol": "BTC"}
        src = MockExchangeDataSource("x", funding=f)
        self.assertEqual(src.get_funding_rate("BTC"), f)

    def test_returns_stored_trades(self):
        t = [{"price": 100, "size": 1}]
        src = MockExchangeDataSource("x", trades=t)
        self.assertEqual(src.get_recent_trades("BTC"), t)

    def test_defaults_empty(self):
        src = MockExchangeDataSource()
        self.assertEqual(src.get_liquidations("BTC"), [])
        self.assertIsNone(src.get_funding_rate("BTC"))
        self.assertEqual(src.get_recent_trades("BTC"), [])


# ---------------------------------------------------------------------------
# MultiExchangeAggregator tests
# ---------------------------------------------------------------------------

class TestMultiExchangeAggregator(unittest.TestCase):
    def test_merge_liquidations_from_two_sources(self):
        now = datetime.now(timezone.utc)
        s1 = MockExchangeDataSource("ex1", liquidations=[
            {"timestamp": (now - timedelta(seconds=1)).isoformat(), "side": "long", "size": 50},
        ])
        s2 = MockExchangeDataSource("ex2", liquidations=[
            {"timestamp": now.isoformat(), "side": "short", "size": 100},
        ])
        agg = MultiExchangeAggregator([s1, s2])
        result = agg.get_all_liquidations("BTC")
        self.assertEqual(len(result), 2)
        # Most recent first
        self.assertEqual(result[0]["exchange"], "ex2")

    def test_funding_spread(self):
        s1 = MockExchangeDataSource("ex1", funding={"rate": 0.001, "symbol": "BTC"})
        s2 = MockExchangeDataSource("ex2", funding={"rate": 0.003, "symbol": "BTC"})
        agg = MultiExchangeAggregator([s1, s2])
        spread = agg.get_funding_spread("BTC")
        self.assertAlmostEqual(spread["max_rate"], 0.003)
        self.assertAlmostEqual(spread["min_rate"], 0.001)
        self.assertAlmostEqual(spread["spread"], 0.002)
        self.assertEqual(len(spread["exchanges"]), 2)

    def test_empty_sources(self):
        agg = MultiExchangeAggregator([])
        self.assertEqual(agg.get_all_liquidations("BTC"), [])
        spread = agg.get_funding_spread("BTC")
        self.assertEqual(spread["spread"], 0.0)

    def test_source_exception_handled(self):
        src = MagicMock(spec=ExchangeDataSource)
        src.get_liquidations.side_effect = Exception("API error")
        src.get_exchange_name.return_value = "broken"
        agg = MultiExchangeAggregator([src])
        result = agg.get_all_liquidations("BTC")
        self.assertEqual(result, [])

    def test_all_funding_rates(self):
        s1 = MockExchangeDataSource("ex1", funding={"rate": 0.001})
        s2 = MockExchangeDataSource("ex2", funding=None)
        agg = MultiExchangeAggregator([s1, s2])
        rates = agg.get_all_funding_rates("BTC")
        self.assertEqual(len(rates), 1)


# ---------------------------------------------------------------------------
# LiquidationAggregator tests
# ---------------------------------------------------------------------------

class TestLiquidationAggregator(unittest.TestCase):
    def _now(self):
        return datetime.now(timezone.utc)

    def test_add_and_get_imbalance(self):
        agg = LiquidationAggregator({"window_seconds": 600})
        agg.add_liquidation({"timestamp": self._now(), "side": "long", "size": 100})
        agg.add_liquidation({"timestamp": self._now(), "side": "short", "size": 50})
        imb = agg.get_imbalance()
        self.assertAlmostEqual(imb["long_vol"], 100)
        self.assertAlmostEqual(imb["short_vol"], 50)
        self.assertEqual(imb["dominant_side"], "long")

    def test_balanced_imbalance(self):
        agg = LiquidationAggregator({"window_seconds": 600})
        agg.add_liquidation({"timestamp": self._now(), "side": "long", "size": 100})
        agg.add_liquidation({"timestamp": self._now(), "side": "short", "size": 100})
        imb = agg.get_imbalance()
        self.assertAlmostEqual(imb["imbalance_ratio"], 0.5)
        self.assertEqual(imb["dominant_side"], "neutral")

    def test_empty_returns_defaults(self):
        agg = LiquidationAggregator()
        imb = agg.get_imbalance()
        self.assertAlmostEqual(imb["imbalance_ratio"], 0.5)

    def test_batch_add(self):
        agg = LiquidationAggregator({"window_seconds": 600})
        events = [
            {"timestamp": self._now(), "side": "long", "size": 10},
            {"timestamp": self._now(), "side": "short", "size": 20},
        ]
        agg.add_liquidations_batch(events)
        imb = agg.get_imbalance()
        self.assertAlmostEqual(imb["long_vol"], 10)
        self.assertAlmostEqual(imb["short_vol"], 20)

    def test_cascade_detected(self):
        agg = LiquidationAggregator({
            "window_seconds": 600,
            "cascade_threshold": 3,
            "cascade_window_seconds": 60,
        })
        for _ in range(5):
            agg.add_liquidation({"timestamp": self._now(), "side": "long", "size": 10})
        cascade = agg.detect_cascade()
        self.assertIsNotNone(cascade)
        self.assertTrue(cascade["is_cascade"])
        self.assertEqual(cascade["count"], 5)

    def test_no_cascade(self):
        agg = LiquidationAggregator({
            "cascade_threshold": 10,
            "cascade_window_seconds": 60,
        })
        agg.add_liquidation({"timestamp": self._now(), "side": "long", "size": 10})
        cascade = agg.detect_cascade()
        self.assertFalse(cascade["is_cascade"])

    def test_cascade_none_when_empty(self):
        agg = LiquidationAggregator()
        self.assertIsNone(agg.detect_cascade())

    def test_summary(self):
        agg = LiquidationAggregator({"window_seconds": 600})
        agg.add_liquidation({"timestamp": self._now(), "side": "long", "size": 50})
        summary = agg.get_summary()
        self.assertIn("imbalance", summary)
        self.assertIn("cascade", summary)
        self.assertEqual(summary["event_count"], 1)

    def test_to_market_state_metadata(self):
        agg = LiquidationAggregator({"window_seconds": 600})
        agg.add_liquidation({"timestamp": self._now(), "side": "short", "size": 25})
        meta = agg.to_market_state_metadata()
        self.assertIn("liquidation_short_vol", meta)
        self.assertAlmostEqual(meta["liquidation_short_vol"], 25)

    def test_prune_old_events(self):
        agg = LiquidationAggregator({"window_seconds": 1})
        old = self._now() - timedelta(seconds=10)
        agg.add_liquidation({"timestamp": old, "side": "long", "size": 100})
        # After prune the old event should be gone
        imb = agg.get_imbalance()
        self.assertAlmostEqual(imb["long_vol"], 0)


# ---------------------------------------------------------------------------
# OrderFlowAnalyzer tests
# ---------------------------------------------------------------------------

class TestOrderFlowAnalyzer(unittest.TestCase):
    def _now(self):
        return datetime.now(timezone.utc)

    def test_add_trade_and_imbalance(self):
        ofa = OrderFlowAnalyzer({"window_seconds": 600})
        ofa.add_trade({"timestamp": self._now(), "price": 100, "size": 5, "side": "buy"})
        ofa.add_trade({"timestamp": self._now(), "price": 101, "size": 3, "side": "sell"})
        imb = ofa.get_imbalance()
        self.assertAlmostEqual(imb["buy_vol"], 5)
        self.assertAlmostEqual(imb["sell_vol"], 3)
        self.assertGreater(imb["imbalance_ratio"], 0.5)

    def test_empty_imbalance(self):
        ofa = OrderFlowAnalyzer()
        imb = ofa.get_imbalance()
        self.assertAlmostEqual(imb["imbalance_ratio"], 0.5)
        self.assertAlmostEqual(imb["imbalance_pct"], 0.0)

    def test_cvd_bullish(self):
        ofa = OrderFlowAnalyzer({"window_seconds": 600})
        ofa.add_trade({"timestamp": self._now(), "price": 100, "size": 10, "side": "buy"})
        ofa.add_trade({"timestamp": self._now(), "price": 100, "size": 3, "side": "sell"})
        cvd = ofa.get_cumulative_delta()
        self.assertGreater(cvd["cvd"], 0)
        self.assertEqual(cvd["trend"], "bullish")

    def test_cvd_bearish(self):
        ofa = OrderFlowAnalyzer({"window_seconds": 600})
        ofa.add_trade({"timestamp": self._now(), "price": 100, "size": 2, "side": "buy"})
        ofa.add_trade({"timestamp": self._now(), "price": 100, "size": 10, "side": "sell"})
        cvd = ofa.get_cumulative_delta()
        self.assertLess(cvd["cvd"], 0)
        self.assertEqual(cvd["trend"], "bearish")

    def test_cvd_neutral(self):
        ofa = OrderFlowAnalyzer({"window_seconds": 600})
        cvd = ofa.get_cumulative_delta()
        self.assertEqual(cvd["trend"], "neutral")

    def test_batch_add(self):
        ofa = OrderFlowAnalyzer({"window_seconds": 600})
        trades = [
            {"timestamp": self._now(), "price": 100, "size": 5, "side": "buy"},
            {"timestamp": self._now(), "price": 101, "size": 5, "side": "sell"},
        ]
        ofa.add_trades_batch(trades)
        imb = ofa.get_imbalance()
        self.assertAlmostEqual(imb["buy_vol"], 5)
        self.assertAlmostEqual(imb["sell_vol"], 5)

    def test_large_trade_detection(self):
        ofa = OrderFlowAnalyzer({"window_seconds": 600, "large_trade_std_mult": 1.0})
        now = self._now()
        for i in range(10):
            ofa.add_trade({"timestamp": now, "price": 100, "size": 1, "side": "buy"})
        # Add one large trade
        ofa.add_trade({"timestamp": now, "price": 100, "size": 100, "side": "sell"})
        large = ofa.detect_large_trades()
        self.assertTrue(len(large) >= 1)
        self.assertEqual(float(large[0]["size"]), 100)

    def test_no_large_trades_with_few_data(self):
        ofa = OrderFlowAnalyzer({"window_seconds": 600})
        ofa.add_trade({"timestamp": self._now(), "price": 100, "size": 5, "side": "buy"})
        large = ofa.detect_large_trades()
        self.assertEqual(large, [])

    def test_vwap(self):
        ofa = OrderFlowAnalyzer({"window_seconds": 600})
        now = self._now()
        ofa.add_trade({"timestamp": now, "price": 100, "size": 10, "side": "buy"})
        ofa.add_trade({"timestamp": now, "price": 200, "size": 10, "side": "sell"})
        vwap = ofa.get_vwap()
        self.assertAlmostEqual(vwap, 150.0)

    def test_vwap_empty(self):
        ofa = OrderFlowAnalyzer()
        self.assertIsNone(ofa.get_vwap())

    def test_to_market_state_metadata(self):
        ofa = OrderFlowAnalyzer({"window_seconds": 600})
        ofa.add_trade({"timestamp": self._now(), "price": 100, "size": 5, "side": "buy"})
        meta = ofa.to_market_state_metadata()
        self.assertIn("order_flow", meta)
        of = meta["order_flow"]
        self.assertIn("buy_vol", of)
        self.assertIn("cvd", of)
        self.assertIn("vwap", of)


# ---------------------------------------------------------------------------
# OrderFlowImbalanceStrategy tests
# ---------------------------------------------------------------------------

class TestOrderFlowImbalanceStrategy(unittest.TestCase):
    def _make_state(self, price=50000, equity=10000, metadata=None):
        return MarketState(
            mark_price=price,
            mid_price=price,
            bid=price - 1,
            ask=price + 1,
            funding_rate=0.0,
            premium=0.0,
            open_interest=0.0,
            equity=equity,
            cash=5000,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata or {},
        )

    def test_entry_long_on_high_buy_imbalance(self):
        strat = OrderFlowImbalanceStrategy({"cvd_confirmation": True})
        ms = self._make_state(metadata={"order_flow": {
            "buy_vol": 8000, "sell_vol": 2000, "imbalance_ratio": 0.8,
            "cvd_trend": "bullish", "cvd": 6000,
        }})
        sig = strat.on_tick(ms)
        self.assertEqual(sig.signal_type, SignalType.ENTER_LONG)

    def test_entry_short_on_high_sell_imbalance(self):
        strat = OrderFlowImbalanceStrategy({"cvd_confirmation": True})
        ms = self._make_state(metadata={"order_flow": {
            "buy_vol": 2000, "sell_vol": 8000, "imbalance_ratio": 0.2,
            "cvd_trend": "bearish", "cvd": -6000,
        }})
        sig = strat.on_tick(ms)
        self.assertEqual(sig.signal_type, SignalType.ENTER_SHORT)

    def test_no_entry_below_threshold(self):
        strat = OrderFlowImbalanceStrategy({"imbalance_threshold": 0.65})
        ms = self._make_state(metadata={"order_flow": {
            "buy_vol": 5500, "sell_vol": 4500, "imbalance_ratio": 0.55,
            "cvd_trend": "bullish", "cvd": 1000,
        }})
        sig = strat.on_tick(ms)
        self.assertEqual(sig.signal_type, SignalType.HOLD)

    def test_no_entry_without_cvd_confirmation(self):
        strat = OrderFlowImbalanceStrategy({"cvd_confirmation": True})
        ms = self._make_state(metadata={"order_flow": {
            "buy_vol": 8000, "sell_vol": 2000, "imbalance_ratio": 0.8,
            "cvd_trend": "bearish", "cvd": -1000,
        }})
        sig = strat.on_tick(ms)
        self.assertEqual(sig.signal_type, SignalType.HOLD)

    def test_entry_without_cvd_confirmation_disabled(self):
        strat = OrderFlowImbalanceStrategy({"cvd_confirmation": False})
        ms = self._make_state(metadata={"order_flow": {
            "buy_vol": 8000, "sell_vol": 2000, "imbalance_ratio": 0.8,
            "cvd_trend": "bearish", "cvd": -1000,
        }})
        sig = strat.on_tick(ms)
        self.assertEqual(sig.signal_type, SignalType.ENTER_LONG)

    def test_exit_on_max_hold(self):
        strat = OrderFlowImbalanceStrategy({"max_hold_periods": 2, "cvd_confirmation": False})
        # Enter
        ms = self._make_state(metadata={"order_flow": {
            "buy_vol": 8000, "sell_vol": 2000, "imbalance_ratio": 0.8,
            "cvd_trend": "bullish",
        }})
        strat.on_tick(ms)
        # Hold ticks
        for _ in range(2):
            ms2 = self._make_state(metadata={"order_flow": {
                "buy_vol": 8000, "sell_vol": 2000, "imbalance_ratio": 0.8,
                "cvd_trend": "bullish",
            }})
            sig = strat.on_tick(ms2)
        self.assertEqual(sig.signal_type, SignalType.EXIT_LONG)

    def test_exit_on_drawdown(self):
        strat = OrderFlowImbalanceStrategy({
            "drawdown_exit_pct": 2.0, "cvd_confirmation": False,
        })
        # Enter at 50000
        ms = self._make_state(price=50000, metadata={"order_flow": {
            "buy_vol": 8000, "sell_vol": 2000, "imbalance_ratio": 0.8,
            "cvd_trend": "bullish",
        }})
        strat.on_tick(ms)
        # Price drops 3%
        ms2 = self._make_state(price=48500, metadata={"order_flow": {
            "buy_vol": 8000, "sell_vol": 2000, "imbalance_ratio": 0.8,
            "cvd_trend": "bullish",
        }})
        sig = strat.on_tick(ms2)
        self.assertEqual(sig.signal_type, SignalType.EXIT_LONG)

    def test_exit_on_imbalance_normalization(self):
        strat = OrderFlowImbalanceStrategy({"cvd_confirmation": False})
        # Enter long
        ms = self._make_state(metadata={"order_flow": {
            "buy_vol": 8000, "sell_vol": 2000, "imbalance_ratio": 0.8,
            "cvd_trend": "bullish",
        }})
        strat.on_tick(ms)
        # Imbalance normalizes (ratio < 0.5)
        ms2 = self._make_state(metadata={"order_flow": {
            "buy_vol": 3000, "sell_vol": 7000, "imbalance_ratio": 0.3,
            "cvd_trend": "bearish",
        }})
        sig = strat.on_tick(ms2)
        self.assertEqual(sig.signal_type, SignalType.EXIT_LONG)

    def test_hold_on_zero_price(self):
        strat = OrderFlowImbalanceStrategy({})
        ms = self._make_state(price=0)
        sig = strat.on_tick(ms)
        self.assertEqual(sig.signal_type, SignalType.HOLD)

    def test_hold_when_no_order_flow(self):
        strat = OrderFlowImbalanceStrategy({})
        ms = self._make_state(metadata={})
        sig = strat.on_tick(ms)
        self.assertEqual(sig.signal_type, SignalType.HOLD)

    def test_min_volume_filter(self):
        strat = OrderFlowImbalanceStrategy({"min_volume": 5000, "cvd_confirmation": False})
        ms = self._make_state(metadata={"order_flow": {
            "buy_vol": 100, "sell_vol": 10, "imbalance_ratio": 0.9,
            "cvd_trend": "bullish",
        }})
        sig = strat.on_tick(ms)
        self.assertEqual(sig.signal_type, SignalType.HOLD)

    def test_state_get_set(self):
        strat = OrderFlowImbalanceStrategy({})
        strat._position_side = "long"
        strat._entry_price = 50000
        strat._hold_periods = 5
        state = strat.get_state()
        self.assertEqual(state["position_side"], "long")

        strat2 = OrderFlowImbalanceStrategy({})
        strat2.set_state(state)
        self.assertEqual(strat2._position_side, "long")
        self.assertEqual(strat2._entry_price, 50000)

    def test_metadata(self):
        strat = OrderFlowImbalanceStrategy({})
        meta = strat.get_metadata()
        self.assertEqual(meta["name"], "order_flow_imbalance")
        self.assertEqual(meta["category"], "order_flow_imbalance")

    def test_uses_analyzer_when_available(self):
        analyzer = MagicMock()
        analyzer.get_imbalance.return_value = {
            "buy_vol": 8000, "sell_vol": 2000, "imbalance_ratio": 0.8,
            "imbalance_pct": 60,
        }
        analyzer.get_cumulative_delta.return_value = {
            "cvd": 6000, "cvd_normalized": 0.6, "trend": "bullish",
        }
        strat = OrderFlowImbalanceStrategy(
            {"cvd_confirmation": True},
            order_flow_analyzer=analyzer,
        )
        ms = self._make_state()
        sig = strat.on_tick(ms)
        self.assertEqual(sig.signal_type, SignalType.ENTER_LONG)
        analyzer.get_imbalance.assert_called_once()

    def test_cascade_boost_from_aggregator(self):
        analyzer = MagicMock()
        analyzer.get_imbalance.return_value = {
            "buy_vol": 6000, "sell_vol": 4000, "imbalance_ratio": 0.6,
            "imbalance_pct": 20,
        }
        analyzer.get_cumulative_delta.return_value = {
            "cvd": 2000, "cvd_normalized": 0.2, "trend": "bullish",
        }
        liq_agg = MagicMock()
        liq_agg.detect_cascade.return_value = {"is_cascade": True}

        strat = OrderFlowImbalanceStrategy(
            {"cvd_confirmation": True, "imbalance_threshold": 0.65,
             "large_trade_boost": 0.1},
            order_flow_analyzer=analyzer,
            liquidation_aggregator=liq_agg,
        )
        ms = self._make_state()
        sig = strat.on_tick(ms)
        # 0.6 >= 0.65 - 0.1 = 0.55, so entry triggers with cascade boost
        self.assertEqual(sig.signal_type, SignalType.ENTER_LONG)


# ---------------------------------------------------------------------------
# BacktestAdapter tests
# ---------------------------------------------------------------------------

class TestOrderFlowImbalanceBacktestAdapter(unittest.TestCase):
    def _make_df(self, n=5, with_buy_sell=False):
        import pandas as pd
        import numpy as np
        dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        data = {
            "open": np.full(n, 100.0),
            "high": np.full(n, 105.0),
            "low": np.full(n, 95.0),
            "close": np.full(n, 100.0),
            "volume": np.full(n, 1000.0),
        }
        if with_buy_sell:
            data["buy_volume"] = np.full(n, 800.0)
            data["sell_volume"] = np.full(n, 200.0)
        return pd.DataFrame(data, index=dates)

    def test_setup_synthesizes_columns(self):
        adapter = OrderFlowImbalanceBacktestAdapter({})
        df = self._make_df()
        adapter.setup(df)
        self.assertIn("buy_volume", df.columns)
        self.assertIn("sell_volume", df.columns)

    def test_setup_preserves_existing_columns(self):
        adapter = OrderFlowImbalanceBacktestAdapter({})
        df = self._make_df(with_buy_sell=True)
        adapter.setup(df)
        self.assertAlmostEqual(df["buy_volume"].iloc[0], 800.0)

    def test_generate_signal(self):
        adapter = OrderFlowImbalanceBacktestAdapter({"cvd_confirmation": False})
        df = self._make_df()
        adapter.setup(df)
        sig = adapter.generate_signal(0, df)
        self.assertIn(sig.signal_type, list(SignalType))


if __name__ == "__main__":
    unittest.main()
