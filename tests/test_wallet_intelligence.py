"""Tests for the wallet intelligence pipeline."""

import sys
import math
import asyncio
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

from intelligence.wallet_provider import (
    WalletDataProvider, MockWalletProvider, HyperliquidWalletProvider,
)
from intelligence.wallet_scorer import WalletScorer
from intelligence.trade_reconstructor import TradeHistoryReconstructor
from intelligence.pattern_analyzer import WalletPatternAnalyzer
from intelligence.coordinator import WalletIntelligenceCoordinator


class TestWalletProvider(unittest.TestCase):
    """Tests for wallet data providers."""

    def test_mock_provider_returns_leaderboard(self):
        board = [
            {"address": "0xaaa", "pnl": 100000, "roi": 50, "trade_count": 200},
            {"address": "0xbbb", "pnl": 80000, "roi": 40, "trade_count": 150},
        ]
        provider = MockWalletProvider(leaderboard=board)
        result = provider.get_leaderboard(limit=10)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["address"], "0xaaa")

    def test_mock_provider_leaderboard_respects_limit(self):
        board = [{"address": f"0x{i}"} for i in range(10)]
        provider = MockWalletProvider(leaderboard=board)
        result = provider.get_leaderboard(limit=3)
        self.assertEqual(len(result), 3)

    def test_mock_provider_returns_trades(self):
        trades = {
            "0xaaa": [
                {"timestamp": datetime(2025, 1, 1, tzinfo=timezone.utc),
                 "symbol": "BTC", "side": "buy", "size": 1.0, "price": 50000, "pnl": 100},
            ]
        }
        provider = MockWalletProvider(trades=trades)
        result = provider.get_wallet_trades("0xaaa")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["symbol"], "BTC")

    def test_mock_provider_empty_for_unknown_address(self):
        provider = MockWalletProvider()
        self.assertEqual(provider.get_wallet_trades("0xunknown"), [])
        self.assertEqual(provider.get_wallet_positions("0xunknown"), [])

    def test_hyperliquid_provider_init(self):
        provider = HyperliquidWalletProvider({"base_url": "https://api.test"})
        self.assertEqual(provider._base_url, "https://api.test")


class TestWalletScorer(unittest.TestCase):
    """Tests for wallet scoring system."""

    def _make_trades(self, pnls, symbols=None):
        """Helper to create trade list from PnL values."""
        trades = []
        for i, pnl in enumerate(pnls):
            trades.append({
                "pnl": pnl,
                "symbol": (symbols[i] if symbols else "BTC"),
                "size": 1.0,
                "timestamp": datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i),
            })
        return trades

    def test_insufficient_trades_returns_zero(self):
        scorer = WalletScorer({"min_trades": 10})
        result = scorer.score_wallet(self._make_trades([100, 200]), "0xaaa")
        self.assertEqual(result["total_score"], 0.0)
        self.assertEqual(result["grade"], "F")
        self.assertFalse(result["recommended"])

    def test_empty_trades_returns_zero(self):
        scorer = WalletScorer()
        result = scorer.score_wallet([], "0xaaa")
        self.assertEqual(result["total_score"], 0.0)

    def test_profitable_trader_scores_high(self):
        scorer = WalletScorer({"min_trades": 5})
        # Consistent profitable trades across multiple symbols
        trades = self._make_trades(
            [100, 120, 110, 130, 105, 115, 125, 140, 108, 112],
            symbols=["BTC", "ETH", "SOL", "BTC", "ETH", "SOL", "BTC", "ETH", "SOL", "BTC"],
        )
        result = scorer.score_wallet(trades, "0xgood")
        self.assertGreater(result["total_score"], 50)
        self.assertIn(result["grade"], ["A", "B", "C"])

    def test_losing_trader_scores_low(self):
        scorer = WalletScorer({"min_trades": 5})
        trades = self._make_trades([-100, -200, -50, -150, -80, -120, -90, -110, -60, -70])
        result = scorer.score_wallet(trades, "0xbad")
        self.assertLess(result["total_score"], 30)

    def test_grade_thresholds(self):
        scorer = WalletScorer()
        self.assertEqual(scorer._compute_grade(95), "A")
        self.assertEqual(scorer._compute_grade(85), "B")
        self.assertEqual(scorer._compute_grade(75), "C")
        self.assertEqual(scorer._compute_grade(65), "D")
        self.assertEqual(scorer._compute_grade(50), "F")

    def test_rank_wallets_sorts_descending(self):
        scorer = WalletScorer()
        scores = [
            {"address": "0xa", "total_score": 30},
            {"address": "0xb", "total_score": 90},
            {"address": "0xc", "total_score": 60},
        ]
        ranked = scorer.rank_wallets(scores)
        self.assertEqual(ranked[0]["address"], "0xb")
        self.assertEqual(ranked[1]["address"], "0xc")
        self.assertEqual(ranked[2]["address"], "0xa")

    def test_component_weights_sum_to_one(self):
        scorer = WalletScorer()
        total = sum(scorer.weights.values())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_score_clamped_0_100(self):
        scorer = WalletScorer({"min_trades": 1})
        # All positive, high values
        trades = self._make_trades([1000] * 20, symbols=[f"SYM{i}" for i in range(20)])
        result = scorer.score_wallet(trades, "0xmax")
        self.assertLessEqual(result["total_score"], 100.0)
        self.assertGreaterEqual(result["total_score"], 0.0)


class TestTradeReconstructor(unittest.TestCase):
    """Tests for trade history reconstruction."""

    def test_empty_trades(self):
        recon = TradeHistoryReconstructor()
        self.assertEqual(recon.reconstruct_positions([]), [])

    def test_single_round_trip(self):
        recon = TradeHistoryReconstructor()
        t1 = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)
        t2 = datetime(2025, 1, 1, 14, 0, tzinfo=timezone.utc)
        trades = [
            {"timestamp": t1, "symbol": "BTC", "side": "buy", "size": 1.0, "price": 50000, "pnl": 0},
            {"timestamp": t2, "symbol": "BTC", "side": "sell", "size": 1.0, "price": 51000, "pnl": 1000},
        ]
        positions = recon.reconstruct_positions(trades)
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]["symbol"], "BTC")
        self.assertEqual(positions[0]["side"], "long")
        self.assertEqual(positions[0]["total_pnl"], 1000)
        self.assertAlmostEqual(positions[0]["avg_entry"], 50000)
        self.assertAlmostEqual(positions[0]["avg_exit"], 51000)

    def test_hold_duration_calculation(self):
        recon = TradeHistoryReconstructor()
        t1 = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)
        t2 = datetime(2025, 1, 1, 14, 0, tzinfo=timezone.utc)
        trades = [
            {"timestamp": t1, "symbol": "BTC", "side": "buy", "size": 1.0, "price": 50000, "pnl": 0},
            {"timestamp": t2, "symbol": "BTC", "side": "sell", "size": 1.0, "price": 51000, "pnl": 500},
        ]
        positions = recon.reconstruct_positions(trades)
        self.assertEqual(positions[0]["hold_duration"], 4 * 3600)  # 4 hours

    def test_multiple_symbols(self):
        recon = TradeHistoryReconstructor()
        t = datetime(2025, 1, 1, tzinfo=timezone.utc)
        trades = [
            {"timestamp": t, "symbol": "BTC", "side": "buy", "size": 1.0, "price": 50000, "pnl": 0},
            {"timestamp": t, "symbol": "ETH", "side": "buy", "size": 10.0, "price": 3000, "pnl": 0},
            {"timestamp": t + timedelta(hours=1), "symbol": "BTC", "side": "sell", "size": 1.0, "price": 51000, "pnl": 1000},
            {"timestamp": t + timedelta(hours=1), "symbol": "ETH", "side": "sell", "size": 10.0, "price": 3100, "pnl": 1000},
        ]
        positions = recon.reconstruct_positions(trades)
        self.assertEqual(len(positions), 2)
        symbols = {p["symbol"] for p in positions}
        self.assertEqual(symbols, {"BTC", "ETH"})

    def test_trade_stats_computation(self):
        recon = TradeHistoryReconstructor()
        positions = [
            {"total_pnl": 100, "hold_duration": 3600, "symbol": "BTC", "max_size": 1.0},
            {"total_pnl": -50, "hold_duration": 7200, "symbol": "ETH", "max_size": 2.0},
            {"total_pnl": 200, "hold_duration": 1800, "symbol": "BTC", "max_size": 1.5},
        ]
        stats = recon.compute_trade_stats(positions)
        self.assertEqual(stats["total_trades"], 3)
        self.assertAlmostEqual(stats["win_rate"], 66.67, places=1)
        self.assertEqual(stats["max_win"], 200)
        self.assertEqual(stats["max_loss"], -50)
        self.assertEqual(stats["symbols_traded"], 2)
        self.assertAlmostEqual(stats["profit_factor"], 6.0)  # 300/50

    def test_empty_positions_stats(self):
        recon = TradeHistoryReconstructor()
        stats = recon.compute_trade_stats([])
        self.assertEqual(stats["total_trades"], 0)
        self.assertEqual(stats["win_rate"], 0.0)


class TestPatternAnalyzer(unittest.TestCase):
    """Tests for wallet pattern analysis."""

    def _make_positions(self, count=10, hours_between=2):
        """Helper to create mock positions."""
        positions = []
        base = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)
        for i in range(count):
            entry_time = base + timedelta(hours=i * hours_between)
            exit_time = entry_time + timedelta(hours=1)
            positions.append({
                "symbol": "BTC",
                "side": "long",
                "entries": [{"price": 50000, "size": 1.0}],
                "exits": [{"price": 50500, "size": 1.0}],
                "total_pnl": 500,
                "hold_duration": 3600,
                "max_size": 1.0,
                "avg_entry": 50000,
                "avg_exit": 50500,
                "entry_time": entry_time,
                "exit_time": exit_time,
            })
        return positions

    def test_entry_patterns_empty(self):
        analyzer = WalletPatternAnalyzer()
        result = analyzer.analyze_entry_patterns([])
        self.assertEqual(result["time_of_day_bias"], {})
        self.assertEqual(result["day_of_week_bias"], {})

    def test_entry_time_of_day_bias(self):
        analyzer = WalletPatternAnalyzer()
        # All entries at hour 10
        positions = self._make_positions(count=5, hours_between=24)
        result = analyzer.analyze_entry_patterns(positions)
        self.assertIn(10, result["time_of_day_bias"])
        self.assertEqual(result["time_of_day_bias"][10], 100.0)

    def test_sizing_fixed(self):
        analyzer = WalletPatternAnalyzer()
        positions = [{"max_size": 1.0} for _ in range(10)]
        result = analyzer.analyze_sizing_patterns(positions)
        self.assertEqual(result["sizing_type"], "fixed")

    def test_sizing_conviction(self):
        analyzer = WalletPatternAnalyzer()
        positions = [{"max_size": s} for s in [0.1, 5.0, 0.2, 10.0, 0.5, 8.0, 0.3, 7.0]]
        result = analyzer.analyze_sizing_patterns(positions)
        self.assertEqual(result["sizing_type"], "conviction")

    def test_exit_patterns_empty(self):
        analyzer = WalletPatternAnalyzer()
        result = analyzer.analyze_exit_patterns([])
        self.assertEqual(result["exit_type"], "unknown")

    def test_hypothesis_generation(self):
        analyzer = WalletPatternAnalyzer()
        positions = self._make_positions(count=10, hours_between=24)
        hypotheses = analyzer.generate_hypotheses(positions)
        self.assertIsInstance(hypotheses, list)
        # Should detect time-of-day bias (all at hour 10) and fixed sizing
        hypothesis_texts = [h["hypothesis"] for h in hypotheses]
        has_time_hypothesis = any("time-of-day" in h for h in hypothesis_texts)
        has_sizing_hypothesis = any("fixed" in h.lower() for h in hypothesis_texts)
        self.assertTrue(has_time_hypothesis or has_sizing_hypothesis)

    def test_hypotheses_sorted_by_score(self):
        analyzer = WalletPatternAnalyzer()
        positions = self._make_positions(count=10, hours_between=24)
        hypotheses = analyzer.generate_hypotheses(positions)
        if len(hypotheses) >= 2:
            scores = [h["confidence"] * h["explanatory_power"] for h in hypotheses]
            self.assertEqual(scores, sorted(scores, reverse=True))


class TestWalletIntelligenceCoordinator(unittest.TestCase):
    """Tests for the coordinator pipeline."""

    def test_discover_and_score_empty_leaderboard(self):
        provider = MockWalletProvider(leaderboard=[])
        scorer = WalletScorer({"min_trades": 2})
        recon = TradeHistoryReconstructor()
        analyzer = WalletPatternAnalyzer()
        coord = WalletIntelligenceCoordinator(
            provider, scorer, recon, analyzer)

        result = asyncio.run(coord.discover_and_score())
        self.assertEqual(result, [])

    def test_discover_and_score_with_data(self):
        base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
        trades = {
            "0xaaa": [
                {"timestamp": base_time + timedelta(hours=i),
                 "symbol": s, "side": "buy", "size": 1.0, "price": 50000, "pnl": 100}
                for i, s in enumerate(["BTC", "ETH", "SOL", "BTC", "ETH",
                                       "SOL", "BTC", "ETH", "SOL", "BTC"])
            ],
        }
        provider = MockWalletProvider(
            leaderboard=[{"address": "0xaaa", "pnl": 1000}],
            trades=trades,
        )
        scorer = WalletScorer({"min_trades": 5})
        recon = TradeHistoryReconstructor()
        analyzer = WalletPatternAnalyzer()
        coord = WalletIntelligenceCoordinator(
            provider, scorer, recon, analyzer,
            config={"lookback_days": 730})

        result = asyncio.run(coord.discover_and_score())
        self.assertEqual(len(result), 1)
        self.assertGreater(result[0]["total_score"], 0)

    def test_analyze_wallet_empty_trades(self):
        provider = MockWalletProvider()
        scorer = WalletScorer()
        recon = TradeHistoryReconstructor()
        analyzer = WalletPatternAnalyzer()
        coord = WalletIntelligenceCoordinator(
            provider, scorer, recon, analyzer)

        result = asyncio.run(coord.analyze_wallet("0xunknown"))
        self.assertEqual(result["positions"], [])
        self.assertEqual(result["hypotheses"], [])

    def test_run_discovery_cycle(self):
        base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
        trades = {
            "0xaaa": [
                {"timestamp": base_time + timedelta(hours=i),
                 "symbol": "BTC", "side": "buy", "size": 1.0, "price": 50000, "pnl": 100}
                for i in range(15)
            ],
        }
        provider = MockWalletProvider(
            leaderboard=[{"address": "0xaaa", "pnl": 1500}],
            trades=trades,
        )
        scorer = WalletScorer({"min_trades": 5, "recommend_threshold": 30})
        recon = TradeHistoryReconstructor()
        analyzer = WalletPatternAnalyzer()
        coord = WalletIntelligenceCoordinator(
            provider, scorer, recon, analyzer,
            config={"top_n_analyze": 5, "lookback_days": 730})

        result = asyncio.run(coord.run_discovery_cycle())
        self.assertEqual(result["discovered"], 1)
        self.assertGreater(result["top_score"], 0)


if __name__ == "__main__":
    unittest.main()
