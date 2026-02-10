"""Phase 7 integration tests.

Tests import availability, config parsing, OODA enrichment,
and storage round-trips for Phase 7 components.
"""

import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestPhase7Imports(unittest.TestCase):
    """Verify all Phase 7 modules are importable."""

    def test_llm_provider_importable(self):
        from ai.llm_provider import LLMProvider, AnthropicProvider, OpenAIProvider, DeepSeekProvider
        self.assertTrue(callable(AnthropicProvider))

    def test_model_factory_importable(self):
        from ai.model_factory import ModelFactory
        self.assertTrue(callable(ModelFactory.create))

    def test_strategy_extractor_importable(self):
        from intelligence.strategy_extractor import StrategyExtractor, ExtractedStrategy
        self.assertTrue(callable(StrategyExtractor))

    def test_multi_exchange_importable(self):
        from data.multi_exchange import (
            ExchangeDataSource, MultiExchangeAggregator,
            MockExchangeDataSource,
        )
        self.assertTrue(callable(MultiExchangeAggregator))

    def test_liquidation_aggregator_importable(self):
        from data.liquidation_aggregator import LiquidationAggregator
        self.assertTrue(callable(LiquidationAggregator))

    def test_order_flow_importable(self):
        from data.order_flow import OrderFlowAnalyzer
        self.assertTrue(callable(OrderFlowAnalyzer))

    def test_order_flow_imbalance_strategy_importable(self):
        from strategy.order_flow_imbalance import (
            OrderFlowImbalanceStrategy,
            OrderFlowImbalanceBacktestAdapter,
        )
        self.assertTrue(callable(OrderFlowImbalanceStrategy))

    def test_trader_ranking_importable(self):
        from intelligence.trader_ranking import TraderRankingSystem
        self.assertTrue(callable(TraderRankingSystem))

    def test_hlp_sentiment_importable(self):
        from intelligence.hlp_sentiment import HLPSentimentTracker
        self.assertTrue(callable(HLPSentimentTracker))

    def test_correlation_divergence_importable(self):
        from strategy.correlation_divergence import (
            CorrelationDivergenceStrategy,
            CorrelationDivergenceBacktestAdapter,
        )
        self.assertTrue(callable(CorrelationDivergenceStrategy))

    def test_phase7_available_flag(self):
        """Simulate the PHASE7_AVAILABLE import block from run_strategy.py."""
        try:
            from ai.llm_provider import LLMProvider
            from ai.model_factory import ModelFactory
            from intelligence.strategy_extractor import StrategyExtractor
            from intelligence.trader_ranking import TraderRankingSystem
            from intelligence.hlp_sentiment import HLPSentimentTracker
            from data.multi_exchange import MultiExchangeAggregator
            from data.liquidation_aggregator import LiquidationAggregator
            from data.order_flow import OrderFlowAnalyzer
            phase7_available = True
        except ImportError:
            phase7_available = False
        self.assertTrue(phase7_available)


class TestPhase7Config(unittest.TestCase):
    """Verify phase7.yaml parses correctly."""

    def test_config_loads(self):
        from core.config import Config
        config = Config.from_yaml("config/phase7.yaml")
        self.assertIsNotNone(config)

    def test_new_strategy_params(self):
        from core.config import Config
        config = Config.from_yaml("config/phase7.yaml")

        of_params = config.get_section("strategies.order_flow_imbalance")
        self.assertEqual(of_params["imbalance_threshold"], 0.65)
        self.assertTrue(of_params["cvd_confirmation"])
        self.assertEqual(of_params["max_hold_periods"], 12)

        cd_params = config.get_section("strategies.correlation_divergence")
        self.assertEqual(cd_params["correlation_window"], 168)
        self.assertEqual(cd_params["zscore_threshold"], 2.0)

    def test_llm_config(self):
        from core.config import Config
        config = Config.from_yaml("config/phase7.yaml")
        self.assertEqual(config.get("llm.primary_provider"), "anthropic")

    def test_order_flow_config(self):
        from core.config import Config
        config = Config.from_yaml("config/phase7.yaml")
        self.assertTrue(config.get("order_flow.enabled"))
        self.assertEqual(config.get("order_flow.window_seconds"), 300)

    def test_trader_ranking_config(self):
        from core.config import Config
        config = Config.from_yaml("config/phase7.yaml")
        self.assertTrue(config.get("trader_ranking.enabled"))
        self.assertEqual(config.get("trader_ranking.top_n"), 100)

    def test_hlp_sentiment_config(self):
        from core.config import Config
        config = Config.from_yaml("config/phase7.yaml")
        self.assertTrue(config.get("hlp_sentiment.enabled"))

    def test_strategies_include_new(self):
        from core.config import Config
        config = Config.from_yaml("config/phase7.yaml")
        strategies = config.get("strategy_runner.strategies")
        names = [s["name"] if isinstance(s, dict) else s for s in strategies]
        self.assertIn("order_flow_imbalance", names)
        self.assertIn("correlation_divergence", names)


class TestPhase7OODAIntegration(unittest.TestCase):
    """Verify OODA orchestrator accepts Phase 7 components."""

    def _make_orchestrator(self, **kwargs):
        from orchestration.ooda import OODAOrchestrator
        return OODAOrchestrator(
            strategy_manager=MagicMock(),
            health_scorer=MagicMock(),
            regime_classifier=MagicMock(),
            timescale=MagicMock(),
            redis_store=MagicMock(),
            config={"symbol": "BTC/USDC", "strategy_name": "test"},
            **kwargs,
        )

    def test_accepts_phase7_kwargs(self):
        ooda = self._make_orchestrator(
            order_flow_analyzer=MagicMock(),
            liquidation_aggregator=MagicMock(),
            hlp_sentiment=MagicMock(),
            trader_ranking=MagicMock(),
        )
        self.assertIsNotNone(ooda.order_flow_analyzer)
        self.assertIsNotNone(ooda.liquidation_aggregator)
        self.assertIsNotNone(ooda.hlp_sentiment)
        self.assertIsNotNone(ooda.trader_ranking)

    def test_regime_suitability_includes_new_strategies(self):
        ooda = self._make_orchestrator()
        self.assertIn("order_flow_imbalance", ooda.regime_suitability)
        self.assertIn("correlation_divergence", ooda.regime_suitability)
        self.assertIn("trending_up", ooda.regime_suitability["order_flow_imbalance"])
        self.assertIn("volatile", ooda.regime_suitability["order_flow_imbalance"])
        self.assertIn("ranging", ooda.regime_suitability["correlation_divergence"])

    def test_observe_collects_order_flow(self):
        of_analyzer = MagicMock()
        of_analyzer.get_imbalance.return_value = {"imbalance_ratio": 0.7}
        of_analyzer.get_cumulative_delta.return_value = {"cvd": 100.0}

        ooda = self._make_orchestrator(order_flow_analyzer=of_analyzer)
        ooda.strategy_manager.get_active_strategies.return_value = ["test"]
        ooda.health_scorer.compute_health_score.return_value = {
            "health_score": 80, "grade": "B"}

        metrics = ooda._observe("hourly")
        self.assertIn("order_flow", metrics)
        self.assertEqual(metrics["order_flow"]["imbalance"]["imbalance_ratio"], 0.7)

    def test_observe_collects_liquidations(self):
        liq_agg = MagicMock()
        liq_agg.get_summary.return_value = {"total_volume": 1000}
        liq_agg.detect_cascade.return_value = None

        ooda = self._make_orchestrator(liquidation_aggregator=liq_agg)
        ooda.strategy_manager.get_active_strategies.return_value = ["test"]
        ooda.health_scorer.compute_health_score.return_value = {
            "health_score": 80, "grade": "B"}

        metrics = ooda._observe("hourly")
        self.assertIn("liquidations", metrics)

    def test_observe_collects_hlp_sentiment(self):
        hlp = MagicMock()
        hlp.get_current_sentiment.return_value = {
            "hlp_direction": "long", "net_notional": 50000}

        ooda = self._make_orchestrator(hlp_sentiment=hlp)
        ooda.strategy_manager.get_active_strategies.return_value = ["test"]
        ooda.health_scorer.compute_health_score.return_value = {
            "health_score": 80, "grade": "B"}

        metrics = ooda._observe("hourly")
        self.assertIn("hlp_sentiment", metrics)
        self.assertEqual(metrics["hlp_sentiment"]["hlp_direction"], "long")

    def test_observe_collects_trader_ranking(self):
        tr = MagicMock()
        tr.get_smart_money_consensus.return_value = {
            "direction": "long", "confidence": 0.8}
        tr.get_contrarian_signal.return_value = {
            "direction": "short", "confidence": 0.7}

        ooda = self._make_orchestrator(trader_ranking=tr)
        ooda.strategy_manager.get_active_strategies.return_value = ["test"]
        ooda.health_scorer.compute_health_score.return_value = {
            "health_score": 80, "grade": "B"}

        metrics = ooda._observe("hourly")
        self.assertIn("trader_ranking", metrics)
        self.assertEqual(
            metrics["trader_ranking"]["smart_money"]["direction"], "long")

    def test_observe_graceful_without_phase7(self):
        """OODA works without Phase 7 components (backward compat)."""
        ooda = self._make_orchestrator()
        ooda.strategy_manager.get_active_strategies.return_value = ["test"]
        ooda.health_scorer.compute_health_score.return_value = {
            "health_score": 80, "grade": "B"}

        metrics = ooda._observe("hourly")
        self.assertNotIn("order_flow", metrics)
        self.assertNotIn("liquidations", metrics)
        self.assertNotIn("hlp_sentiment", metrics)
        self.assertNotIn("trader_ranking", metrics)


class TestPhase7Metrics(unittest.TestCase):
    """Verify Phase 7 Prometheus metrics are defined."""

    def test_order_flow_metrics(self):
        from monitoring.metrics import (
            order_flow_imbalance_ratio,
            order_flow_cvd,
            large_trade_detected_total,
        )
        self.assertIsNotNone(order_flow_imbalance_ratio)
        self.assertIsNotNone(order_flow_cvd)
        self.assertIsNotNone(large_trade_detected_total)

    def test_liquidation_metrics(self):
        from monitoring.metrics import (
            liquidation_cascade_active,
            liquidation_imbalance_ratio,
            cross_exchange_liquidation_count,
        )
        self.assertIsNotNone(liquidation_cascade_active)
        self.assertIsNotNone(liquidation_imbalance_ratio)

    def test_trader_ranking_metrics(self):
        from monitoring.metrics import (
            trader_top_consensus_pct,
            trader_bottom_bias_pct,
            contrarian_signal_confidence,
        )
        self.assertIsNotNone(trader_top_consensus_pct)
        self.assertIsNotNone(contrarian_signal_confidence)

    def test_hlp_metrics(self):
        from monitoring.metrics import (
            hlp_net_exposure,
            hlp_sentiment_direction,
        )
        self.assertIsNotNone(hlp_net_exposure)
        self.assertIsNotNone(hlp_sentiment_direction)

    def test_llm_metrics(self):
        from monitoring.metrics import (
            llm_requests_total,
            llm_latency_seconds,
            strategy_extraction_count,
        )
        self.assertIsNotNone(llm_requests_total)
        self.assertIsNotNone(llm_latency_seconds)
        self.assertIsNotNone(strategy_extraction_count)


class TestPhase7StorageMethods(unittest.TestCase):
    """Verify Phase 7 storage methods exist (mock-based)."""

    def test_timescale_order_flow_methods_exist(self):
        from storage.timescale import TimescaleStore
        self.assertTrue(hasattr(TimescaleStore, 'insert_order_flow_snapshot'))
        self.assertTrue(hasattr(TimescaleStore, 'query_order_flow_snapshots'))

    def test_timescale_liquidation_method_exists(self):
        from storage.timescale import TimescaleStore
        self.assertTrue(hasattr(TimescaleStore, 'insert_liquidation_event'))

    def test_timescale_trader_ranking_method_exists(self):
        from storage.timescale import TimescaleStore
        self.assertTrue(hasattr(TimescaleStore, 'insert_trader_ranking_snapshot'))

    def test_timescale_hlp_sentiment_methods_exist(self):
        from storage.timescale import TimescaleStore
        self.assertTrue(hasattr(TimescaleStore, 'insert_hlp_sentiment'))
        self.assertTrue(hasattr(TimescaleStore, 'query_hlp_sentiment'))

    def test_timescale_strategy_extraction_method_exists(self):
        from storage.timescale import TimescaleStore
        self.assertTrue(hasattr(TimescaleStore, 'insert_strategy_extraction'))

    def test_redis_order_flow_methods_exist(self):
        from storage.redis_store import RedisStore
        self.assertTrue(hasattr(RedisStore, 'set_order_flow'))
        self.assertTrue(hasattr(RedisStore, 'get_order_flow'))

    def test_redis_liquidation_methods_exist(self):
        from storage.redis_store import RedisStore
        self.assertTrue(hasattr(RedisStore, 'set_liquidation_state'))
        self.assertTrue(hasattr(RedisStore, 'get_liquidation_state'))

    def test_redis_hlp_methods_exist(self):
        from storage.redis_store import RedisStore
        self.assertTrue(hasattr(RedisStore, 'set_hlp_sentiment'))
        self.assertTrue(hasattr(RedisStore, 'get_hlp_sentiment'))

    def test_redis_trader_ranking_methods_exist(self):
        from storage.redis_store import RedisStore
        self.assertTrue(hasattr(RedisStore, 'set_trader_rankings'))
        self.assertTrue(hasattr(RedisStore, 'get_trader_rankings'))


class TestPhase5ConfigBackwardCompat(unittest.TestCase):
    """Verify phase5.yaml still loads without Phase 7 sections."""

    def test_phase5_config_loads(self):
        from core.config import Config
        config = Config.from_yaml("config/phase5.yaml")
        self.assertIsNotNone(config)
        # Phase 7 sections should not be present
        self.assertIsNone(config.get("order_flow.enabled", None))
        self.assertIsNone(config.get("trader_ranking.enabled", None))


if __name__ == "__main__":
    unittest.main()
