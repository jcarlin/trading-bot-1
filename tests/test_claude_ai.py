"""Tests for the Claude AI integration layer."""

import sys
import json
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.claude_client import ClaudeClient
from ai.decision_engine import AIDecisionEngine, DECISION_SYSTEM_PROMPT
from ai.strategy_analyst import AIStrategyAnalyst
from ai.report_writer import AIReportWriter


class TestClaudeClient(unittest.TestCase):
    """Tests for ClaudeClient wrapper."""

    def test_init_with_config(self):
        client = ClaudeClient({"model": "test-model", "max_tokens": 1000})
        self.assertEqual(client.model, "test-model")
        self.assertEqual(client.max_tokens, 1000)

    def test_default_config(self):
        client = ClaudeClient({})
        self.assertEqual(client.model, "claude-sonnet-4-5-20250929")
        self.assertEqual(client.max_tokens, 2000)
        self.assertAlmostEqual(client.temperature, 0.3)

    @patch.dict('sys.modules', {'anthropic': None})
    def test_is_available_false_without_package(self):
        client = ClaudeClient({})
        client._initialized = False
        client._client = None
        # Force re-init
        result = client._get_client()
        # Should handle ImportError gracefully

    def test_complete_returns_none_when_no_client(self):
        client = ClaudeClient({})
        client._initialized = True
        client._client = None
        result = client.complete("system", "user")
        self.assertIsNone(result)

    def test_complete_returns_text(self):
        client = ClaudeClient({})
        mock_anthropic = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hello world")]
        mock_anthropic.messages.create.return_value = mock_response
        client._initialized = True
        client._client = mock_anthropic

        result = client.complete("system", "user")
        self.assertEqual(result, "Hello world")

    def test_complete_returns_none_on_api_error(self):
        client = ClaudeClient({})
        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.side_effect = Exception("API error")
        client._initialized = True
        client._client = mock_anthropic

        result = client.complete("system", "user")
        self.assertIsNone(result)

    def test_complete_json_parses_valid_json(self):
        client = ClaudeClient({})
        mock_anthropic = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"action": "pause_strategy"}')]
        mock_anthropic.messages.create.return_value = mock_response
        client._initialized = True
        client._client = mock_anthropic

        result = client.complete_json("system", "user")
        self.assertEqual(result, {"action": "pause_strategy"})

    def test_complete_json_extracts_from_code_block(self):
        client = ClaudeClient({})
        mock_anthropic = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(
            text='Here is the result:\n```json\n{"action": "no_action"}\n```')]
        mock_anthropic.messages.create.return_value = mock_response
        client._initialized = True
        client._client = mock_anthropic

        result = client.complete_json("system", "user")
        self.assertEqual(result, {"action": "no_action"})

    def test_complete_json_returns_none_on_invalid_json(self):
        client = ClaudeClient({})
        mock_anthropic = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="not valid json at all")]
        mock_anthropic.messages.create.return_value = mock_response
        client._initialized = True
        client._client = mock_anthropic

        result = client.complete_json("system", "user")
        self.assertIsNone(result)

    def test_is_available_true_with_client(self):
        client = ClaudeClient({})
        client._initialized = True
        client._client = MagicMock()
        self.assertTrue(client.is_available())

    def test_is_available_false_without_client(self):
        client = ClaudeClient({})
        client._initialized = True
        client._client = None
        self.assertFalse(client.is_available())


class TestAIDecisionEngine(unittest.TestCase):
    """Tests for AIDecisionEngine."""

    def setUp(self):
        self.mock_claude = MagicMock()
        self.mock_claude.is_available.return_value = True
        self.engine = AIDecisionEngine(self.mock_claude, {"min_confidence": 0.6})

    def test_decide_returns_none_when_disabled(self):
        engine = AIDecisionEngine(self.mock_claude, {"enabled": False})
        result = engine.decide({}, {}, {}, "hourly", [])
        self.assertIsNone(result)

    def test_decide_returns_none_when_unavailable(self):
        self.mock_claude.is_available.return_value = False
        result = self.engine.decide({}, {}, {}, "hourly", [])
        self.assertIsNone(result)

    def test_decide_returns_none_when_claude_returns_none(self):
        self.mock_claude.complete_json.return_value = None
        result = self.engine.decide({}, {}, {}, "hourly", [])
        self.assertIsNone(result)

    def test_decide_returns_none_for_no_action(self):
        self.mock_claude.complete_json.return_value = {"action": "no_action"}
        result = self.engine.decide({}, {}, {}, "hourly", [])
        self.assertIsNone(result)

    def test_decide_returns_none_below_confidence(self):
        self.mock_claude.complete_json.return_value = {
            "action": "pause_strategy",
            "confidence": 0.3,
        }
        result = self.engine.decide({}, {}, {}, "hourly", [])
        self.assertIsNone(result)

    def test_decide_returns_decision_above_confidence(self):
        self.mock_claude.complete_json.return_value = {
            "action": "pause_strategy",
            "strategy_name": "funding_rate_arb",
            "reason": "Health critically low",
            "hypothesis": "Pausing will prevent losses",
            "confidence": 0.8,
            "alternatives": [],
            "reasoning_chain": ["Step 1", "Step 2"],
        }
        result = self.engine.decide({}, {}, {}, "hourly", ["funding_rate_arb"])
        self.assertIsNotNone(result)
        self.assertEqual(result["action"], "pause_strategy")
        self.assertEqual(result["source"], "ai_engine")
        self.assertEqual(result["confidence"], 0.8)

    def test_decide_rejects_invalid_action(self):
        self.mock_claude.complete_json.return_value = {
            "action": "destroy_everything",
            "confidence": 0.9,
        }
        result = self.engine.decide({}, {}, {}, "hourly", [])
        self.assertIsNone(result)

    def test_sanitize_metrics_handles_nested(self):
        metrics = {
            "health": {"health_score": 75.0, "grade": "B"},
            "report": "A" * 5000,  # Long string
            "complex": {"nested": object()},  # Non-serializable
        }
        sanitized = self.engine._sanitize_metrics(metrics)
        self.assertEqual(sanitized["health"]["health_score"], 75.0)
        self.assertEqual(len(sanitized["report"]), 2000)  # Truncated

    def test_build_context_is_valid_json(self):
        context = self.engine._build_context(
            {"health": {"score": 50}}, {"regime": "trending"},
            {"suitable": True}, "hourly", ["strat1"])
        parsed = json.loads(context)
        self.assertEqual(parsed["checkpoint_type"], "hourly")


class TestAIStrategyAnalyst(unittest.TestCase):
    """Tests for AIStrategyAnalyst."""

    def test_returns_none_when_unavailable(self):
        mock_claude = MagicMock()
        mock_claude.is_available.return_value = False
        analyst = AIStrategyAnalyst(mock_claude)
        result = analyst.analyze_strategy("test", {}, {}, {})
        self.assertIsNone(result)

    def test_analyze_strategy_calls_claude(self):
        mock_claude = MagicMock()
        mock_claude.is_available.return_value = True
        mock_claude.complete_json.return_value = {"assessment": "Good"}
        analyst = AIStrategyAnalyst(mock_claude)
        result = analyst.analyze_strategy("test", {"pnl": 100}, {"score": 80}, {"regime": "trending"})
        self.assertEqual(result, {"assessment": "Good"})
        mock_claude.complete_json.assert_called_once()

    def test_compare_strategies_calls_claude(self):
        mock_claude = MagicMock()
        mock_claude.is_available.return_value = True
        mock_claude.complete_json.return_value = {"rankings": ["a", "b"]}
        analyst = AIStrategyAnalyst(mock_claude)
        result = analyst.compare_strategies({"a": {}, "b": {}})
        self.assertEqual(result["rankings"], ["a", "b"])


class TestAIReportWriter(unittest.TestCase):
    """Tests for AIReportWriter."""

    def test_returns_none_when_unavailable(self):
        mock_claude = MagicMock()
        mock_claude.is_available.return_value = False
        writer = AIReportWriter(mock_claude)
        result = writer.generate_weekly_report({}, {}, [], {})
        self.assertIsNone(result)

    def test_generate_weekly_report_returns_text(self):
        mock_claude = MagicMock()
        mock_claude.is_available.return_value = True
        mock_claude.complete.return_value = "Weekly report text"
        writer = AIReportWriter(mock_claude)
        result = writer.generate_weekly_report({"pnl": 100}, {}, [], {})
        self.assertEqual(result, "Weekly report text")

    def test_generate_audit_narrative(self):
        mock_claude = MagicMock()
        mock_claude.is_available.return_value = True
        mock_claude.complete.return_value = "Audit narrative"
        writer = AIReportWriter(mock_claude)
        result = writer.generate_audit_narrative({"decisions": 10})
        self.assertEqual(result, "Audit narrative")

    def test_generate_wallet_briefing(self):
        mock_claude = MagicMock()
        mock_claude.is_available.return_value = True
        mock_claude.complete.return_value = "Wallet briefing"
        writer = AIReportWriter(mock_claude)
        result = writer.generate_wallet_briefing({"top_score": 85})
        self.assertEqual(result, "Wallet briefing")


if __name__ == "__main__":
    unittest.main()
