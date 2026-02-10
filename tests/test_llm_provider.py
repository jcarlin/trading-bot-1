"""Tests for LLM provider abstraction, model factory, and strategy extractor."""

import sys
import os
import json
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai.llm_provider import LLMProvider, AnthropicProvider, OpenAIProvider, DeepSeekProvider
from ai.model_factory import ModelFactory
from intelligence.strategy_extractor import (
    StrategyExtractor,
    ExtractedStrategy,
    DANGEROUS_IMPORTS,
)


# ---------------------------------------------------------------------------
# LLMProvider ABC
# ---------------------------------------------------------------------------

class TestLLMProviderABC(unittest.TestCase):
    """Test that LLMProvider cannot be instantiated directly."""

    def test_cannot_instantiate_abc(self):
        with self.assertRaises(TypeError):
            LLMProvider({})

    def test_default_config_values(self):
        """Concrete subclass inherits default config parsing."""
        class DummyProvider(LLMProvider):
            def _init_client(self): return "ok"
            def complete(self, system, user): return "text"
            def get_provider_name(self): return "dummy"

        p = DummyProvider({})
        self.assertEqual(p.model, "")
        self.assertEqual(p.max_tokens, 2000)
        self.assertEqual(p.temperature, 0.3)
        self.assertEqual(p.timeout, 30)
        self.assertIsNone(p._api_key)

    def test_config_override(self):
        class DummyProvider(LLMProvider):
            def _init_client(self): return "ok"
            def complete(self, system, user): return "text"
            def get_provider_name(self): return "dummy"

        cfg = {"model": "m", "max_tokens": 100, "temperature": 0.9,
               "timeout_seconds": 60, "api_key": "sk-test"}
        p = DummyProvider(cfg)
        self.assertEqual(p.model, "m")
        self.assertEqual(p.max_tokens, 100)
        self.assertEqual(p.temperature, 0.9)
        self.assertEqual(p.timeout, 60)
        self.assertEqual(p._api_key, "sk-test")


# ---------------------------------------------------------------------------
# complete_json base implementation
# ---------------------------------------------------------------------------

class TestCompleteJSON(unittest.TestCase):
    """Test the base complete_json JSON extraction logic."""

    def _make_provider(self, complete_return):
        class Stub(LLMProvider):
            def _init_client(self): return "ok"
            def complete(self_, system, user): return complete_return
            def get_provider_name(self): return "stub"
        return Stub({})

    def test_valid_json(self):
        p = self._make_provider('{"a": 1}')
        self.assertEqual(p.complete_json("s", "u"), {"a": 1})

    def test_json_in_markdown_block(self):
        p = self._make_provider('Here is the result:\n```json\n{"b": 2}\n```\nDone.')
        self.assertEqual(p.complete_json("s", "u"), {"b": 2})

    def test_json_in_code_block(self):
        p = self._make_provider('```\n{"c": 3}\n```')
        self.assertEqual(p.complete_json("s", "u"), {"c": 3})

    def test_invalid_json_returns_none(self):
        p = self._make_provider("not json at all")
        self.assertIsNone(p.complete_json("s", "u"))

    def test_none_input_returns_none(self):
        p = self._make_provider(None)
        self.assertIsNone(p.complete_json("s", "u"))

    def test_empty_string_returns_none(self):
        p = self._make_provider("")
        self.assertIsNone(p.complete_json("s", "u"))


# ---------------------------------------------------------------------------
# AnthropicProvider
# ---------------------------------------------------------------------------

class TestAnthropicProvider(unittest.TestCase):

    def test_default_model(self):
        p = AnthropicProvider({"api_key": "sk"})
        self.assertEqual(p.model, "claude-sonnet-4-5-20250929")

    def test_custom_model(self):
        p = AnthropicProvider({"model": "claude-opus-4-6", "api_key": "sk"})
        self.assertEqual(p.model, "claude-opus-4-6")

    def test_get_provider_name(self):
        p = AnthropicProvider({"api_key": "sk"})
        self.assertEqual(p.get_provider_name(), "anthropic")

    @patch.dict("sys.modules", {"anthropic": None})
    def test_import_error_returns_none(self):
        p = AnthropicProvider({"api_key": "sk"})
        self.assertIsNone(p._init_client())
        self.assertFalse(p.is_available())

    def test_lazy_init_called_once(self):
        p = AnthropicProvider({"api_key": "sk"})
        p._init_client = MagicMock(return_value="client")
        p._initialized = False
        # First call triggers init
        c1 = p._get_client()
        self.assertEqual(c1, "client")
        # Second call uses cached
        c2 = p._get_client()
        self.assertEqual(c2, "client")
        p._init_client.assert_called_once()

    def test_complete_success(self):
        p = AnthropicProvider({"api_key": "sk"})
        mock_client = MagicMock()
        content_block = MagicMock()
        content_block.text = "response text"
        mock_client.messages.create.return_value = MagicMock(content=[content_block])
        p._client = mock_client
        p._initialized = True

        result = p.complete("system prompt", "user prompt")
        self.assertEqual(result, "response text")
        mock_client.messages.create.assert_called_once()

    def test_complete_api_error_returns_none(self):
        p = AnthropicProvider({"api_key": "sk"})
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("API down")
        p._client = mock_client
        p._initialized = True

        result = p.complete("s", "u")
        self.assertIsNone(result)

    def test_complete_no_client_returns_none(self):
        p = AnthropicProvider({"api_key": "sk"})
        p._client = None
        p._initialized = True
        self.assertIsNone(p.complete("s", "u"))

    def test_is_available_true(self):
        p = AnthropicProvider({"api_key": "sk"})
        p._client = MagicMock()
        p._initialized = True
        self.assertTrue(p.is_available())


# ---------------------------------------------------------------------------
# OpenAIProvider
# ---------------------------------------------------------------------------

class TestOpenAIProvider(unittest.TestCase):

    def test_default_model(self):
        p = OpenAIProvider({"api_key": "sk"})
        self.assertEqual(p.model, "gpt-4o")

    def test_get_provider_name(self):
        p = OpenAIProvider({"api_key": "sk"})
        self.assertEqual(p.get_provider_name(), "openai")

    def test_base_url_support(self):
        p = OpenAIProvider({"api_key": "sk", "base_url": "http://localhost:8080"})
        self.assertEqual(p._base_url, "http://localhost:8080")

    def test_base_url_none_by_default(self):
        p = OpenAIProvider({"api_key": "sk"})
        self.assertIsNone(p._base_url)

    @patch.dict("sys.modules", {"openai": None})
    def test_import_error_returns_none(self):
        p = OpenAIProvider({"api_key": "sk"})
        self.assertIsNone(p._init_client())

    def test_complete_success(self):
        p = OpenAIProvider({"api_key": "sk"})
        mock_client = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = "openai response"
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
        p._client = mock_client
        p._initialized = True

        result = p.complete("sys", "usr")
        self.assertEqual(result, "openai response")

    def test_complete_api_error_returns_none(self):
        p = OpenAIProvider({"api_key": "sk"})
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("fail")
        p._client = mock_client
        p._initialized = True
        self.assertIsNone(p.complete("s", "u"))

    def test_complete_no_client_returns_none(self):
        p = OpenAIProvider({"api_key": "sk"})
        p._client = None
        p._initialized = True
        self.assertIsNone(p.complete("s", "u"))


# ---------------------------------------------------------------------------
# DeepSeekProvider
# ---------------------------------------------------------------------------

class TestDeepSeekProvider(unittest.TestCase):

    def test_default_model(self):
        p = DeepSeekProvider({"api_key": "sk"})
        self.assertEqual(p.model, "deepseek-chat")

    def test_base_url_override(self):
        p = DeepSeekProvider({"api_key": "sk"})
        self.assertEqual(p._base_url, "https://api.deepseek.com")

    def test_custom_base_url_preserved(self):
        p = DeepSeekProvider({"api_key": "sk", "base_url": "http://custom"})
        self.assertEqual(p._base_url, "http://custom")

    def test_get_provider_name(self):
        p = DeepSeekProvider({"api_key": "sk"})
        self.assertEqual(p.get_provider_name(), "deepseek")

    def test_inherits_openai(self):
        self.assertTrue(issubclass(DeepSeekProvider, OpenAIProvider))


# ---------------------------------------------------------------------------
# ModelFactory
# ---------------------------------------------------------------------------

class TestModelFactory(unittest.TestCase):

    def test_create_anthropic(self):
        p = ModelFactory.create("anthropic", {"api_key": "sk"})
        self.assertIsInstance(p, AnthropicProvider)

    def test_create_openai(self):
        p = ModelFactory.create("openai", {"api_key": "sk"})
        self.assertIsInstance(p, OpenAIProvider)

    def test_create_deepseek(self):
        p = ModelFactory.create("deepseek", {"api_key": "sk"})
        self.assertIsInstance(p, DeepSeekProvider)

    def test_create_unknown_returns_none(self):
        p = ModelFactory.create("unknown_provider", {})
        self.assertIsNone(p)

    def test_create_case_insensitive(self):
        p = ModelFactory.create("ANTHROPIC", {"api_key": "sk"})
        self.assertIsInstance(p, AnthropicProvider)

    def test_from_config(self):
        p = ModelFactory.from_config({"provider": "openai", "api_key": "sk"})
        self.assertIsInstance(p, OpenAIProvider)

    def test_from_config_no_provider_returns_none(self):
        p = ModelFactory.from_config({"api_key": "sk"})
        self.assertIsNone(p)

    def test_from_config_empty_provider_returns_none(self):
        p = ModelFactory.from_config({"provider": "", "api_key": "sk"})
        self.assertIsNone(p)

    def test_create_with_fallback_first_available(self):
        configs = [
            {"provider": "anthropic", "api_key": "sk"},
            {"provider": "openai", "api_key": "sk"},
        ]
        with patch.object(AnthropicProvider, '_init_client', return_value=MagicMock()):
            result = ModelFactory.create_with_fallback(configs)
        self.assertIsInstance(result, AnthropicProvider)

    def test_create_with_fallback_skips_unavailable(self):
        configs = [
            {"provider": "anthropic", "api_key": "sk"},
            {"provider": "openai", "api_key": "sk"},
        ]
        with patch.object(AnthropicProvider, '_init_client', return_value=None), \
             patch.object(OpenAIProvider, '_init_client', return_value=MagicMock()):
            result = ModelFactory.create_with_fallback(configs)
        self.assertIsInstance(result, OpenAIProvider)

    def test_create_with_fallback_none_available(self):
        configs = [
            {"provider": "anthropic", "api_key": "sk"},
        ]
        with patch.object(AnthropicProvider, '_init_client', return_value=None):
            result = ModelFactory.create_with_fallback(configs)
        self.assertIsNone(result)

    def test_create_with_fallback_empty_list(self):
        result = ModelFactory.create_with_fallback([])
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# StrategyExtractor
# ---------------------------------------------------------------------------

class TestStrategyExtractor(unittest.TestCase):

    def _make_llm(self, complete_return=None, complete_json_return=None, available=True):
        llm = MagicMock()
        llm.is_available.return_value = available
        llm.complete.return_value = complete_return
        llm.complete_json.return_value = complete_json_return
        return llm

    def test_parse_text_removes_urls(self):
        ext = StrategyExtractor(None)
        result = ext.parse_text("Check https://example.com and trade")
        self.assertEqual(result, "Check and trade")

    def test_parse_text_collapses_whitespace(self):
        ext = StrategyExtractor(None)
        result = ext.parse_text("lots   of   spaces\n\nnewlines")
        self.assertEqual(result, "lots of spaces newlines")

    def test_parse_text_strips(self):
        ext = StrategyExtractor(None)
        result = ext.parse_text("  padded  ")
        self.assertEqual(result, "padded")

    def test_extract_rules_success(self):
        rules = {
            "name": "ema_cross",
            "description": "EMA crossover",
            "entry_logic": "fast > slow",
            "exit_logic": "fast < slow",
            "parameters": {"fast": 10, "slow": 20},
        }
        llm = self._make_llm(complete_json_return=rules)
        ext = StrategyExtractor(llm)
        result = ext.extract_rules("Buy when fast EMA crosses above slow EMA")
        self.assertEqual(result, rules)

    def test_extract_rules_missing_keys_returns_none(self):
        llm = self._make_llm(complete_json_return={"name": "partial"})
        ext = StrategyExtractor(llm)
        result = ext.extract_rules("incomplete strategy")
        self.assertIsNone(result)

    def test_extract_rules_llm_unavailable_returns_none(self):
        llm = self._make_llm(available=False)
        ext = StrategyExtractor(llm)
        result = ext.extract_rules("test")
        self.assertIsNone(result)

    def test_extract_rules_llm_returns_none(self):
        llm = self._make_llm(complete_json_return=None)
        ext = StrategyExtractor(llm)
        result = ext.extract_rules("test")
        self.assertIsNone(result)

    def test_generate_code_success(self):
        code = "class MyStrategy(BaseStrategy): pass"
        llm = self._make_llm(complete_return=f"```python\n{code}\n```")
        ext = StrategyExtractor(llm)
        result = ext.generate_code({"name": "my", "description": "", "entry_logic": "", "exit_logic": "", "parameters": {}})
        self.assertEqual(result, code)

    def test_generate_code_plain_response(self):
        code = "class MyStrategy(BaseStrategy): pass"
        llm = self._make_llm(complete_return=code)
        ext = StrategyExtractor(llm)
        result = ext.generate_code({"name": "test"})
        self.assertEqual(result, code)

    def test_generate_code_llm_unavailable(self):
        llm = self._make_llm(available=False)
        ext = StrategyExtractor(llm)
        result = ext.generate_code({"name": "test"})
        self.assertIsNone(result)

    def test_validate_code_valid(self):
        code = "x = 1 + 2\ny = x * 3"
        ext = StrategyExtractor(None)
        is_valid, errors = ext.validate_code(code)
        self.assertTrue(is_valid)
        self.assertEqual(errors, [])

    def test_validate_code_syntax_error(self):
        code = "def broken(:\n    pass"
        ext = StrategyExtractor(None)
        is_valid, errors = ext.validate_code(code)
        self.assertFalse(is_valid)
        self.assertTrue(any("Syntax error" in e for e in errors))

    def test_validate_code_dangerous_import_os(self):
        code = "import os\nx = 1"
        ext = StrategyExtractor(None)
        is_valid, errors = ext.validate_code(code)
        self.assertFalse(is_valid)
        self.assertTrue(any("os" in e for e in errors))

    def test_validate_code_dangerous_from_import(self):
        code = "from subprocess import run\nx = 1"
        ext = StrategyExtractor(None)
        is_valid, errors = ext.validate_code(code)
        self.assertFalse(is_valid)
        self.assertTrue(any("subprocess" in e for e in errors))

    def test_validate_code_safe_imports(self):
        code = "import numpy as np\nimport pandas as pd\nx = 1"
        ext = StrategyExtractor(None)
        is_valid, errors = ext.validate_code(code)
        self.assertTrue(is_valid)

    def test_extract_from_text_full_pipeline(self):
        rules = {
            "name": "test_strat",
            "description": "A test",
            "entry_logic": "price > sma",
            "exit_logic": "price < sma",
            "parameters": {"period": 20},
        }
        valid_code = "x = 1 + 2"
        llm = self._make_llm(
            complete_return=valid_code,
            complete_json_return=rules,
        )
        ext = StrategyExtractor(llm, {"max_retries": 0})
        result = ext.extract_from_text("Buy above SMA", source="test", source_ref="ref1")

        self.assertIsNotNone(result)
        self.assertIsInstance(result, ExtractedStrategy)
        self.assertEqual(result.name, "test_strat")
        self.assertEqual(result.source, "test")
        self.assertEqual(result.source_ref, "ref1")
        self.assertEqual(result.generated_code, valid_code)

    def test_extract_from_text_empty_text_returns_none(self):
        llm = self._make_llm()
        ext = StrategyExtractor(llm)
        result = ext.extract_from_text("   ")
        self.assertIsNone(result)

    def test_extract_from_text_rules_fail_returns_none(self):
        llm = self._make_llm(complete_json_return=None)
        ext = StrategyExtractor(llm, {"max_retries": 0})
        result = ext.extract_from_text("some strategy text")
        self.assertIsNone(result)

    def test_extract_from_text_code_fail_returns_none(self):
        rules = {
            "name": "t", "description": "d", "entry_logic": "e",
            "exit_logic": "x", "parameters": {},
        }
        llm = self._make_llm(complete_return=None, complete_json_return=rules)
        ext = StrategyExtractor(llm, {"max_retries": 0})
        result = ext.extract_from_text("some strategy")
        self.assertIsNone(result)

    def test_extract_from_text_invalid_code_retries(self):
        rules = {
            "name": "t", "description": "d", "entry_logic": "e",
            "exit_logic": "x", "parameters": {},
        }
        # Return invalid code (syntax error), then valid on retry
        bad_code = "def broken(:"
        good_code = "x = 1"
        llm = self._make_llm(complete_json_return=rules)
        llm.complete.side_effect = [bad_code, good_code]
        ext = StrategyExtractor(llm, {"max_retries": 1})
        result = ext.extract_from_text("strategy text")
        self.assertIsNotNone(result)
        self.assertEqual(result.generated_code, good_code)

    def test_extractor_with_none_provider(self):
        ext = StrategyExtractor(None)
        self.assertIsNone(ext.extract_rules("test"))
        self.assertIsNone(ext.generate_code({"name": "test"}))


# ---------------------------------------------------------------------------
# ClaudeClient.get_provider_name()
# ---------------------------------------------------------------------------

class TestClaudeClientProviderName(unittest.TestCase):

    def test_get_provider_name(self):
        from ai.claude_client import ClaudeClient
        client = ClaudeClient({"api_key": "sk"})
        self.assertEqual(client.get_provider_name(), "anthropic")


if __name__ == "__main__":
    unittest.main()
