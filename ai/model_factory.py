"""Factory for creating LLM provider instances.

Maps provider name strings to concrete LLMProvider subclasses and
supports fallback chains for resilient multi-provider setups.
"""

import logging
from typing import Optional

from ai.llm_provider import (
    AnthropicProvider,
    DeepSeekProvider,
    LLMProvider,
    OpenAIProvider,
)

logger = logging.getLogger(__name__)

_PROVIDER_MAP = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "deepseek": DeepSeekProvider,
}


class ModelFactory:
    """Creates LLMProvider instances from config dicts."""

    @staticmethod
    def create(provider_name: str, config: dict) -> Optional[LLMProvider]:
        """Create a provider by name.

        Args:
            provider_name: One of "anthropic", "openai", "deepseek".
            config: Provider configuration dict.

        Returns:
            LLMProvider instance, or None for unknown provider names.
        """
        cls = _PROVIDER_MAP.get(provider_name.lower())
        if cls is None:
            logger.warning("Unknown LLM provider: %s", provider_name)
            return None
        return cls(config)

    @staticmethod
    def from_config(config: dict) -> Optional[LLMProvider]:
        """Create a provider from a config dict containing a 'provider' key.

        Args:
            config: Dict with at least a "provider" key naming the provider.

        Returns:
            LLMProvider instance, or None if no provider specified.
        """
        provider_name = config.get("provider")
        if not provider_name:
            logger.warning("No 'provider' key in config")
            return None
        return ModelFactory.create(provider_name, config)

    @staticmethod
    def create_with_fallback(configs: list[dict]) -> Optional[LLMProvider]:
        """Try each config in order, returning the first available provider.

        Args:
            configs: List of config dicts, each with a "provider" key.

        Returns:
            First available LLMProvider, or None if none are available.
        """
        for cfg in configs:
            provider = ModelFactory.from_config(cfg)
            if provider is not None and provider.is_available():
                return provider
        return None
