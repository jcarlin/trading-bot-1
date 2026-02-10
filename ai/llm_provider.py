"""LLM provider abstraction for multi-model support.

Defines a common interface for interacting with different LLM providers
(Anthropic, OpenAI, DeepSeek, etc.) with lazy initialization and graceful
fallback when SDKs are not installed.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Provides lazy client initialization, availability checking, and
    JSON extraction from LLM responses. Subclasses implement SDK-specific
    initialization and completion logic.
    """

    def __init__(self, config: dict):
        self.model = config.get("model", "")
        self.max_tokens = config.get("max_tokens", 2000)
        self.temperature = config.get("temperature", 0.3)
        self.timeout = config.get("timeout_seconds", 30)
        self._api_key = config.get("api_key") or None
        self._client = None
        self._initialized = False

    @abstractmethod
    def _init_client(self) -> object:
        """Initialize the SDK client. Return client object or None on failure."""
        ...

    def _get_client(self):
        """Lazy-init the underlying SDK client."""
        if self._initialized:
            return self._client
        self._initialized = True
        self._client = self._init_client()
        return self._client

    def is_available(self) -> bool:
        """Check if the provider client is usable."""
        return self._get_client() is not None

    @abstractmethod
    def complete(self, system: str, user: str) -> Optional[str]:
        """Send a completion request. Returns response text or None."""
        ...

    def complete_json(self, system: str, user: str) -> Optional[dict]:
        """Request structured JSON output from the provider.

        Extracts JSON from markdown code blocks if present.
        Returns parsed dict, or None on failure.
        """
        text = self.complete(system, user)
        if text is None:
            return None
        try:
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            return json.loads(text.strip())
        except (json.JSONDecodeError, IndexError):
            logger.warning("Failed to parse JSON response from %s", self.get_provider_name())
            return None

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return a string identifier for this provider."""
        ...


class AnthropicProvider(LLMProvider):
    """LLM provider wrapping the Anthropic Python SDK."""

    def __init__(self, config: dict):
        if not config.get("model"):
            config = {**config, "model": "claude-sonnet-4-5-20250929"}
        super().__init__(config)

    def _init_client(self) -> object:
        try:
            import anthropic
            return anthropic.Anthropic(
                api_key=self._api_key,
                timeout=self.timeout,
            )
        except ImportError:
            logger.warning("anthropic package not installed, Anthropic provider disabled")
            return None
        except Exception:
            logger.exception("Failed to initialize Anthropic client")
            return None

    def complete(self, system: str, user: str) -> Optional[str]:
        client = self._get_client()
        if client is None:
            return None
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return response.content[0].text
        except Exception:
            logger.exception("Anthropic API call failed")
            return None

    def get_provider_name(self) -> str:
        return "anthropic"


class OpenAIProvider(LLMProvider):
    """LLM provider wrapping the OpenAI Python SDK."""

    def __init__(self, config: dict):
        if not config.get("model"):
            config = {**config, "model": "gpt-4o"}
        self._base_url = config.get("base_url") or None
        super().__init__(config)

    def _init_client(self) -> object:
        try:
            import openai
            kwargs = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            return openai.OpenAI(**kwargs)
        except ImportError:
            logger.warning("openai package not installed, OpenAI provider disabled")
            return None
        except Exception:
            logger.exception("Failed to initialize OpenAI client")
            return None

    def complete(self, system: str, user: str) -> Optional[str]:
        client = self._get_client()
        if client is None:
            return None
        try:
            response = client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return response.choices[0].message.content
        except Exception:
            logger.exception("OpenAI API call failed")
            return None

    def get_provider_name(self) -> str:
        return "openai"


class DeepSeekProvider(OpenAIProvider):
    """LLM provider for DeepSeek, using the OpenAI-compatible API."""

    DEEPSEEK_BASE_URL = "https://api.deepseek.com"

    def __init__(self, config: dict):
        if not config.get("model"):
            config = {**config, "model": "deepseek-chat"}
        if not config.get("base_url"):
            config = {**config, "base_url": self.DEEPSEEK_BASE_URL}
        super().__init__(config)

    def get_provider_name(self) -> str:
        return "deepseek"
