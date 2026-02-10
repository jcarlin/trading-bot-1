"""Thin wrapper around the Anthropic Python SDK for trading bot use."""

import json
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class ClaudeClient:
    """Manages Claude API interactions for the trading bot.

    Wraps anthropic.Anthropic client with lazy init, retry, and
    structured output parsing. Returns None gracefully if package
    not installed or API key missing.
    """

    def __init__(self, config: dict):
        self.model = config.get("model", "claude-sonnet-4-5-20250929")
        self.max_tokens = config.get("max_tokens", 2000)
        self.temperature = config.get("temperature", 0.3)
        self.timeout = config.get("timeout_seconds", 30)
        self._api_key = config.get("api_key") or None
        self._client = None
        self._initialized = False

    def _get_client(self):
        """Lazy-init the Anthropic client."""
        if self._initialized:
            return self._client
        self._initialized = True
        try:
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=self._api_key,
                timeout=self.timeout,
            )
        except ImportError:
            logger.warning("anthropic package not installed, Claude AI features disabled")
            self._client = None
        except Exception:
            logger.exception("Failed to initialize Anthropic client")
            self._client = None
        return self._client

    def is_available(self) -> bool:
        """Check if Claude client is usable."""
        return self._get_client() is not None

    def complete(self, system: str, user: str) -> Optional[str]:
        """Send a completion request to Claude.

        Returns response text, or None if unavailable/error.
        """
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
            logger.exception("Claude API call failed")
            return None

    def complete_json(self, system: str, user: str) -> Optional[dict]:
        """Request structured JSON output from Claude.

        Returns parsed dict, or None on failure.
        """
        text = self.complete(system, user)
        if text is None:
            return None
        try:
            # Try to extract JSON from markdown code block
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            return json.loads(text.strip())
        except (json.JSONDecodeError, IndexError):
            logger.warning("Failed to parse Claude JSON response")
            return None

    def get_provider_name(self) -> str:
        """Return provider name for compatibility with LLMProvider interface."""
        return "anthropic"
