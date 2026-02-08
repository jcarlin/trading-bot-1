"""Telegram Bot API notification channel."""

import json
import logging
import time
import urllib.request
import urllib.error
from collections import deque

from .base import NotificationChannel

logger = logging.getLogger(__name__)

LEVEL_PREFIXES = {
    "info": "INFO",
    "warning": "WARNING",
    "critical": "CRITICAL",
    "trade": "TRADE",
}


class TelegramNotifier(NotificationChannel):
    """Sends notifications via Telegram Bot API.

    Uses urllib (no external deps) to POST to api.telegram.org.
    """

    def __init__(self, config: dict):
        self.bot_token = config.get("bot_token", "")
        self.chat_id = config.get("chat_id", "")
        self.rate_limit_per_minute = int(config.get("rate_limit_per_minute", 20))
        self.message_format = config.get("message_format", "markdown")

        # Rate-limit tracking: timestamps of recent sends
        self._send_timestamps: deque = deque()

    async def send(self, message: str, level: str = "info", metadata: dict = None) -> bool:
        """Send a message via Telegram."""
        if not self.is_available():
            logger.debug("TelegramNotifier not available (missing token/chat_id)")
            return False

        if not self._check_rate_limit():
            logger.warning("Telegram rate limit exceeded, dropping message")
            return False

        formatted = self._format_message(message, level, metadata)

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": formatted,
            "parse_mode": "Markdown" if self.message_format == "markdown" else "HTML",
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    self._send_timestamps.append(time.monotonic())
                    return True
                else:
                    logger.warning("Telegram API returned status %d", resp.status)
                    return False
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            logger.warning("Telegram send failed: %s", exc)
            return False

    def is_available(self) -> bool:
        """Available when bot_token and chat_id are configured."""
        return bool(self.bot_token) and bool(self.chat_id)

    def _format_message(self, message: str, level: str, metadata: dict = None) -> str:
        """Format message with level prefix."""
        prefix = LEVEL_PREFIXES.get(level, "INFO")
        text = f"*[{prefix}]* {message}"
        if metadata:
            details = "\n".join(f"  {k}: {v}" for k, v in metadata.items())
            text += f"\n```\n{details}\n```"
        return text

    def _check_rate_limit(self) -> bool:
        """Return True if we haven't exceeded rate_limit_per_minute."""
        now = time.monotonic()
        cutoff = now - 60.0

        # Purge old entries
        while self._send_timestamps and self._send_timestamps[0] < cutoff:
            self._send_timestamps.popleft()

        return len(self._send_timestamps) < self.rate_limit_per_minute
