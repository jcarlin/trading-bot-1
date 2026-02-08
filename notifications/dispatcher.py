"""Routes notifications to channels based on level and type."""

import hashlib
import logging
import time

logger = logging.getLogger(__name__)

LEVEL_PRIORITY = {
    "info": 0,
    "trade": 1,
    "warning": 2,
    "critical": 3,
}


class NotificationDispatcher:
    """Routes notifications to channels based on level/type.

    Supports multiple channels, per-channel level filtering, and deduplication.
    """

    def __init__(self, config: dict = None):
        config = config or {}
        self.dedup_window_s = int(config.get("dedup_window_s", 60))

        # List of (channel, min_level_priority)
        self._channels: list = []
        # hash -> timestamp for dedup
        self._recent_hashes: dict = {}

    async def dispatch(self, message: str, level: str, event_type: str = "",
                       metadata: dict = None) -> None:
        """Route a notification to all eligible channels."""
        msg_hash = self._hash_message(message, level)

        if self._should_deduplicate(msg_hash):
            logger.debug("Deduplicating notification: %s", message[:80])
            return

        self._recent_hashes[msg_hash] = time.monotonic()

        level_pri = LEVEL_PRIORITY.get(level, 0)

        for channel, min_pri in self._channels:
            if level_pri < min_pri:
                continue
            try:
                if channel.is_available():
                    await channel.send(message, level=level, metadata=metadata)
            except Exception:
                logger.exception("Failed to dispatch to channel %s", type(channel).__name__)

    def add_channel(self, channel, min_level: str = "info") -> None:
        """Register a notification channel with a minimum level filter."""
        min_pri = LEVEL_PRIORITY.get(min_level, 0)
        self._channels.append((channel, min_pri))

    def _should_deduplicate(self, message_hash: str) -> bool:
        """Check if this message was sent recently."""
        now = time.monotonic()

        # Purge expired entries
        expired = [h for h, ts in self._recent_hashes.items()
                   if now - ts > self.dedup_window_s]
        for h in expired:
            del self._recent_hashes[h]

        return message_hash in self._recent_hashes

    @staticmethod
    def _hash_message(message: str, level: str) -> str:
        """Create a hash for deduplication."""
        content = f"{level}:{message}"
        return hashlib.md5(content.encode()).hexdigest()
