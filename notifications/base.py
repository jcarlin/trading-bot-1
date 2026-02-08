"""Abstract notification channel interface."""

from abc import ABC, abstractmethod


class NotificationChannel(ABC):
    """Abstract notification channel."""

    @abstractmethod
    async def send(self, message: str, level: str = "info", metadata: dict = None) -> bool:
        """Send notification.

        Args:
            message: The notification text.
            level: One of info/warning/critical/trade.
            metadata: Optional extra data.

        Returns:
            True if sent successfully.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether this channel is configured and ready."""
        ...
