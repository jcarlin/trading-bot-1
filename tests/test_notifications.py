"""Tests for the notification system: channels, formatter, dispatcher, escalation."""

import asyncio
import json
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from notifications.base import NotificationChannel
from notifications.telegram_bot import TelegramNotifier
from notifications.formatter import NotificationFormatter
from notifications.dispatcher import NotificationDispatcher
from notifications.escalation import EscalationManager


# ---------------------------------------------------------------------------
# TestNotificationChannel (5 tests)
# ---------------------------------------------------------------------------

class TestNotificationChannel(unittest.TestCase):
    """Tests for the abstract NotificationChannel interface."""

    def test_cannot_instantiate_abstract(self):
        with self.assertRaises(TypeError):
            NotificationChannel()

    def test_subclass_must_implement_send(self):
        class Incomplete(NotificationChannel):
            def is_available(self):
                return True
        with self.assertRaises(TypeError):
            Incomplete()

    def test_subclass_must_implement_is_available(self):
        class Incomplete(NotificationChannel):
            async def send(self, message, level="info", metadata=None):
                return True
        with self.assertRaises(TypeError):
            Incomplete()

    def test_complete_subclass_instantiates(self):
        class Complete(NotificationChannel):
            async def send(self, message, level="info", metadata=None):
                return True
            def is_available(self):
                return True
        c = Complete()
        self.assertTrue(c.is_available())

    def test_complete_subclass_send(self):
        class Complete(NotificationChannel):
            async def send(self, message, level="info", metadata=None):
                return True
            def is_available(self):
                return True
        c = Complete()
        result = asyncio.run(c.send("hello"))
        self.assertTrue(result)


# ---------------------------------------------------------------------------
# TestTelegramNotifier (12 tests)
# ---------------------------------------------------------------------------

class TestTelegramNotifier(unittest.TestCase):
    """Tests for the TelegramNotifier channel."""

    def _make_notifier(self, **overrides):
        cfg = {"bot_token": "test-token", "chat_id": "12345",
               "rate_limit_per_minute": 20}
        cfg.update(overrides)
        return TelegramNotifier(cfg)

    def test_is_available_with_token_and_chat_id(self):
        n = self._make_notifier()
        self.assertTrue(n.is_available())

    def test_not_available_without_token(self):
        n = self._make_notifier(bot_token="")
        self.assertFalse(n.is_available())

    def test_not_available_without_chat_id(self):
        n = self._make_notifier(chat_id="")
        self.assertFalse(n.is_available())

    @patch("notifications.telegram_bot.urllib.request.urlopen")
    def test_send_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        n = self._make_notifier()
        result = asyncio.run(n.send("test message"))
        self.assertTrue(result)
        mock_urlopen.assert_called_once()

    @patch("notifications.telegram_bot.urllib.request.urlopen")
    def test_send_failure_http_error(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("fail")

        n = self._make_notifier()
        result = asyncio.run(n.send("test message"))
        self.assertFalse(result)

    @patch("notifications.telegram_bot.urllib.request.urlopen")
    def test_send_failure_bad_status(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 403
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        n = self._make_notifier()
        result = asyncio.run(n.send("test message"))
        self.assertFalse(result)

    @patch("notifications.telegram_bot.urllib.request.urlopen")
    def test_rate_limiting(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        n = self._make_notifier(rate_limit_per_minute=3)
        # Send 3 successfully
        for i in range(3):
            asyncio.run(n.send(f"msg {i}"))
        # 4th should be rate-limited
        result = asyncio.run(n.send("msg 4"))
        self.assertFalse(result)

    def test_format_message_info_level(self):
        n = self._make_notifier()
        msg = n._format_message("hello", "info", None)
        self.assertIn("[INFO]", msg)
        self.assertIn("hello", msg)

    def test_format_message_critical_level(self):
        n = self._make_notifier()
        msg = n._format_message("alert", "critical", None)
        self.assertIn("[CRITICAL]", msg)

    def test_format_message_with_metadata(self):
        n = self._make_notifier()
        msg = n._format_message("test", "info", {"key": "val"})
        self.assertIn("key", msg)
        self.assertIn("val", msg)

    def test_format_message_trade_level(self):
        n = self._make_notifier()
        msg = n._format_message("fill", "trade", None)
        self.assertIn("[TRADE]", msg)

    def test_config_parsing(self):
        n = self._make_notifier(rate_limit_per_minute="30", message_format="html")
        self.assertEqual(n.rate_limit_per_minute, 30)
        self.assertEqual(n.message_format, "html")


# ---------------------------------------------------------------------------
# TestNotificationFormatter (15 tests)
# ---------------------------------------------------------------------------

class TestNotificationFormatter(unittest.TestCase):
    """Tests for the NotificationFormatter."""

    def setUp(self):
        self.fmt = NotificationFormatter()

    def test_format_trade_long_entry(self):
        fill = {"symbol": "BTC/USDC", "side": "buy", "quantity": 0.1, "price": 50000}
        msg = self.fmt.format_trade(fill, "funding_rate_arb")
        self.assertIn("BUY", msg)
        self.assertIn("BTC/USDC", msg)
        self.assertIn("funding_rate_arb", msg)

    def test_format_trade_short_entry(self):
        fill = {"symbol": "ETH/USDC", "side": "sell", "quantity": 1.0, "price": 3000}
        msg = self.fmt.format_trade(fill, "momentum")
        self.assertIn("SELL", msg)
        self.assertIn("ETH/USDC", msg)

    def test_format_trade_exit_with_pnl(self):
        fill = {"symbol": "BTC/USDC", "side": "sell", "quantity": 0.1,
                "price": 51000, "pnl": 100.0}
        msg = self.fmt.format_trade(fill, "arb")
        self.assertIn("+100.00", msg)

    def test_format_trade_with_signal_metadata(self):
        fill = {"symbol": "BTC/USDC", "side": "buy", "quantity": 0.1, "price": 50000}
        meta = {"reason": "funding rate negative"}
        msg = self.fmt.format_trade(fill, "arb", meta)
        self.assertIn("funding rate negative", msg)

    def test_format_ooda_decision_with_confidence(self):
        decision = {"action": "pause_strategy", "strategy_name": "momentum",
                     "reason": "health low", "confidence": 0.85}
        msg = self.fmt.format_ooda_decision(decision, "hourly")
        self.assertIn("pause_strategy", msg)
        self.assertIn("momentum", msg)
        self.assertIn("85%", msg)

    def test_format_ooda_decision_no_confidence(self):
        decision = {"action": "adjust_risk", "reason": "volatile regime"}
        msg = self.fmt.format_ooda_decision(decision, "daily")
        self.assertIn("adjust_risk", msg)
        self.assertIn("daily", msg)

    def test_format_health_alert_degraded(self):
        msg = self.fmt.format_health_alert("momentum", 80, 45, "D")
        self.assertIn("degraded", msg)
        self.assertIn("momentum", msg)
        self.assertIn("D", msg)

    def test_format_health_alert_improved(self):
        msg = self.fmt.format_health_alert("arb", 45, 80, "B")
        self.assertIn("improved", msg)

    def test_format_risk_event_drawdown(self):
        msg = self.fmt.format_risk_event("drawdown_breach", {"drawdown_pct": 12.5})
        self.assertIn("drawdown_breach", msg)
        self.assertIn("12.5", msg)

    def test_format_risk_event_empty_details(self):
        msg = self.fmt.format_risk_event("position_limit", {})
        self.assertIn("position_limit", msg)

    def test_format_escalation(self):
        msg = self.fmt.format_escalation("portfolio_drawdown",
                                          {"drawdown_pct": 11.0, "threshold": 10.0})
        self.assertIn("ESCALATION", msg)
        self.assertIn("portfolio_drawdown", msg)

    def test_format_daily_summary(self):
        metrics = {"total_pnl": 150.5, "sharpe_ratio": 1.2, "trade_count": 12}
        health = {"health_score": 75, "grade": "B"}
        regime = {"regime": "trending_up", "confidence": 0.85}
        msg = self.fmt.format_daily_summary(metrics, health, regime)
        self.assertIn("Daily Summary", msg)
        self.assertIn("150.50", msg)
        self.assertIn("B", msg)
        self.assertIn("trending_up", msg)

    def test_format_trade_missing_fields(self):
        fill = {}
        msg = self.fmt.format_trade(fill, "test")
        self.assertIn("???", msg)

    def test_format_ooda_decision_empty(self):
        msg = self.fmt.format_ooda_decision({}, "weekly")
        self.assertIn("none", msg)

    def test_format_daily_summary_empty_data(self):
        msg = self.fmt.format_daily_summary({}, {}, {})
        self.assertIn("Daily Summary", msg)


# ---------------------------------------------------------------------------
# TestNotificationDispatcher (10 tests)
# ---------------------------------------------------------------------------

class TestNotificationDispatcher(unittest.TestCase):
    """Tests for the NotificationDispatcher."""

    def _make_channel(self, available=True, send_result=True):
        ch = MagicMock()
        ch.is_available.return_value = available
        ch.send = AsyncMock(return_value=send_result)
        return ch

    def test_dispatch_to_single_channel(self):
        d = NotificationDispatcher()
        ch = self._make_channel()
        d.add_channel(ch)
        asyncio.run(d.dispatch("hello", "info"))
        ch.send.assert_called_once()

    def test_dispatch_to_multiple_channels(self):
        d = NotificationDispatcher()
        ch1 = self._make_channel()
        ch2 = self._make_channel()
        d.add_channel(ch1)
        d.add_channel(ch2)
        asyncio.run(d.dispatch("hello", "info"))
        ch1.send.assert_called_once()
        ch2.send.assert_called_once()

    def test_level_filtering_skips_low_priority(self):
        d = NotificationDispatcher()
        ch = self._make_channel()
        d.add_channel(ch, min_level="critical")
        asyncio.run(d.dispatch("hello", "info"))
        ch.send.assert_not_called()

    def test_level_filtering_allows_high_priority(self):
        d = NotificationDispatcher()
        ch = self._make_channel()
        d.add_channel(ch, min_level="warning")
        asyncio.run(d.dispatch("alert", "critical"))
        ch.send.assert_called_once()

    def test_deduplication_within_window(self):
        d = NotificationDispatcher({"dedup_window_s": 60})
        ch = self._make_channel()
        d.add_channel(ch)
        asyncio.run(d.dispatch("same msg", "info"))
        asyncio.run(d.dispatch("same msg", "info"))
        # Only sent once due to dedup
        self.assertEqual(ch.send.call_count, 1)

    def test_dedup_different_messages_pass(self):
        d = NotificationDispatcher({"dedup_window_s": 60})
        ch = self._make_channel()
        d.add_channel(ch)
        asyncio.run(d.dispatch("msg A", "info"))
        asyncio.run(d.dispatch("msg B", "info"))
        self.assertEqual(ch.send.call_count, 2)

    def test_dedup_expired_allows_resend(self):
        d = NotificationDispatcher({"dedup_window_s": 0})  # 0s window = instant expire
        ch = self._make_channel()
        d.add_channel(ch)
        asyncio.run(d.dispatch("same msg", "info"))
        # Force expiry by manipulating timestamps
        for h in list(d._recent_hashes.keys()):
            d._recent_hashes[h] = time.monotonic() - 100
        asyncio.run(d.dispatch("same msg", "info"))
        self.assertEqual(ch.send.call_count, 2)

    def test_empty_channels(self):
        d = NotificationDispatcher()
        # Should not raise
        asyncio.run(d.dispatch("hello", "info"))

    def test_channel_send_failure_graceful(self):
        d = NotificationDispatcher()
        ch = self._make_channel()
        ch.send = AsyncMock(side_effect=Exception("boom"))
        d.add_channel(ch)
        # Should not raise
        asyncio.run(d.dispatch("hello", "info"))

    def test_unavailable_channel_skipped(self):
        d = NotificationDispatcher()
        ch = self._make_channel(available=False)
        d.add_channel(ch)
        asyncio.run(d.dispatch("hello", "info"))
        ch.send.assert_not_called()


# ---------------------------------------------------------------------------
# TestEscalationManager (8 tests)
# ---------------------------------------------------------------------------

class TestEscalationManager(unittest.TestCase):
    """Tests for the EscalationManager."""

    def test_drawdown_above_threshold_triggers(self):
        em = EscalationManager({"drawdown_threshold": 10.0})
        self.assertTrue(em.check_drawdown_escalation(11.0))

    def test_drawdown_below_threshold_no_trigger(self):
        em = EscalationManager({"drawdown_threshold": 10.0})
        self.assertFalse(em.check_drawdown_escalation(8.0))

    def test_strategy_loss_exceeds_3x_triggers(self):
        em = EscalationManager()
        self.assertTrue(em.check_strategy_loss_escalation("test", -300, 90))

    def test_strategy_loss_below_3x_no_trigger(self):
        em = EscalationManager()
        self.assertFalse(em.check_strategy_loss_escalation("test", -100, 90))

    def test_regime_unprecedented_triggers(self):
        em = EscalationManager()
        self.assertTrue(em.check_regime_escalation({"regime": "unprecedented"}))

    def test_regime_normal_no_trigger(self):
        em = EscalationManager()
        self.assertFalse(em.check_regime_escalation({"regime": "trending_up"}))

    def test_exchange_degraded_triggers(self):
        em = EscalationManager()
        self.assertTrue(em.check_exchange_health({"status": "degraded"}))

    def test_exchange_ok_no_trigger(self):
        em = EscalationManager()
        self.assertFalse(em.check_exchange_health({"status": "ok"}))

    def test_wallet_10x_triggers(self):
        em = EscalationManager()
        self.assertTrue(em.check_wallet_activity("0xabc", 1100, 100))

    def test_wallet_normal_no_trigger(self):
        em = EscalationManager()
        self.assertFalse(em.check_wallet_activity("0xabc", 500, 100))

    def test_evaluate_all_processes_all_checks(self):
        dispatcher = MagicMock()
        dispatcher.dispatch = AsyncMock()
        em = EscalationManager({"drawdown_threshold": 10.0}, dispatcher)
        state = {
            "drawdown_pct": 12.0,
            "strategy_losses": {"test": {"session_loss": -400, "avg_daily_pnl": 100}},
            "regime": {"regime": "unprecedented"},
            "api_status": {"status": "degraded"},
            "wallet_alerts": [{"address": "0x1", "position_size": 1100, "normal_size": 100}],
        }
        results = asyncio.run(em.evaluate_all(state))
        # Should have 5 escalations (drawdown, loss, regime, exchange, wallet)
        self.assertEqual(len(results), 5)
        # Dispatcher should have been called for each
        self.assertEqual(dispatcher.dispatch.call_count, 5)

    def test_evaluate_all_no_triggers(self):
        em = EscalationManager({"drawdown_threshold": 10.0})
        state = {
            "drawdown_pct": 2.0,
            "strategy_losses": {},
            "regime": {"regime": "trending_up"},
            "api_status": {"status": "ok"},
            "wallet_alerts": [],
        }
        results = asyncio.run(em.evaluate_all(state))
        self.assertEqual(len(results), 0)

    def test_strategy_loss_zero_avg_pnl_no_trigger(self):
        em = EscalationManager()
        self.assertFalse(em.check_strategy_loss_escalation("test", -100, 0))


if __name__ == "__main__":
    unittest.main()
