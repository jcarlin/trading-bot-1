"""Formats trading events into human-readable notification messages."""

from datetime import datetime, timezone


class NotificationFormatter:
    """Formats trading events into human-readable notification messages."""

    def format_trade(self, fill: dict, strategy_name: str, signal_metadata: dict = None) -> str:
        """Format a trade fill into a notification message."""
        symbol = fill.get("symbol", "???")
        side = fill.get("side", "???")
        qty = fill.get("quantity", 0)
        price = fill.get("price", fill.get("fill_price", 0))
        pnl = fill.get("pnl")

        msg = f"{side.upper()} {symbol} qty={qty} @ {price} [{strategy_name}]"
        if pnl is not None:
            msg += f" PnL={pnl:+.2f}"
        if signal_metadata:
            reason = signal_metadata.get("reason", "")
            if reason:
                msg += f" ({reason})"
        return msg

    def format_ooda_decision(self, decision: dict, checkpoint_type: str) -> str:
        """Format an OODA decision into a notification message."""
        action = decision.get("action", "none")
        strategy = decision.get("strategy_name", "")
        reason = decision.get("reason", "")
        confidence = decision.get("confidence", 0)

        msg = f"OODA [{checkpoint_type}] {action}"
        if strategy:
            msg += f" on {strategy}"
        if reason:
            msg += f": {reason}"
        if confidence:
            msg += f" (confidence={confidence:.0%})"
        return msg

    def format_health_alert(self, strategy_name: str, old_score: float,
                            new_score: float, grade: str) -> str:
        """Format a health score change alert."""
        direction = "improved" if new_score > old_score else "degraded"
        return (f"Health {direction}: {strategy_name} "
                f"{old_score:.0f} -> {new_score:.0f} (grade={grade})")

    def format_risk_event(self, event_type: str, details: dict) -> str:
        """Format a risk event notification."""
        msg = f"Risk event: {event_type}"
        if details:
            extra = ", ".join(f"{k}={v}" for k, v in details.items())
            msg += f" [{extra}]"
        return msg

    def format_escalation(self, trigger: str, context: dict) -> str:
        """Format an escalation notification."""
        msg = f"ESCALATION: {trigger}"
        if context:
            extra = ", ".join(f"{k}={v}" for k, v in context.items())
            msg += f" | {extra}"
        return msg

    def format_daily_summary(self, metrics: dict, health: dict, regime: dict) -> str:
        """Format a daily summary notification."""
        lines = ["Daily Summary"]
        lines.append("---")

        # Metrics
        pnl = metrics.get("total_pnl", 0)
        sharpe = metrics.get("sharpe_ratio", 0)
        trades = metrics.get("trade_count", 0)
        lines.append(f"PnL: {pnl:+.2f} | Sharpe: {sharpe:.2f} | Trades: {trades}")

        # Health
        score = health.get("health_score", 0)
        grade = health.get("grade", "N/A")
        lines.append(f"Health: {score:.0f}/100 ({grade})")

        # Regime
        r = regime.get("regime", "unknown")
        conf = regime.get("confidence", 0)
        lines.append(f"Regime: {r} (confidence={conf:.0%})")

        return "\n".join(lines)
