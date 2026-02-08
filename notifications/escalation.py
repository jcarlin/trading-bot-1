"""Human escalation triggers from CLAUDE.md Section 5."""

import logging

logger = logging.getLogger(__name__)


class EscalationManager:
    """Human escalation triggers from CLAUDE.md Section 5.

    Triggers:
    - Portfolio drawdown > 10%
    - Strategy loses > 3x avg daily PnL in one session
    - Market regime "unprecedented"
    - Exchange API degradation
    - Tracked wallet position > 10x normal
    """

    def __init__(self, config: dict = None, dispatcher=None):
        config = config or {}
        self.dispatcher = dispatcher

        # Thresholds (configurable)
        self.drawdown_threshold = float(config.get("drawdown_threshold", 10.0))
        self.loss_multiplier = float(config.get("loss_multiplier", 3.0))
        self.wallet_size_multiplier = float(config.get("wallet_size_multiplier", 10.0))

    def check_drawdown_escalation(self, drawdown_pct: float) -> bool:
        """Escalate if portfolio drawdown exceeds threshold."""
        return drawdown_pct > self.drawdown_threshold

    def check_strategy_loss_escalation(self, strategy_name: str,
                                       session_loss: float,
                                       avg_daily_pnl: float) -> bool:
        """Escalate if strategy loses > 3x avg daily PnL in one session."""
        if avg_daily_pnl == 0:
            return False
        return abs(session_loss) > self.loss_multiplier * abs(avg_daily_pnl)

    def check_regime_escalation(self, regime: dict) -> bool:
        """Escalate if market regime is unprecedented."""
        regime_name = regime.get("regime", "")
        return regime_name == "unprecedented"

    def check_exchange_health(self, api_status: dict) -> bool:
        """Escalate if exchange API shows degradation."""
        status = api_status.get("status", "ok")
        return status in ("degraded", "down", "error")

    def check_wallet_activity(self, wallet_address: str,
                              position_size: float,
                              normal_size: float) -> bool:
        """Escalate if tracked wallet takes position > 10x normal."""
        if normal_size <= 0:
            return False
        return position_size > self.wallet_size_multiplier * normal_size

    async def evaluate_all(self, system_state: dict) -> list:
        """Evaluate all escalation triggers against current system state.

        Returns list of triggered escalations.
        """
        escalations = []

        # 1. Drawdown check
        drawdown = system_state.get("drawdown_pct", 0)
        if self.check_drawdown_escalation(drawdown):
            esc = {
                "trigger": "portfolio_drawdown",
                "details": {"drawdown_pct": drawdown,
                            "threshold": self.drawdown_threshold},
            }
            escalations.append(esc)

        # 2. Strategy loss check
        strategy_losses = system_state.get("strategy_losses", {})
        for sname, info in strategy_losses.items():
            session_loss = info.get("session_loss", 0)
            avg_pnl = info.get("avg_daily_pnl", 0)
            if self.check_strategy_loss_escalation(sname, session_loss, avg_pnl):
                esc = {
                    "trigger": "strategy_loss",
                    "details": {"strategy": sname,
                                "session_loss": session_loss,
                                "avg_daily_pnl": avg_pnl},
                }
                escalations.append(esc)

        # 3. Regime check
        regime = system_state.get("regime", {})
        if self.check_regime_escalation(regime):
            esc = {
                "trigger": "unprecedented_regime",
                "details": regime,
            }
            escalations.append(esc)

        # 4. Exchange health
        api_status = system_state.get("api_status", {})
        if self.check_exchange_health(api_status):
            esc = {
                "trigger": "exchange_degradation",
                "details": api_status,
            }
            escalations.append(esc)

        # 5. Wallet activity
        wallet_alerts = system_state.get("wallet_alerts", [])
        for wa in wallet_alerts:
            addr = wa.get("address", "")
            pos = wa.get("position_size", 0)
            normal = wa.get("normal_size", 0)
            if self.check_wallet_activity(addr, pos, normal):
                esc = {
                    "trigger": "wallet_activity",
                    "details": wa,
                }
                escalations.append(esc)

        # Dispatch escalations if dispatcher available
        if self.dispatcher and escalations:
            for esc in escalations:
                try:
                    await self.dispatcher.dispatch(
                        message=f"ESCALATION: {esc['trigger']} - {esc['details']}",
                        level="critical",
                        event_type="escalation",
                        metadata=esc["details"],
                    )
                except Exception:
                    logger.exception("Failed to dispatch escalation")

        return escalations
