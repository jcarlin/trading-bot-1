"""System-level audit: exports decision history, generates full audit reports."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class SystemAuditor:
    """Generates system audit reports and exports decision history.

    Combines decision auditor results, orchestrator scorecard,
    and strategy performance into a comprehensive audit.
    """

    def __init__(self, timescale, decision_auditor=None,
                 orchestrator_scorecard=None, config: dict = None):
        self.timescale = timescale
        self.decision_auditor = decision_auditor
        self.orchestrator_scorecard = orchestrator_scorecard
        config = config or {}
        self.lookback_days = config.get("lookback_days", 30)

    def generate_audit(self, lookback_days: Optional[int] = None) -> dict:
        """Generate a full system audit.

        Returns:
            Dict with decision_audit, scorecard, strategy_performance,
            system_events, and recommendations.
        """
        lookback = lookback_days or self.lookback_days
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback)

        audit = {
            "generated_at": end.isoformat(),
            "lookback_days": lookback,
            "decision_audit": {},
            "scorecard": {},
            "system_events_summary": {},
            "recommendations": [],
        }

        # Decision audit
        if self.decision_auditor:
            try:
                audit["decision_audit"] = self.decision_auditor.audit_decisions(
                    lookback_days=lookback)
            except Exception:
                logger.debug("Failed to run decision audit")

        # Orchestrator scorecard
        if self.orchestrator_scorecard:
            try:
                audit["scorecard"] = self.orchestrator_scorecard.compute_scorecard(
                    lookback_days=lookback)
            except Exception:
                logger.debug("Failed to compute scorecard")

        # System events summary
        try:
            events = self._summarize_system_events(start, end)
            audit["system_events_summary"] = events
        except Exception:
            logger.debug("Failed to summarize system events")

        # Recommendations
        audit["recommendations"] = self._generate_recommendations(audit)

        return audit

    def export_decision_history(self, lookback_days: Optional[int] = None) -> list[dict]:
        """Export raw decision history from TimescaleDB.

        Returns:
            List of decision dicts ordered by time.
        """
        lookback = lookback_days or self.lookback_days
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback)

        try:
            # Query all decision types
            decisions = self._query_all_decisions(start, end)
            return decisions
        except Exception:
            logger.exception("Failed to export decision history")
            return []

    def _query_all_decisions(self, start: datetime, end: datetime) -> list[dict]:
        """Query all decisions across all strategies."""
        try:
            return self.timescale.query_decisions(start, end)
        except TypeError:
            # Fallback: some implementations require strategy_name
            return []

    def _summarize_system_events(self, start: datetime, end: datetime) -> dict:
        """Summarize system events by type."""
        summary = {
            "total_events": 0,
            "by_type": {},
            "by_severity": {},
        }

        try:
            # Query system events by each known type
            event_types = [
                "market_regime", "health_score", "report",
                "correlation_snapshot", "wallet_score", "wallet_analysis",
            ]

            for evt_type in event_types:
                try:
                    events = self.timescale.query_system_events_by_type(
                        evt_type, start, end)
                    count = len(events) if events else 0
                    summary["by_type"][evt_type] = count
                    summary["total_events"] += count
                except Exception:
                    # Method may not exist yet, skip
                    pass

        except Exception:
            logger.debug("Failed to query system events")

        return summary

    def _generate_recommendations(self, audit: dict) -> list[str]:
        """Generate audit recommendations based on findings."""
        recs = []

        # Scorecard-based recommendations
        scorecard = audit.get("scorecard", {})
        accuracy = scorecard.get("accuracy_pct", 0)
        grade = scorecard.get("grade", "F")

        if accuracy < 50:
            recs.append(
                f"CRITICAL: Orchestrator accuracy is {accuracy:.0f}% (grade={grade}). "
                "Review decision rules and thresholds.")
        elif accuracy < 70:
            recs.append(
                f"WARNING: Orchestrator accuracy is {accuracy:.0f}%. "
                "Consider tuning decision confidence thresholds.")

        # Calibration-based
        cal = scorecard.get("confidence_calibration", {})
        if not cal.get("is_well_calibrated", True):
            recs.append(
                "Confidence calibration is poor. High-confidence decisions "
                "are not more accurate than low-confidence ones.")

        # Decision audit based
        decision_audit = audit.get("decision_audit", {})
        net_pnl = sum(
            d.get("pnl_impact", 0)
            for d in decision_audit.get("details", [])
        )
        if net_pnl < -50:
            recs.append(
                f"Net PnL impact of decisions is ${net_pnl:.2f}. "
                "Decision-making is destroying value.")

        if not recs:
            recs.append("System operating within normal parameters.")

        return recs

    def generate_audit_report(self, audit: Optional[dict] = None) -> str:
        """Generate a human-readable audit report."""
        if audit is None:
            audit = self.generate_audit()

        lines = [
            "=" * 60,
            "SYSTEM AUDIT REPORT",
            "=" * 60,
            f"Generated: {audit.get('generated_at', 'N/A')}",
            f"Lookback: {audit.get('lookback_days', 0)} days",
            "",
        ]

        # Scorecard section
        scorecard = audit.get("scorecard", {})
        lines.append("--- ORCHESTRATOR SCORECARD ---")
        lines.append(f"Grade: {scorecard.get('grade', 'N/A')} "
                     f"({scorecard.get('accuracy_pct', 0):.1f}% accuracy)")
        lines.append(f"Decisions: {scorecard.get('total_decisions', 0)} total")
        lines.append(f"Net PnL Impact: ${scorecard.get('net_pnl_impact', 0):.2f}")
        lines.append("")

        # Decision audit section
        decision_audit = audit.get("decision_audit", {})
        lines.append("--- DECISION AUDIT ---")
        lines.append(f"Total audited: {decision_audit.get('total_decisions', 0)}")
        lines.append(f"Correct: {decision_audit.get('correct', 0)}")
        lines.append(f"Incorrect: {decision_audit.get('incorrect', 0)}")
        lines.append("")

        # System events
        events = audit.get("system_events_summary", {})
        lines.append("--- SYSTEM EVENTS ---")
        lines.append(f"Total: {events.get('total_events', 0)}")
        for evt_type, count in events.get("by_type", {}).items():
            lines.append(f"  {evt_type}: {count}")
        lines.append("")

        # Recommendations
        lines.append("--- RECOMMENDATIONS ---")
        for rec in audit.get("recommendations", []):
            lines.append(f"  - {rec}")

        return "\n".join(lines)
