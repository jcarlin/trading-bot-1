"""Claude-powered natural language report generation."""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

REPORT_SYSTEM_PROMPT = """You are a trading system report writer. Generate clear,
concise, actionable reports from structured trading metrics.

Format: Plain text with clear section headers. Use specific numbers.
Highlight anomalies, trends, and actionable insights.
Keep reports under 1500 words. Use bullet points for key findings."""


class AIReportWriter:
    """Uses Claude to generate natural language reports from metrics."""

    def __init__(self, claude_client):
        self.claude = claude_client

    def generate_weekly_report(self, metrics: dict, health: dict,
                                decisions: list, regime: dict) -> Optional[str]:
        """Generate a Claude-written weekly performance report."""
        if self.claude is None or not self.claude.is_available():
            return None

        context = json.dumps({
            "metrics": metrics,
            "health": health,
            "decisions_count": len(decisions) if decisions else 0,
            "regime": regime,
        }, indent=2, default=str)

        return self.claude.complete(
            REPORT_SYSTEM_PROMPT,
            f"Generate a weekly trading performance report:\n\n{context}")

    def generate_audit_narrative(self, audit_data: dict) -> Optional[str]:
        """Generate narrative explanation of a system audit."""
        if self.claude is None or not self.claude.is_available():
            return None

        context = json.dumps(audit_data, indent=2, default=str)
        return self.claude.complete(
            REPORT_SYSTEM_PROMPT,
            f"Generate an audit narrative:\n\n{context}")

    def generate_wallet_briefing(self, wallet_analysis: dict) -> Optional[str]:
        """Generate a briefing on wallet intelligence findings."""
        if self.claude is None or not self.claude.is_available():
            return None

        context = json.dumps(wallet_analysis, indent=2, default=str)
        return self.claude.complete(
            REPORT_SYSTEM_PROMPT,
            f"Generate a wallet intelligence briefing:\n\n{context}")
