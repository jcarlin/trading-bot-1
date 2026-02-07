"""Orchestrator scorecard: grades the quality of OODA decision-making.

Aggregates decision auditor results into accuracy %, letter grades,
and confidence calibration metrics.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class OrchestratorScorecard:
    """Computes a scorecard for the OODA orchestrator's decision quality.

    Evaluates:
    - Overall accuracy (% of correct decisions)
    - Letter grade (A-F)
    - Confidence calibration (do high-confidence decisions perform better?)
    - Decision type breakdown
    - Net PnL impact of all decisions
    """

    GRADE_THRESHOLDS = [
        (90.0, "A"),
        (80.0, "B"),
        (70.0, "C"),
        (60.0, "D"),
    ]

    def __init__(self, decision_auditor, config: dict = None):
        self.auditor = decision_auditor
        config = config or {}
        self.lookback_days = config.get("lookback_days", 30)

    def compute_scorecard(self, lookback_days: Optional[int] = None) -> dict:
        """Compute the full orchestrator scorecard.

        Returns:
            Dict with accuracy, grade, calibration, breakdown, net_pnl_impact.
        """
        lookback = lookback_days or self.lookback_days
        audit = self.auditor.audit_decisions(lookback_days=lookback)

        details = audit.get("details", [])

        accuracy = audit.get("accuracy_pct", 0.0)
        grade = self._compute_grade(accuracy)
        calibration = self._compute_calibration(details)
        breakdown = self._compute_breakdown(details)
        net_pnl = self._compute_net_pnl_impact(details)

        return {
            "accuracy_pct": accuracy,
            "grade": grade,
            "total_decisions": audit.get("total_decisions", 0),
            "correct": audit.get("correct", 0),
            "incorrect": audit.get("incorrect", 0),
            "confidence_calibration": calibration,
            "decision_breakdown": breakdown,
            "net_pnl_impact": round(net_pnl, 2),
            "lookback_days": lookback,
        }

    def _compute_grade(self, accuracy: float) -> str:
        """Map accuracy percentage to letter grade."""
        for threshold, grade in self.GRADE_THRESHOLDS:
            if accuracy >= threshold:
                return grade
        return "F"

    def _compute_calibration(self, details: list[dict]) -> dict:
        """Compute confidence calibration.

        Bins decisions by confidence level and measures accuracy in each bin.
        Well-calibrated = high-confidence decisions are more often correct.
        """
        bins = {
            "low": {"range": "0.0-0.4", "total": 0, "correct": 0},
            "medium": {"range": "0.4-0.7", "total": 0, "correct": 0},
            "high": {"range": "0.7-1.0", "total": 0, "correct": 0},
        }

        for d in details:
            decision = d.get("decision", {})
            confidence = decision.get("confidence", 0.0)
            was_correct = d.get("was_correct", False)

            if confidence < 0.4:
                bin_key = "low"
            elif confidence < 0.7:
                bin_key = "medium"
            else:
                bin_key = "high"

            bins[bin_key]["total"] += 1
            if was_correct:
                bins[bin_key]["correct"] += 1

        # Compute accuracy per bin
        for bin_data in bins.values():
            total = bin_data["total"]
            bin_data["accuracy_pct"] = round(
                bin_data["correct"] / total * 100, 1) if total > 0 else 0.0

        # Well-calibrated if high > medium > low accuracy
        high_acc = bins["high"]["accuracy_pct"]
        med_acc = bins["medium"]["accuracy_pct"]
        low_acc = bins["low"]["accuracy_pct"]

        is_calibrated = high_acc >= med_acc >= low_acc if (
            bins["high"]["total"] > 0 and bins["medium"]["total"] > 0
        ) else True  # Not enough data to evaluate

        return {
            "bins": bins,
            "is_well_calibrated": is_calibrated,
        }

    def _compute_breakdown(self, details: list[dict]) -> dict:
        """Break down accuracy by decision type."""
        by_type = {}

        for d in details:
            decision = d.get("decision", {})
            action = decision.get("action", "unknown")
            was_correct = d.get("was_correct", False)

            if action not in by_type:
                by_type[action] = {"total": 0, "correct": 0}

            by_type[action]["total"] += 1
            if was_correct:
                by_type[action]["correct"] += 1

        # Add accuracy per type
        for data in by_type.values():
            data["accuracy_pct"] = round(
                data["correct"] / data["total"] * 100, 1) if data["total"] > 0 else 0.0

        return by_type

    def _compute_net_pnl_impact(self, details: list[dict]) -> float:
        """Sum up the PnL impact of all decisions."""
        return sum(d.get("pnl_impact", 0.0) for d in details)

    def generate_summary(self, scorecard: Optional[dict] = None) -> str:
        """Generate a human-readable scorecard summary."""
        if scorecard is None:
            scorecard = self.compute_scorecard()

        lines = [
            "=== Orchestrator Scorecard ===",
            f"Grade: {scorecard['grade']} ({scorecard['accuracy_pct']:.1f}% accuracy)",
            f"Decisions: {scorecard['total_decisions']} total, "
            f"{scorecard['correct']} correct, {scorecard['incorrect']} incorrect",
            f"Net PnL Impact: ${scorecard['net_pnl_impact']:.2f}",
            f"Lookback: {scorecard['lookback_days']} days",
            "",
        ]

        # Calibration
        cal = scorecard.get("confidence_calibration", {})
        bins = cal.get("bins", {})
        lines.append("Confidence Calibration:")
        for level, data in bins.items():
            lines.append(f"  {level} ({data['range']}): "
                        f"{data['accuracy_pct']:.0f}% accuracy "
                        f"({data['correct']}/{data['total']})")
        cal_status = "WELL CALIBRATED" if cal.get("is_well_calibrated") else "POORLY CALIBRATED"
        lines.append(f"  Status: {cal_status}")
        lines.append("")

        # Breakdown
        breakdown = scorecard.get("decision_breakdown", {})
        if breakdown:
            lines.append("Decision Type Breakdown:")
            for action, data in sorted(breakdown.items()):
                lines.append(f"  {action}: {data['accuracy_pct']:.0f}% "
                            f"({data['correct']}/{data['total']})")

        return "\n".join(lines)
