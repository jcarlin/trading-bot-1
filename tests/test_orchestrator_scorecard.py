"""Tests for OrchestratorScorecard."""

import sys
import os
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orchestration.orchestrator_scorecard import OrchestratorScorecard


class TestOrchestratorScorecardGrades(unittest.TestCase):
    """Test letter grade computation."""

    def _make_scorecard(self, accuracy, total=10, details=None):
        auditor = MagicMock()
        correct = int(total * accuracy / 100)
        incorrect = total - correct
        auditor.audit_decisions.return_value = {
            "total_decisions": total,
            "correct": correct,
            "incorrect": incorrect,
            "accuracy_pct": accuracy,
            "details": details or [],
        }
        return OrchestratorScorecard(auditor)

    def test_grade_a(self):
        sc = self._make_scorecard(95.0)
        result = sc.compute_scorecard()
        self.assertEqual(result["grade"], "A")

    def test_grade_b(self):
        sc = self._make_scorecard(85.0)
        result = sc.compute_scorecard()
        self.assertEqual(result["grade"], "B")

    def test_grade_c(self):
        sc = self._make_scorecard(75.0)
        result = sc.compute_scorecard()
        self.assertEqual(result["grade"], "C")

    def test_grade_d(self):
        sc = self._make_scorecard(65.0)
        result = sc.compute_scorecard()
        self.assertEqual(result["grade"], "D")

    def test_grade_f(self):
        sc = self._make_scorecard(40.0)
        result = sc.compute_scorecard()
        self.assertEqual(result["grade"], "F")

    def test_zero_decisions(self):
        auditor = MagicMock()
        auditor.audit_decisions.return_value = {
            "total_decisions": 0, "correct": 0, "incorrect": 0,
            "accuracy_pct": 0.0, "details": [],
        }
        sc = OrchestratorScorecard(auditor)
        result = sc.compute_scorecard()
        self.assertEqual(result["grade"], "F")
        self.assertEqual(result["total_decisions"], 0)


class TestOrchestratorScorecardCalibration(unittest.TestCase):
    """Test confidence calibration."""

    def test_well_calibrated(self):
        details = [
            {"decision": {"confidence": 0.9, "action": "pause"}, "was_correct": True, "pnl_impact": 10},
            {"decision": {"confidence": 0.9, "action": "pause"}, "was_correct": True, "pnl_impact": 5},
            {"decision": {"confidence": 0.5, "action": "adjust"}, "was_correct": True, "pnl_impact": 3},
            {"decision": {"confidence": 0.5, "action": "adjust"}, "was_correct": False, "pnl_impact": -2},
            {"decision": {"confidence": 0.2, "action": "resume"}, "was_correct": False, "pnl_impact": -5},
            {"decision": {"confidence": 0.2, "action": "resume"}, "was_correct": False, "pnl_impact": -3},
        ]
        auditor = MagicMock()
        auditor.audit_decisions.return_value = {
            "total_decisions": 6, "correct": 3, "incorrect": 3,
            "accuracy_pct": 50.0, "details": details,
        }
        sc = OrchestratorScorecard(auditor)
        result = sc.compute_scorecard()
        cal = result["confidence_calibration"]
        self.assertTrue(cal["is_well_calibrated"])
        self.assertEqual(cal["bins"]["high"]["accuracy_pct"], 100.0)

    def test_poorly_calibrated(self):
        details = [
            {"decision": {"confidence": 0.9, "action": "pause"}, "was_correct": False, "pnl_impact": -10},
            {"decision": {"confidence": 0.9, "action": "pause"}, "was_correct": False, "pnl_impact": -5},
            {"decision": {"confidence": 0.5, "action": "adjust"}, "was_correct": True, "pnl_impact": 8},
            {"decision": {"confidence": 0.5, "action": "adjust"}, "was_correct": True, "pnl_impact": 5},
            {"decision": {"confidence": 0.2, "action": "resume"}, "was_correct": True, "pnl_impact": 20},
            {"decision": {"confidence": 0.2, "action": "resume"}, "was_correct": True, "pnl_impact": 15},
        ]
        auditor = MagicMock()
        auditor.audit_decisions.return_value = {
            "total_decisions": 4, "correct": 2, "incorrect": 2,
            "accuracy_pct": 50.0, "details": details,
        }
        sc = OrchestratorScorecard(auditor)
        result = sc.compute_scorecard()
        cal = result["confidence_calibration"]
        self.assertFalse(cal["is_well_calibrated"])


class TestOrchestratorScorecardBreakdown(unittest.TestCase):
    """Test per-action type breakdown."""

    def test_breakdown_by_type(self):
        details = [
            {"decision": {"confidence": 0.8, "action": "pause_strategy"}, "was_correct": True, "pnl_impact": 10},
            {"decision": {"confidence": 0.7, "action": "pause_strategy"}, "was_correct": True, "pnl_impact": 5},
            {"decision": {"confidence": 0.6, "action": "resume_strategy"}, "was_correct": False, "pnl_impact": -8},
            {"decision": {"confidence": 0.5, "action": "adjust_risk"}, "was_correct": True, "pnl_impact": 3},
        ]
        auditor = MagicMock()
        auditor.audit_decisions.return_value = {
            "total_decisions": 4, "correct": 3, "incorrect": 1,
            "accuracy_pct": 75.0, "details": details,
        }
        sc = OrchestratorScorecard(auditor)
        result = sc.compute_scorecard()
        breakdown = result["decision_breakdown"]
        self.assertEqual(breakdown["pause_strategy"]["total"], 2)
        self.assertEqual(breakdown["pause_strategy"]["correct"], 2)
        self.assertEqual(breakdown["pause_strategy"]["accuracy_pct"], 100.0)
        self.assertEqual(breakdown["resume_strategy"]["accuracy_pct"], 0.0)
        self.assertEqual(breakdown["adjust_risk"]["accuracy_pct"], 100.0)


class TestOrchestratorScorecardPnlImpact(unittest.TestCase):
    """Test net PnL impact computation."""

    def test_positive_net_impact(self):
        details = [
            {"decision": {"confidence": 0.8, "action": "pause"}, "was_correct": True, "pnl_impact": 50.0},
            {"decision": {"confidence": 0.6, "action": "resume"}, "was_correct": False, "pnl_impact": -20.0},
        ]
        auditor = MagicMock()
        auditor.audit_decisions.return_value = {
            "total_decisions": 2, "correct": 1, "incorrect": 1,
            "accuracy_pct": 50.0, "details": details,
        }
        sc = OrchestratorScorecard(auditor)
        result = sc.compute_scorecard()
        self.assertEqual(result["net_pnl_impact"], 30.0)

    def test_negative_net_impact(self):
        details = [
            {"decision": {"confidence": 0.5, "action": "pause"}, "was_correct": False, "pnl_impact": -100.0},
        ]
        auditor = MagicMock()
        auditor.audit_decisions.return_value = {
            "total_decisions": 1, "correct": 0, "incorrect": 1,
            "accuracy_pct": 0.0, "details": details,
        }
        sc = OrchestratorScorecard(auditor)
        result = sc.compute_scorecard()
        self.assertEqual(result["net_pnl_impact"], -100.0)


class TestOrchestratorScorecardSummary(unittest.TestCase):
    """Test summary report generation."""

    def test_summary_format(self):
        details = [
            {"decision": {"confidence": 0.8, "action": "pause_strategy"}, "was_correct": True, "pnl_impact": 10},
        ]
        auditor = MagicMock()
        auditor.audit_decisions.return_value = {
            "total_decisions": 1, "correct": 1, "incorrect": 0,
            "accuracy_pct": 100.0, "details": details,
        }
        sc = OrchestratorScorecard(auditor)
        summary = sc.generate_summary()
        self.assertIn("Orchestrator Scorecard", summary)
        self.assertIn("Grade: A", summary)
        self.assertIn("Confidence Calibration", summary)
        self.assertIn("pause_strategy", summary)

    def test_summary_from_provided_scorecard(self):
        sc = OrchestratorScorecard(MagicMock())
        scorecard = {
            "accuracy_pct": 75.0,
            "grade": "C",
            "total_decisions": 4,
            "correct": 3,
            "incorrect": 1,
            "confidence_calibration": {
                "bins": {
                    "low": {"range": "0.0-0.4", "total": 1, "correct": 0, "accuracy_pct": 0.0},
                    "medium": {"range": "0.4-0.7", "total": 1, "correct": 1, "accuracy_pct": 100.0},
                    "high": {"range": "0.7-1.0", "total": 2, "correct": 2, "accuracy_pct": 100.0},
                },
                "is_well_calibrated": True,
            },
            "decision_breakdown": {"pause_strategy": {"total": 2, "correct": 2, "accuracy_pct": 100.0}},
            "net_pnl_impact": 25.0,
            "lookback_days": 30,
        }
        summary = sc.generate_summary(scorecard)
        self.assertIn("Grade: C", summary)
        self.assertIn("$25.00", summary)

    def test_custom_lookback(self):
        auditor = MagicMock()
        auditor.audit_decisions.return_value = {
            "total_decisions": 0, "correct": 0, "incorrect": 0,
            "accuracy_pct": 0.0, "details": [],
        }
        sc = OrchestratorScorecard(auditor, {"lookback_days": 90})
        result = sc.compute_scorecard()
        self.assertEqual(result["lookback_days"], 90)


if __name__ == "__main__":
    unittest.main()
