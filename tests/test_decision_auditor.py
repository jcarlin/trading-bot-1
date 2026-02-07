"""Tests for DecisionAuditor."""

import sys
import os
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orchestration.decision_auditor import DecisionAuditor


class TestDecisionAuditorSetup(unittest.TestCase):
    """Test auditor initialization."""

    def test_default_config(self):
        auditor = DecisionAuditor(MagicMock(), MagicMock())
        self.assertEqual(auditor.lookback_days, 30)
        self.assertEqual(auditor.evaluation_window_hours, 24)

    def test_custom_config(self):
        auditor = DecisionAuditor(MagicMock(), MagicMock(), {
            "lookback_days": 60,
            "evaluation_window_hours": 48,
        })
        self.assertEqual(auditor.lookback_days, 60)
        self.assertEqual(auditor.evaluation_window_hours, 48)


class TestDecisionAuditorNoDecisions(unittest.TestCase):
    """Test audit with no decisions."""

    def test_empty_decisions(self):
        ts = MagicMock()
        ts.query_decisions.return_value = []
        auditor = DecisionAuditor(ts, MagicMock())
        result = auditor.audit_decisions()
        self.assertEqual(result["total_decisions"], 0)
        self.assertEqual(result["accuracy_pct"], 0.0)

    def test_none_decisions(self):
        ts = MagicMock()
        ts.query_decisions.return_value = None
        auditor = DecisionAuditor(ts, MagicMock())
        result = auditor.audit_decisions()
        self.assertEqual(result["total_decisions"], 0)

    def test_query_failure(self):
        ts = MagicMock()
        ts.query_decisions.side_effect = Exception("DB error")
        auditor = DecisionAuditor(ts, MagicMock())
        result = auditor.audit_decisions()
        self.assertEqual(result["total_decisions"], 0)


class TestDecisionAuditorPauseEvaluation(unittest.TestCase):
    """Test evaluation of pause decisions."""

    def _make_auditor(self, fills_after=None, fills_before=None):
        ts = MagicMock()
        ts.query_decisions.return_value = [{
            "time": datetime.now(timezone.utc) - timedelta(days=1),
            "action": {"action": "pause_strategy"},
            "strategy": "test_strat",
            "hypothesis": "Will prevent losses",
            "confidence": 0.8,
            "context": {},
            "outcome": {},
        }]

        call_count = [0]
        def query_fills_side_effect(strategy, start, end):
            call_count[0] += 1
            if call_count[0] == 1:
                return fills_after or []
            return fills_before or []

        ts.query_fills_by_strategy.side_effect = query_fills_side_effect
        return DecisionAuditor(ts, MagicMock())

    def test_pause_correct_when_would_have_lost(self):
        # Counterfactual shows losses (pre-decision was losing)
        auditor = self._make_auditor(
            fills_after=[],
            fills_before=[{"realized_pnl": -50.0}],
        )
        result = auditor.audit_decisions()
        detail = result["details"][0]
        self.assertTrue(detail["was_correct"])

    def test_pause_incorrect_when_would_have_profited(self):
        # Counterfactual shows profits (pre-decision was winning)
        auditor = self._make_auditor(
            fills_after=[],
            fills_before=[{"realized_pnl": 100.0}],
        )
        result = auditor.audit_decisions()
        detail = result["details"][0]
        self.assertFalse(detail["was_correct"])


class TestDecisionAuditorResumeEvaluation(unittest.TestCase):
    """Test evaluation of resume decisions."""

    def _make_auditor(self, fills_after):
        ts = MagicMock()
        ts.query_decisions.return_value = [{
            "time": datetime.now(timezone.utc) - timedelta(days=1),
            "action": {"action": "resume_strategy"},
            "strategy": "test_strat",
            "hypothesis": "Strategy will recover",
            "confidence": 0.7,
            "context": {},
            "outcome": {},
        }]
        ts.query_fills_by_strategy.return_value = fills_after
        return DecisionAuditor(ts, MagicMock())

    def test_resume_correct_when_profitable(self):
        auditor = self._make_auditor(
            fills_after=[{"realized_pnl": 25.0}, {"realized_pnl": 15.0}],
        )
        result = auditor.audit_decisions()
        self.assertTrue(result["details"][0]["was_correct"])
        self.assertEqual(result["accuracy_pct"], 100.0)

    def test_resume_incorrect_when_losing(self):
        auditor = self._make_auditor(
            fills_after=[{"realized_pnl": -30.0}],
        )
        result = auditor.audit_decisions()
        self.assertFalse(result["details"][0]["was_correct"])


class TestDecisionAuditorNoAction(unittest.TestCase):
    """Test evaluation of no-action decisions."""

    def _make_auditor(self, fills_after):
        ts = MagicMock()
        ts.query_decisions.return_value = [{
            "time": datetime.now(timezone.utc) - timedelta(days=1),
            "action": {"action": "none"},
            "strategy": "test_strat",
            "hypothesis": "No action needed",
            "confidence": 0.5,
            "context": {},
            "outcome": {},
        }]
        ts.query_fills_by_strategy.return_value = fills_after
        return DecisionAuditor(ts, MagicMock())

    def test_no_action_correct_when_no_drawdown(self):
        auditor = self._make_auditor(
            fills_after=[{"realized_pnl": 10.0}],
        )
        result = auditor.audit_decisions()
        self.assertTrue(result["details"][0]["was_correct"])

    def test_no_action_incorrect_when_drawdown(self):
        # Large drawdown means action was needed
        auditor = self._make_auditor(
            fills_after=[{"realized_pnl": -50.0}, {"realized_pnl": -30.0}],
        )
        result = auditor.audit_decisions()
        detail = result["details"][0]
        # max_dd is computed from fills; with -80 total and peak=0, dd=80 > 5%
        self.assertFalse(detail["was_correct"])


class TestDecisionAuditorAdjustment(unittest.TestCase):
    """Test evaluation of risk/allocation adjustment decisions."""

    def _make_auditor(self, fills_after):
        ts = MagicMock()
        ts.query_decisions.return_value = [{
            "time": datetime.now(timezone.utc) - timedelta(days=1),
            "action": {"action": "adjust_risk"},
            "strategy": "test_strat",
            "hypothesis": "Tighter risk will help",
            "confidence": 0.6,
            "context": {},
            "outcome": {},
        }]
        ts.query_fills_by_strategy.return_value = fills_after
        return DecisionAuditor(ts, MagicMock())

    def test_adjustment_correct_when_risk_contained(self):
        auditor = self._make_auditor(
            fills_after=[{"realized_pnl": 5.0}],
        )
        result = auditor.audit_decisions()
        self.assertTrue(result["details"][0]["was_correct"])

    def test_adjustment_incorrect_when_large_loss(self):
        auditor = self._make_auditor(
            fills_after=[{"realized_pnl": -20.0}],
        )
        result = auditor.audit_decisions()
        detail = result["details"][0]
        # pnl=-20, dd=20 > 5 => incorrect
        self.assertFalse(detail["was_correct"])


class TestDecisionAuditorSingleAudit(unittest.TestCase):
    """Test auditing a single decision."""

    def test_audit_single(self):
        ts = MagicMock()
        ts.query_fills_by_strategy.return_value = [{"realized_pnl": 10.0}]
        auditor = DecisionAuditor(ts, MagicMock())

        decision = {
            "time": datetime.now(timezone.utc),
            "action": "resume_strategy",
            "strategy": "test_strat",
            "hypothesis": "Should recover",
            "confidence": 0.7,
        }
        result = auditor.audit_single(decision)
        self.assertTrue(result["was_correct"])
        self.assertIn("pnl_impact", result)


class TestDecisionAuditorAccuracy(unittest.TestCase):
    """Test overall accuracy computation."""

    def test_mixed_accuracy(self):
        ts = MagicMock()
        ts.query_decisions.return_value = [
            {
                "time": datetime.now(timezone.utc) - timedelta(days=1),
                "action": {"action": "resume_strategy"},
                "strategy": "s1",
                "hypothesis": "", "confidence": 0.8,
                "context": {}, "outcome": {},
            },
            {
                "time": datetime.now(timezone.utc) - timedelta(days=2),
                "action": {"action": "resume_strategy"},
                "strategy": "s2",
                "hypothesis": "", "confidence": 0.5,
                "context": {}, "outcome": {},
            },
        ]

        call_count = [0]
        def side_effect(strategy, start, end):
            call_count[0] += 1
            if strategy == "s1":
                return [{"realized_pnl": 50.0}]
            return [{"realized_pnl": -30.0}]

        ts.query_fills_by_strategy.side_effect = side_effect

        auditor = DecisionAuditor(ts, MagicMock())
        result = auditor.audit_decisions()
        self.assertEqual(result["total_decisions"], 2)
        self.assertEqual(result["correct"], 1)
        self.assertEqual(result["incorrect"], 1)
        self.assertAlmostEqual(result["accuracy_pct"], 50.0)


if __name__ == "__main__":
    unittest.main()
