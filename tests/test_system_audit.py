"""Tests for SystemAuditor and PerformanceAttribution."""

import sys
import os
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from audit.system_auditor import SystemAuditor
from audit.performance_attribution import PerformanceAttribution


class TestSystemAuditorSetup(unittest.TestCase):
    """Test SystemAuditor initialization."""

    def test_default_config(self):
        auditor = SystemAuditor(MagicMock())
        self.assertEqual(auditor.lookback_days, 30)

    def test_custom_config(self):
        auditor = SystemAuditor(MagicMock(), config={"lookback_days": 90})
        self.assertEqual(auditor.lookback_days, 90)


class TestSystemAuditorGenerate(unittest.TestCase):
    """Test audit generation."""

    def test_empty_audit(self):
        auditor = SystemAuditor(MagicMock())
        result = auditor.generate_audit()
        self.assertIn("generated_at", result)
        self.assertIn("recommendations", result)
        self.assertEqual(result["lookback_days"], 30)

    def test_with_decision_auditor(self):
        dec_auditor = MagicMock()
        dec_auditor.audit_decisions.return_value = {
            "total_decisions": 5, "correct": 4, "incorrect": 1,
            "accuracy_pct": 80.0, "details": [],
        }
        auditor = SystemAuditor(MagicMock(), decision_auditor=dec_auditor)
        result = auditor.generate_audit()
        self.assertEqual(result["decision_audit"]["total_decisions"], 5)

    def test_with_scorecard(self):
        scorecard = MagicMock()
        scorecard.compute_scorecard.return_value = {
            "accuracy_pct": 90.0, "grade": "A",
            "total_decisions": 10, "correct": 9, "incorrect": 1,
            "confidence_calibration": {"is_well_calibrated": True},
            "net_pnl_impact": 50.0,
        }
        auditor = SystemAuditor(MagicMock(), orchestrator_scorecard=scorecard)
        result = auditor.generate_audit()
        self.assertEqual(result["scorecard"]["grade"], "A")

    def test_with_both_components(self):
        dec_auditor = MagicMock()
        dec_auditor.audit_decisions.return_value = {
            "total_decisions": 3, "correct": 2, "incorrect": 1,
            "accuracy_pct": 66.7, "details": [],
        }
        scorecard = MagicMock()
        scorecard.compute_scorecard.return_value = {
            "accuracy_pct": 66.7, "grade": "D",
            "total_decisions": 3,
            "confidence_calibration": {"is_well_calibrated": False},
            "net_pnl_impact": -10.0,
        }
        auditor = SystemAuditor(
            MagicMock(), decision_auditor=dec_auditor,
            orchestrator_scorecard=scorecard)
        result = auditor.generate_audit()
        self.assertIn("WARNING", result["recommendations"][0])


class TestSystemAuditorRecommendations(unittest.TestCase):
    """Test recommendation generation."""

    def test_critical_accuracy(self):
        auditor = SystemAuditor(MagicMock())
        audit = {"scorecard": {"accuracy_pct": 30, "grade": "F",
                               "confidence_calibration": {"is_well_calibrated": True}},
                 "decision_audit": {"details": []}}
        recs = auditor._generate_recommendations(audit)
        self.assertTrue(any("CRITICAL" in r for r in recs))

    def test_poor_calibration(self):
        auditor = SystemAuditor(MagicMock())
        audit = {"scorecard": {"accuracy_pct": 80, "grade": "B",
                               "confidence_calibration": {"is_well_calibrated": False}},
                 "decision_audit": {"details": []}}
        recs = auditor._generate_recommendations(audit)
        self.assertTrue(any("calibration" in r.lower() for r in recs))

    def test_negative_pnl_impact(self):
        auditor = SystemAuditor(MagicMock())
        audit = {"scorecard": {"accuracy_pct": 80, "grade": "B",
                               "confidence_calibration": {"is_well_calibrated": True}},
                 "decision_audit": {"details": [
                     {"pnl_impact": -100.0},
                 ]}}
        recs = auditor._generate_recommendations(audit)
        self.assertTrue(any("destroying" in r.lower() for r in recs))

    def test_healthy_system(self):
        auditor = SystemAuditor(MagicMock())
        audit = {"scorecard": {"accuracy_pct": 90, "grade": "A",
                               "confidence_calibration": {"is_well_calibrated": True}},
                 "decision_audit": {"details": []}}
        recs = auditor._generate_recommendations(audit)
        self.assertTrue(any("normal" in r.lower() for r in recs))


class TestSystemAuditorExport(unittest.TestCase):
    """Test decision history export."""

    def test_export_empty(self):
        ts = MagicMock()
        ts.query_decisions.return_value = []
        auditor = SystemAuditor(ts)
        result = auditor.export_decision_history()
        self.assertEqual(result, [])

    def test_export_with_type_error(self):
        ts = MagicMock()
        ts.query_decisions.side_effect = TypeError("missing arg")
        auditor = SystemAuditor(ts)
        result = auditor.export_decision_history()
        self.assertEqual(result, [])


class TestSystemAuditorReport(unittest.TestCase):
    """Test audit report generation."""

    def test_report_format(self):
        auditor = SystemAuditor(MagicMock())
        audit = {
            "generated_at": "2025-01-01T00:00:00",
            "lookback_days": 30,
            "scorecard": {"grade": "B", "accuracy_pct": 80.0,
                          "total_decisions": 5, "net_pnl_impact": 25.0},
            "decision_audit": {"total_decisions": 5, "correct": 4, "incorrect": 1},
            "system_events_summary": {"total_events": 100,
                                       "by_type": {"market_regime": 50}},
            "recommendations": ["System healthy."],
        }
        report = auditor.generate_audit_report(audit)
        self.assertIn("SYSTEM AUDIT REPORT", report)
        self.assertIn("Grade: B", report)
        self.assertIn("market_regime: 50", report)


class TestPerformanceAttributionSetup(unittest.TestCase):
    """Test PerformanceAttribution initialization."""

    def test_default_config(self):
        pa = PerformanceAttribution(MagicMock(), ["a", "b"])
        self.assertEqual(pa.lookback_hours, 720)

    def test_custom_config(self):
        pa = PerformanceAttribution(MagicMock(), ["a"], {"lookback_hours": 168})
        self.assertEqual(pa.lookback_hours, 168)


class TestPerformanceAttributionCompute(unittest.TestCase):
    """Test attribution computation."""

    def test_empty_fills(self):
        ts = MagicMock()
        ts.query_fills_by_strategy.return_value = []
        pa = PerformanceAttribution(ts, ["strat_a", "strat_b"])
        result = pa.compute_attribution()
        self.assertEqual(result["total_pnl"], 0.0)
        self.assertIn("strat_a", result["strategy_contributions"])

    def test_single_strategy_with_fills(self):
        ts = MagicMock()
        ts.query_fills_by_strategy.return_value = [
            {"realized_pnl": 10.0},
            {"realized_pnl": -3.0},
            {"realized_pnl": 5.0},
        ]
        pa = PerformanceAttribution(ts, ["strat_a"])
        result = pa.compute_attribution()
        self.assertAlmostEqual(result["total_pnl"], 12.0, places=2)
        self.assertEqual(result["best_performer"], "strat_a")
        self.assertEqual(result["worst_performer"], "strat_a")

    def test_multiple_strategies(self):
        ts = MagicMock()

        def fills_side_effect(name, start, end):
            if name == "winner":
                return [{"realized_pnl": 50.0}]
            elif name == "loser":
                return [{"realized_pnl": -20.0}]
            return []

        ts.query_fills_by_strategy.side_effect = fills_side_effect
        pa = PerformanceAttribution(ts, ["winner", "loser"])
        result = pa.compute_attribution()
        self.assertAlmostEqual(result["total_pnl"], 30.0, places=2)
        self.assertEqual(result["best_performer"], "winner")
        self.assertEqual(result["worst_performer"], "loser")

        # Check percentage contributions
        contributions = result["strategy_contributions"]
        self.assertGreater(contributions["winner"]["pnl_pct"], 0)
        self.assertLess(contributions["loser"]["pnl_pct"], 0)

    def test_drawdown_computation(self):
        ts = MagicMock()
        ts.query_fills_by_strategy.return_value = [
            {"realized_pnl": 10.0},
            {"realized_pnl": -15.0},
            {"realized_pnl": 5.0},
        ]
        pa = PerformanceAttribution(ts, ["strat_a"])
        result = pa.compute_attribution()
        # Peak=10, then drops to -5, DD=15
        dd = result["strategy_contributions"]["strat_a"]["max_drawdown"]
        self.assertAlmostEqual(dd, 15.0, places=2)

    def test_query_failure_handled(self):
        ts = MagicMock()
        ts.query_fills_by_strategy.side_effect = Exception("DB error")
        pa = PerformanceAttribution(ts, ["strat_a"])
        result = pa.compute_attribution()
        self.assertEqual(result["total_pnl"], 0.0)


class TestPerformanceAttributionReport(unittest.TestCase):
    """Test attribution report generation."""

    def test_report_format(self):
        ts = MagicMock()
        ts.query_fills_by_strategy.return_value = [
            {"realized_pnl": 25.0},
        ]
        pa = PerformanceAttribution(ts, ["strat_a"])
        report = pa.generate_attribution_report()
        self.assertIn("Performance Attribution", report)
        self.assertIn("strat_a", report)
        self.assertIn("25.0", report)

    def test_report_from_provided_attribution(self):
        pa = PerformanceAttribution(MagicMock(), ["a"])
        attribution = {
            "total_pnl": 100.0,
            "lookback_hours": 168,
            "best_performer": "a",
            "worst_performer": "a",
            "strategy_contributions": {
                "a": {"pnl": 100.0, "pnl_pct": 100.0,
                      "trade_count": 5, "max_drawdown": 10.0}
            },
        }
        report = pa.generate_attribution_report(attribution)
        self.assertIn("$100.0", report)


if __name__ == "__main__":
    unittest.main()
