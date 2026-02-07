"""Decision auditor: evaluates past OODA decisions against actual outcomes.

Pulls decision history from TimescaleDB and computes counterfactual PnL
to determine whether decisions helped or hurt performance.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class DecisionAuditor:
    """Audits past OODA orchestrator decisions.

    For each decision (pause, resume, adjust_risk, adjust_allocation),
    compares actual post-decision performance against a counterfactual
    where no action was taken.
    """

    def __init__(self, timescale, redis_store, config: dict = None):
        self.timescale = timescale
        self.redis_store = redis_store
        config = config or {}
        self.lookback_days = config.get("lookback_days", 30)
        self.evaluation_window_hours = config.get("evaluation_window_hours", 24)

    def audit_decisions(self, lookback_days: Optional[int] = None) -> dict:
        """Audit all OODA decisions within the lookback period.

        Returns:
            Dict with total decisions, correct count, accuracy,
            and per-decision audit details.
        """
        lookback = lookback_days or self.lookback_days
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback)

        decisions = self._fetch_decisions(start, end)

        if not decisions:
            return {
                "total_decisions": 0,
                "correct": 0,
                "incorrect": 0,
                "accuracy_pct": 0.0,
                "details": [],
            }

        audited = []
        correct = 0
        incorrect = 0

        for decision in decisions:
            result = self._evaluate_decision(decision)
            audited.append(result)
            if result["was_correct"]:
                correct += 1
            else:
                incorrect += 1

        total = correct + incorrect
        accuracy = (correct / total * 100) if total > 0 else 0.0

        return {
            "total_decisions": total,
            "correct": correct,
            "incorrect": incorrect,
            "accuracy_pct": round(accuracy, 1),
            "details": audited,
        }

    def audit_single(self, decision: dict) -> dict:
        """Audit a single decision.

        Args:
            decision: Dict with time, action, strategy, hypothesis, confidence.

        Returns:
            Audit result with was_correct, reason, pnl_impact.
        """
        return self._evaluate_decision(decision)

    def _fetch_decisions(self, start: datetime, end: datetime) -> list[dict]:
        """Pull decision history from TimescaleDB."""
        try:
            rows = self.timescale.query_decisions(start, end)
            if not rows:
                return []
            return [self._normalize_decision(r) for r in rows if r]
        except Exception:
            logger.exception("Failed to fetch decisions")
            return []

    def _normalize_decision(self, row: dict) -> dict:
        """Normalize a raw decision row into a standard format."""
        action = row.get("action", {})
        if isinstance(action, dict):
            action_type = action.get("action", "none")
        else:
            action_type = str(action)

        return {
            "time": row.get("time", datetime.now(timezone.utc)),
            "action": action_type,
            "strategy": row.get("strategy", "unknown"),
            "hypothesis": row.get("hypothesis", ""),
            "confidence": row.get("confidence", 0.0),
            "context": row.get("context", {}),
            "outcome": row.get("outcome", {}),
        }

    def _evaluate_decision(self, decision: dict) -> dict:
        """Evaluate whether a decision was correct.

        Compares post-decision performance against the decision's
        stated hypothesis. A decision is correct if:
        - pause_strategy: strategy would have lost money had it continued
        - resume_strategy: strategy made money after resuming
        - adjust_risk/adjust_allocation: strategy improved after adjustment
        - none/no_action: no harmful event occurred
        """
        action = decision.get("action", "none")
        strategy = decision.get("strategy", "unknown")
        decision_time = decision.get("time", datetime.now(timezone.utc))

        if isinstance(decision_time, str):
            try:
                decision_time = datetime.fromisoformat(decision_time)
            except (ValueError, TypeError):
                decision_time = datetime.now(timezone.utc)

        # Get performance after decision
        post_perf = self._get_post_decision_performance(
            strategy, decision_time, self.evaluation_window_hours)

        if action in ("none", "no_action"):
            return self._evaluate_no_action(decision, post_perf)
        elif action == "pause_strategy":
            return self._evaluate_pause(decision, post_perf)
        elif action == "resume_strategy":
            return self._evaluate_resume(decision, post_perf)
        elif action in ("adjust_risk", "adjust_allocation"):
            return self._evaluate_adjustment(decision, post_perf)
        else:
            return {
                "decision": decision,
                "was_correct": True,  # unknown actions get benefit of doubt
                "reason": f"Unknown action type: {action}",
                "pnl_impact": 0.0,
            }

    def _evaluate_no_action(self, decision: dict, post_perf: dict) -> dict:
        """No action is correct if nothing bad happened."""
        pnl = post_perf.get("pnl_after", 0.0)
        dd = post_perf.get("max_dd_after", 0.0)

        # Correct if no significant drawdown occurred
        was_correct = dd < 5.0

        reason = (f"No action taken. Post-decision: pnl={pnl:.2f}, dd={dd:.1f}%. "
                  f"{'Correct — no significant drawdown.' if was_correct else 'Incorrect — drawdown suggests action was needed.'}")

        return {
            "decision": decision,
            "was_correct": was_correct,
            "reason": reason,
            "pnl_impact": pnl,
        }

    def _evaluate_pause(self, decision: dict, post_perf: dict) -> dict:
        """Pause is correct if the market moved against the strategy's position."""
        counterfactual_pnl = post_perf.get("counterfactual_pnl", 0.0)

        # Pause was correct if counterfactual shows losses
        was_correct = counterfactual_pnl < 0

        reason = (f"Strategy paused. Counterfactual PnL if continued: {counterfactual_pnl:.2f}. "
                  f"{'Correct — avoided losses.' if was_correct else 'Incorrect — would have been profitable.'}")

        return {
            "decision": decision,
            "was_correct": was_correct,
            "reason": reason,
            "pnl_impact": -counterfactual_pnl,  # Impact = losses avoided
        }

    def _evaluate_resume(self, decision: dict, post_perf: dict) -> dict:
        """Resume is correct if the strategy made money after resuming."""
        pnl = post_perf.get("pnl_after", 0.0)

        was_correct = pnl >= 0

        reason = (f"Strategy resumed. Post-resume PnL: {pnl:.2f}. "
                  f"{'Correct — profitable after resume.' if was_correct else 'Incorrect — lost money after resume.'}")

        return {
            "decision": decision,
            "was_correct": was_correct,
            "reason": reason,
            "pnl_impact": pnl,
        }

    def _evaluate_adjustment(self, decision: dict, post_perf: dict) -> dict:
        """Risk/allocation adjustment is correct if performance improved."""
        pnl = post_perf.get("pnl_after", 0.0)
        dd = post_perf.get("max_dd_after", 0.0)

        # Correct if drawdown stayed contained and PnL wasn't deeply negative
        was_correct = dd < 5.0 and pnl > -1.0

        reason = (f"Risk/allocation adjusted. Post-adjustment: pnl={pnl:.2f}, dd={dd:.1f}%. "
                  f"{'Correct — risk contained.' if was_correct else 'Incorrect — risk not contained.'}")

        return {
            "decision": decision,
            "was_correct": was_correct,
            "reason": reason,
            "pnl_impact": pnl,
        }

    def _get_post_decision_performance(self, strategy: str,
                                        decision_time: datetime,
                                        window_hours: int) -> dict:
        """Get strategy performance in the window after a decision.

        Returns:
            Dict with pnl_after, max_dd_after, counterfactual_pnl.
        """
        try:
            end = decision_time + timedelta(hours=window_hours)
            fills = self.timescale.query_fills_by_strategy(
                strategy, decision_time, end)

            pnl = 0.0
            max_dd = 0.0

            if fills:
                pnl = sum(f.get("realized_pnl", 0.0) for f in fills)

                # Estimate max drawdown from fills
                cumulative = 0.0
                peak = 0.0
                for f in fills:
                    cumulative += f.get("realized_pnl", 0.0)
                    peak = max(peak, cumulative)
                    dd = peak - cumulative
                    max_dd = max(max_dd, dd)

            # Always compute counterfactual (important for pause decisions
            # where post-decision fills may be empty)
            counterfactual = self._estimate_counterfactual(
                strategy, decision_time, window_hours)

            return {
                "pnl_after": round(pnl, 2),
                "max_dd_after": round(max_dd, 2),
                "counterfactual_pnl": round(counterfactual, 2),
            }
        except Exception:
            logger.debug("Failed to get post-decision performance")
            return {"pnl_after": 0.0, "max_dd_after": 0.0, "counterfactual_pnl": 0.0}

    def _estimate_counterfactual(self, strategy: str,
                                  decision_time: datetime,
                                  window_hours: int) -> float:
        """Estimate counterfactual PnL (what would have happened without the decision).

        Uses pre-decision PnL rate extrapolated to the window.
        """
        try:
            pre_start = decision_time - timedelta(hours=window_hours)
            pre_fills = self.timescale.query_fills_by_strategy(
                strategy, pre_start, decision_time)

            if not pre_fills:
                return 0.0

            pre_pnl = sum(f.get("realized_pnl", 0.0) for f in pre_fills)
            # Extrapolate the pre-decision trend
            return pre_pnl  # Simple: assume similar PnL would have continued

        except Exception:
            return 0.0
