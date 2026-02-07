"""Claude-powered decision engine for the OODA orchestrator."""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

DECISION_SYSTEM_PROMPT = """You are the AI decision engine for an autonomous crypto trading system.
You receive structured metrics about strategy performance, market regime, and portfolio state.
You must decide whether to take action and what action to take.

Available actions:
- "no_action": No action needed, continue monitoring
- "pause_strategy": Pause a specific strategy (when health critically low or decay too high)
- "resume_strategy": Resume a paused strategy (when conditions improve)
- "adjust_risk": Tighten or loosen risk parameters for a strategy
- "adjust_allocation": Change allocation weight for a strategy
- "promote_strategy": Promote a shadow strategy to live

Decision principles:
- Capital preservation is the prime directive
- Every decision must be bounded and reversible
- Epistemic humility: prefer smaller adjustments over drastic changes
- Log your reasoning chain clearly

Respond with ONLY valid JSON (no markdown, no explanation outside JSON):
{
  "action": "no_action|pause_strategy|resume_strategy|adjust_risk|adjust_allocation|promote_strategy",
  "strategy_name": "name if applicable, otherwise null",
  "reason": "concise explanation",
  "hypothesis": "what you expect to happen",
  "confidence": 0.0-1.0,
  "alternatives": [{"action": "...", "reason": "..."}],
  "reasoning_chain": ["step 1", "step 2", "..."]
}"""


class AIDecisionEngine:
    """Uses Claude to make OODA decisions with full reasoning chains.

    Falls back to None (triggering rule-based fallback) if Claude
    is unavailable or returns low-confidence decisions.
    """

    AVAILABLE_ACTIONS = [
        "no_action", "pause_strategy", "resume_strategy",
        "adjust_risk", "adjust_allocation", "promote_strategy",
    ]

    def __init__(self, claude_client, config: dict = None):
        self.claude = claude_client
        config = config or {}
        self.enabled = config.get("enabled", True)
        self.min_confidence = config.get("min_confidence", 0.6)

    def decide(self, metrics: dict, regime: dict, assessment: dict,
               checkpoint_type: str, active_strategies: list[str]) -> Optional[dict]:
        """Ask Claude to reason about what action to take.

        Returns decision dict or None (triggers rule-based fallback).
        """
        if not self.enabled or self.claude is None:
            return None

        if not self.claude.is_available():
            return None

        context = self._build_context(
            metrics, regime, assessment, checkpoint_type, active_strategies)

        result = self.claude.complete_json(DECISION_SYSTEM_PROMPT, context)

        if result is None:
            logger.info("Claude decision engine unavailable, falling back to rules")
            return None

        return self._parse_response(result)

    def _build_context(self, metrics, regime, assessment,
                       checkpoint_type, active_strategies) -> str:
        """Format all context into a structured message for Claude."""
        context = {
            "checkpoint_type": checkpoint_type,
            "active_strategies": active_strategies,
            "market_regime": regime,
            "regime_assessment": assessment,
            "metrics": self._sanitize_metrics(metrics),
        }
        return json.dumps(context, indent=2, default=str)

    def _parse_response(self, result: dict) -> Optional[dict]:
        """Validate and parse Claude's response."""
        action = result.get("action", "no_action")

        if action == "no_action":
            return None

        if action not in self.AVAILABLE_ACTIONS:
            logger.warning("Claude returned invalid action: %s", action)
            return None

        confidence = result.get("confidence", 0.0)
        if not isinstance(confidence, (int, float)):
            confidence = 0.0

        if confidence < self.min_confidence:
            logger.info("Claude decision confidence %.2f below threshold %.2f",
                        confidence, self.min_confidence)
            return None

        # Build validated decision
        decision = {
            "action": action,
            "strategy_name": result.get("strategy_name"),
            "reason": result.get("reason", "AI decision"),
            "hypothesis": result.get("hypothesis", ""),
            "confidence": confidence,
            "alternatives": result.get("alternatives", []),
            "reasoning_chain": result.get("reasoning_chain", []),
            "source": "ai_engine",
        }
        return decision

    def _sanitize_metrics(self, metrics: dict) -> dict:
        """Remove non-serializable items from metrics."""
        sanitized = {}
        for key, value in metrics.items():
            if key == "report":
                sanitized[key] = str(value)[:2000]
            elif isinstance(value, dict):
                sanitized[key] = {
                    k: v for k, v in value.items()
                    if isinstance(v, (int, float, str, bool, list, dict, type(None)))
                }
            elif isinstance(value, (int, float, str, bool, list, type(None))):
                sanitized[key] = value
        return sanitized
