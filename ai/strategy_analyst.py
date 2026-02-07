"""Claude-powered strategy analysis for deeper insights."""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

ANALYST_SYSTEM_PROMPT = """You are a quantitative trading strategy analyst.
Given strategy performance metrics, market regime data, and historical context,
provide actionable analysis and specific parameter recommendations.

Be precise with numbers. Reference specific metrics. Identify patterns.
Suggest concrete parameter adjustments (e.g., "increase EMA slow period from 26 to 34").

Respond with ONLY valid JSON:
{
  "assessment": "brief overall assessment",
  "strengths": ["list of strategy strengths"],
  "weaknesses": ["list of strategy weaknesses"],
  "parameter_recommendations": [
    {"parameter": "param_name", "current": "value", "recommended": "value", "reason": "..."}
  ],
  "regime_fit": "how well strategy fits current regime",
  "risk_concerns": ["specific risk issues"],
  "outlook": "forward-looking assessment"
}"""


class AIStrategyAnalyst:
    """Uses Claude to provide deep strategy analysis beyond metrics."""

    def __init__(self, claude_client):
        self.claude = claude_client

    def analyze_strategy(self, strategy_name: str, metrics: dict,
                         health: dict, regime: dict) -> Optional[dict]:
        """Get Claude's analysis of a strategy's current state."""
        if self.claude is None or not self.claude.is_available():
            return None

        context = json.dumps({
            "strategy_name": strategy_name,
            "performance_metrics": metrics,
            "health_score": health,
            "market_regime": regime,
        }, indent=2, default=str)

        return self.claude.complete_json(ANALYST_SYSTEM_PROMPT, context)

    def compare_strategies(self, strategy_metrics: dict) -> Optional[dict]:
        """Get Claude's comparative analysis of multiple strategies."""
        if self.claude is None or not self.claude.is_available():
            return None

        context = json.dumps(strategy_metrics, indent=2, default=str)
        prompt = f"Compare these strategies and rank them:\n\n{context}"

        return self.claude.complete_json(ANALYST_SYSTEM_PROMPT, prompt)
