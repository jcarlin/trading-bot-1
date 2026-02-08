"""Hybrid regime classifier combining rule-based and ML approaches.

Uses weighted voting between the existing MarketRegimeClassifier (rule-based)
and the MLRegimeClassifier to produce a more robust regime classification.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class HybridRegimeClassifier:
    """Combines rule-based and ML regime classifiers with weighted voting.

    When both classifiers agree, confidence is boosted.
    When they disagree, confidence is penalized.

    Falls back to rule-only classification if ML is unavailable.
    """

    def __init__(self, rule_classifier, ml_classifier, config: Optional[dict] = None):
        self.rule_classifier = rule_classifier
        self.ml_classifier = ml_classifier
        config = config or {}

        self.rule_weight = config.get("rule_weight", 0.4)
        self.ml_weight = config.get("ml_weight", 0.6)
        self.agreement_boost = config.get("agreement_boost", 0.15)
        self.disagreement_penalty = config.get("disagreement_penalty", 0.20)

    def classify(self, candles) -> dict:
        """Classify regime using both classifiers with weighted voting.

        Returns:
            dict with keys: regime, confidence, rule_regime, ml_regime,
                  agreement, rule_confidence, ml_confidence.
            Falls back to rule-only if ML unavailable.
        """
        # Always get rule-based result
        rule_result = self.rule_classifier.classify(candles)
        rule_regime = rule_result.get("regime", "unknown")
        rule_confidence = rule_result.get("confidence", 0.0)

        # Try ML classification
        ml_result = None
        if self.ml_classifier is not None:
            ml_result = self.ml_classifier.classify(candles)

        # Fallback: ML unavailable
        if ml_result is None:
            return {
                "regime": rule_regime,
                "confidence": rule_confidence,
                "rule_regime": rule_regime,
                "ml_regime": None,
                "agreement": None,
                "rule_confidence": rule_confidence,
                "ml_confidence": 0.0,
            }

        ml_regime = ml_result.get("regime", "unknown")
        ml_confidence = ml_result.get("confidence", 0.0)

        agreement = rule_regime == ml_regime

        if agreement:
            # Both agree: use shared regime, boost confidence
            combined_confidence = (
                self.rule_weight * rule_confidence + self.ml_weight * ml_confidence
            )
            combined_confidence = min(1.0, combined_confidence + self.agreement_boost)
            regime = rule_regime
        else:
            # Disagree: use the higher-weighted classifier's regime, penalize confidence
            if self.ml_weight * ml_confidence >= self.rule_weight * rule_confidence:
                regime = ml_regime
            else:
                regime = rule_regime
            combined_confidence = (
                self.rule_weight * rule_confidence + self.ml_weight * ml_confidence
            )
            combined_confidence = max(0.0, combined_confidence - self.disagreement_penalty)

        return {
            "regime": regime,
            "confidence": round(combined_confidence, 4),
            "rule_regime": rule_regime,
            "ml_regime": ml_regime,
            "agreement": agreement,
            "rule_confidence": round(rule_confidence, 4),
            "ml_confidence": round(ml_confidence, 4),
        }
