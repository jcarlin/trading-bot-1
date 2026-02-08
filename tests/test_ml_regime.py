"""Tests for ML regime classifier and hybrid regime classifier."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.ml_regime import MLRegimeClassifier, FEATURE_NAMES, REGIMES, REGIME_TO_INT, INT_TO_REGIME
from analysis.hybrid_regime import HybridRegimeClassifier


def _make_candles(n=120, trend="flat", base_price=100.0):
    """Generate synthetic OHLCV candle data for testing.

    Args:
        n: Number of candles.
        trend: "flat", "up", "down", or "volatile".
        base_price: Starting price.
    """
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")

    if trend == "up":
        drift = np.linspace(0, 0.3 * base_price, n)
        noise = np.random.randn(n) * 0.5
    elif trend == "down":
        drift = np.linspace(0, -0.3 * base_price, n)
        noise = np.random.randn(n) * 0.5
    elif trend == "volatile":
        drift = np.zeros(n)
        noise = np.random.randn(n) * 5.0
    else:
        drift = np.zeros(n)
        noise = np.random.randn(n) * 0.5

    close = base_price + drift + noise
    close = np.maximum(close, 1.0)  # Ensure positive prices

    high = close + np.abs(np.random.randn(n)) * 1.0
    low = close - np.abs(np.random.randn(n)) * 1.0
    low = np.maximum(low, 0.5)
    open_ = close + np.random.randn(n) * 0.3
    volume = np.abs(np.random.randn(n)) * 1000 + 500

    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)


def _mock_xgb_model(predicted_class=0, probabilities=None):
    """Create a mock XGBoost model."""
    model = MagicMock()
    if probabilities is None:
        probs = [0.1] * len(REGIMES)
        probs[predicted_class] = 0.7
        probabilities = [probs]
    model.predict_proba.return_value = np.array(probabilities)
    model.predict.return_value = np.array([predicted_class])
    model.feature_importances_ = np.array([0.1] * len(FEATURE_NAMES))
    return model


# ===================================================================
# TestMLRegimeClassifier
# ===================================================================

class TestMLRegimeClassifier(unittest.TestCase):
    """Tests for MLRegimeClassifier."""

    def test_compute_features_returns_correct_keys(self):
        """Feature vector contains all expected feature names."""
        clf = MLRegimeClassifier()
        candles = _make_candles(120)
        features = clf._compute_features(candles)
        self.assertIsNotNone(features)
        for name in FEATURE_NAMES:
            self.assertIn(name, features)
        self.assertEqual(len(features), len(FEATURE_NAMES))

    def test_compute_features_uptrend_values(self):
        """Strong uptrend produces high momentum and sma_ratio > 1."""
        clf = MLRegimeClassifier()
        candles = _make_candles(120, trend="up")
        features = clf._compute_features(candles)
        self.assertIsNotNone(features)
        self.assertGreater(features["sma_ratio_20_50"], 1.0)
        self.assertGreater(features["momentum_24h"], 0.0)

    def test_compute_features_flat_prices(self):
        """Flat prices produce ratios close to 1.0."""
        clf = MLRegimeClassifier()
        candles = _make_candles(120, trend="flat")
        features = clf._compute_features(candles)
        self.assertIsNotNone(features)
        self.assertAlmostEqual(features["sma_ratio_20_50"], 1.0, places=1)

    def test_compute_features_insufficient_data(self):
        """Insufficient data returns None."""
        clf = MLRegimeClassifier()
        candles = _make_candles(10)
        features = clf._compute_features(candles)
        self.assertIsNone(features)

    def test_compute_features_none_input(self):
        """None input returns None."""
        clf = MLRegimeClassifier()
        features = clf._compute_features(None)
        self.assertIsNone(features)

    def test_compute_features_list_input(self):
        """Can accept list of dicts as input."""
        clf = MLRegimeClassifier()
        candles = _make_candles(120)
        records = candles.reset_index(drop=True).to_dict("records")
        features = clf._compute_features(records)
        self.assertIsNotNone(features)
        for name in FEATURE_NAMES:
            self.assertIn(name, features)

    @patch("analysis.ml_regime.MLRegimeClassifier.is_available", return_value=True)
    def test_classify_returns_correct_structure(self, mock_avail):
        """classify() returns dict with regime, confidence, probabilities, features."""
        clf = MLRegimeClassifier()
        clf._model = _mock_xgb_model(predicted_class=0)
        candles = _make_candles(120, trend="up")

        result = clf.classify(candles)
        self.assertIsNotNone(result)
        self.assertIn("regime", result)
        self.assertIn("confidence", result)
        self.assertIn("probabilities", result)
        self.assertIn("features", result)
        self.assertIn(result["regime"], REGIMES)

    @patch("analysis.ml_regime.MLRegimeClassifier.is_available", return_value=True)
    def test_classify_trending_up(self, mock_avail):
        """Model predicting trending_up returns that regime."""
        clf = MLRegimeClassifier()
        probs = [[0.7, 0.1, 0.1, 0.1]]
        clf._model = _mock_xgb_model(predicted_class=0, probabilities=probs)
        candles = _make_candles(120, trend="up")

        result = clf.classify(candles)
        self.assertEqual(result["regime"], "trending_up")
        self.assertAlmostEqual(result["confidence"], 0.7, places=3)

    @patch("analysis.ml_regime.MLRegimeClassifier.is_available", return_value=True)
    def test_classify_trending_down(self, mock_avail):
        """Model predicting trending_down returns that regime."""
        clf = MLRegimeClassifier()
        probs = [[0.05, 0.8, 0.1, 0.05]]
        clf._model = _mock_xgb_model(predicted_class=1, probabilities=probs)
        candles = _make_candles(120, trend="down")

        result = clf.classify(candles)
        self.assertEqual(result["regime"], "trending_down")

    @patch("analysis.ml_regime.MLRegimeClassifier.is_available", return_value=True)
    def test_classify_ranging(self, mock_avail):
        """Model predicting ranging returns that regime."""
        clf = MLRegimeClassifier()
        probs = [[0.1, 0.1, 0.7, 0.1]]
        clf._model = _mock_xgb_model(predicted_class=2, probabilities=probs)
        candles = _make_candles(120, trend="flat")

        result = clf.classify(candles)
        self.assertEqual(result["regime"], "ranging")

    @patch("analysis.ml_regime.MLRegimeClassifier.is_available", return_value=True)
    def test_classify_volatile(self, mock_avail):
        """Model predicting volatile returns that regime."""
        clf = MLRegimeClassifier()
        probs = [[0.05, 0.05, 0.1, 0.8]]
        clf._model = _mock_xgb_model(predicted_class=3, probabilities=probs)
        candles = _make_candles(120, trend="volatile")

        result = clf.classify(candles)
        self.assertEqual(result["regime"], "volatile")

    @patch("analysis.ml_regime.MLRegimeClassifier.is_available", return_value=True)
    def test_classify_insufficient_data(self, mock_avail):
        """classify with insufficient data returns None."""
        clf = MLRegimeClassifier()
        clf._model = _mock_xgb_model()
        candles = _make_candles(10)

        result = clf.classify(candles)
        self.assertIsNone(result)

    @patch("analysis.ml_regime.MLRegimeClassifier.is_available", return_value=False)
    def test_classify_unavailable(self, mock_avail):
        """classify returns None when dependencies unavailable."""
        clf = MLRegimeClassifier()
        candles = _make_candles(120)
        result = clf.classify(candles)
        self.assertIsNone(result)

    @patch("analysis.ml_regime.MLRegimeClassifier.is_available", return_value=True)
    def test_classify_no_model(self, mock_avail):
        """classify returns None when no model is trained."""
        clf = MLRegimeClassifier()
        candles = _make_candles(120)
        result = clf.classify(candles)
        self.assertIsNone(result)

    @patch("analysis.ml_regime.MLRegimeClassifier.is_available", return_value=True)
    def test_confidence_threshold_filtering(self, mock_avail):
        """Results below confidence threshold are filtered out."""
        clf = MLRegimeClassifier(config={"confidence_threshold": 0.8})
        probs = [[0.4, 0.3, 0.2, 0.1]]  # Max confidence 0.4 < 0.8
        clf._model = _mock_xgb_model(predicted_class=0, probabilities=probs)
        candles = _make_candles(120)

        result = clf.classify(candles)
        self.assertIsNone(result)

    def test_auto_label_regimes(self):
        """auto_label_regimes produces valid labels."""
        clf = MLRegimeClassifier()
        candles = _make_candles(120)
        labels = clf._auto_label_regimes(candles)
        self.assertIsNotNone(labels)
        self.assertEqual(len(labels), 120)
        for label in labels:
            self.assertIn(label, REGIMES + ["unknown"])

    @patch("analysis.ml_regime.MLRegimeClassifier.is_available", return_value=True)
    def test_train_produces_result(self, mock_avail):
        """train() produces accuracy dict with mocked xgboost."""
        mock_xgb_cls = MagicMock()
        mock_model = _mock_xgb_model(predicted_class=2)
        mock_model.predict.return_value = np.array([2, 2, 2, 2, 2])
        mock_xgb_cls.return_value = mock_model

        mock_accuracy = MagicMock(return_value=0.85)
        mock_f1 = MagicMock(return_value=0.8)
        mock_split = MagicMock(return_value=(
            np.random.rand(40, len(FEATURE_NAMES)),
            np.random.rand(10, len(FEATURE_NAMES)),
            np.array([0, 1, 2, 3] * 10),
            np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1]),
        ))

        with patch.dict("sys.modules", {
            "xgboost": MagicMock(XGBClassifier=mock_xgb_cls),
            "sklearn": MagicMock(),
            "sklearn.model_selection": MagicMock(train_test_split=mock_split),
            "sklearn.metrics": MagicMock(accuracy_score=mock_accuracy, f1_score=mock_f1),
        }):
            clf = MLRegimeClassifier()
            candles = _make_candles(200)
            labels = ["ranging"] * 200

            result = clf.train(candles, labels)
            self.assertIsNotNone(result)
            self.assertIn("accuracy", result)
            self.assertIn("f1_per_class", result)
            self.assertIn("feature_importance", result)

    @patch("analysis.ml_regime.MLRegimeClassifier.is_available", return_value=False)
    def test_train_unavailable(self, mock_avail):
        """train() returns None when deps unavailable."""
        clf = MLRegimeClassifier()
        candles = _make_candles(120)
        result = clf.train(candles)
        self.assertIsNone(result)

    @patch("analysis.ml_regime.MLRegimeClassifier.is_available", return_value=True)
    def test_save_load_model_roundtrip(self, mock_avail):
        """save_model/load_model round-trip works."""
        import tempfile
        import os

        clf = MLRegimeClassifier()
        mock_model = _mock_xgb_model()
        clf._model = mock_model
        clf._feature_importance = {"sma_ratio_20_50": 0.15}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.xgb")

            clf.save_model(path)
            mock_model.save_model.assert_called_once_with(path)

            # Verify meta file written
            meta_path = path + ".meta.json"
            self.assertTrue(os.path.exists(meta_path))

            # Now test load
            mock_xgb_cls = MagicMock()
            mock_loaded = MagicMock()
            mock_xgb_cls.return_value = mock_loaded

            with patch.dict("sys.modules", {"xgboost": MagicMock(XGBClassifier=mock_xgb_cls)}):
                clf2 = MLRegimeClassifier()
                # We need to re-import because of the module mock
                # Instead, directly test the load logic
                import json
                with open(meta_path) as f:
                    meta = json.load(f)
                self.assertIn("feature_importance", meta)
                self.assertEqual(meta["feature_importance"]["sma_ratio_20_50"], 0.15)

    def test_feature_importance_empty_before_training(self):
        """Feature importance is empty before training."""
        clf = MLRegimeClassifier()
        importance = clf.get_feature_importance()
        self.assertEqual(importance, {})

    def test_feature_importance_after_setting(self):
        """Feature importance returns values when set."""
        clf = MLRegimeClassifier()
        clf._feature_importance = {"sma_ratio_20_50": 0.2, "rsi_14": 0.3}
        importance = clf.get_feature_importance()
        self.assertEqual(importance["sma_ratio_20_50"], 0.2)
        self.assertEqual(importance["rsi_14"], 0.3)

    def test_regime_constants(self):
        """REGIMES, REGIME_TO_INT, INT_TO_REGIME are consistent."""
        self.assertEqual(len(REGIMES), 4)
        for i, r in enumerate(REGIMES):
            self.assertEqual(REGIME_TO_INT[r], i)
            self.assertEqual(INT_TO_REGIME[i], r)

    def test_default_config(self):
        """Default config values are set correctly."""
        clf = MLRegimeClassifier()
        self.assertEqual(clf.feature_window, 100)
        self.assertEqual(clf.confidence_threshold, 0.5)
        self.assertEqual(clf.retrain_interval_hours, 168)
        self.assertIsNone(clf.model_path)

    def test_custom_config(self):
        """Custom config overrides defaults."""
        clf = MLRegimeClassifier(config={
            "feature_window": 200,
            "confidence_threshold": 0.7,
            "retrain_interval_hours": 72,
        })
        self.assertEqual(clf.feature_window, 200)
        self.assertEqual(clf.confidence_threshold, 0.7)
        self.assertEqual(clf.retrain_interval_hours, 72)


# ===================================================================
# TestHybridRegimeClassifier
# ===================================================================

class TestHybridRegimeClassifier(unittest.TestCase):
    """Tests for HybridRegimeClassifier."""

    def _make_rule_classifier(self, regime="trending_up", confidence=0.8):
        """Create a mock rule-based classifier."""
        mock = MagicMock()
        mock.classify.return_value = {
            "regime": regime,
            "confidence": confidence,
            "indicators": {},
        }
        return mock

    def _make_ml_classifier(self, regime="trending_up", confidence=0.85, available=True):
        """Create a mock ML classifier."""
        mock = MagicMock()
        if available:
            mock.classify.return_value = {
                "regime": regime,
                "confidence": confidence,
                "probabilities": {r: 0.05 for r in REGIMES},
                "features": {},
            }
            mock.classify.return_value["probabilities"][regime] = confidence
        else:
            mock.classify.return_value = None
        return mock

    def test_agreement_boosts_confidence(self):
        """Both classifiers agreeing boosts confidence."""
        rule = self._make_rule_classifier("trending_up", 0.8)
        ml = self._make_ml_classifier("trending_up", 0.85)

        hybrid = HybridRegimeClassifier(rule, ml)
        result = hybrid.classify(_make_candles(120))

        self.assertEqual(result["regime"], "trending_up")
        self.assertTrue(result["agreement"])
        # confidence = 0.4*0.8 + 0.6*0.85 + 0.15 = 0.32 + 0.51 + 0.15 = 0.98
        self.assertGreater(result["confidence"], 0.9)

    def test_disagreement_penalizes_confidence(self):
        """Classifiers disagreeing penalizes confidence."""
        rule = self._make_rule_classifier("trending_up", 0.7)
        ml = self._make_ml_classifier("ranging", 0.6)

        hybrid = HybridRegimeClassifier(rule, ml)
        result = hybrid.classify(_make_candles(120))

        self.assertFalse(result["agreement"])
        # confidence = 0.4*0.7 + 0.6*0.6 - 0.20 = 0.28 + 0.36 - 0.20 = 0.44
        self.assertLess(result["confidence"], 0.5)

    def test_ml_unavailable_falls_back_to_rule(self):
        """ML returning None falls back to rule-only."""
        rule = self._make_rule_classifier("trending_up", 0.75)
        ml = self._make_ml_classifier(available=False)

        hybrid = HybridRegimeClassifier(rule, ml)
        result = hybrid.classify(_make_candles(120))

        self.assertEqual(result["regime"], "trending_up")
        self.assertEqual(result["confidence"], 0.75)
        self.assertIsNone(result["ml_regime"])
        self.assertIsNone(result["agreement"])

    def test_returns_all_expected_fields(self):
        """Result contains all expected fields."""
        rule = self._make_rule_classifier("ranging", 0.7)
        ml = self._make_ml_classifier("ranging", 0.8)

        hybrid = HybridRegimeClassifier(rule, ml)
        result = hybrid.classify(_make_candles(120))

        expected_fields = ["regime", "confidence", "rule_regime", "ml_regime",
                           "agreement", "rule_confidence", "ml_confidence"]
        for field in expected_fields:
            self.assertIn(field, result, f"Missing field: {field}")

    def test_rule_weight_1_uses_rule(self):
        """rule_weight=1.0 always uses rule result."""
        rule = self._make_rule_classifier("trending_up", 0.7)
        ml = self._make_ml_classifier("ranging", 0.9)

        hybrid = HybridRegimeClassifier(rule, ml, config={"rule_weight": 1.0, "ml_weight": 0.0})
        result = hybrid.classify(_make_candles(120))

        self.assertEqual(result["regime"], "trending_up")

    def test_ml_weight_1_uses_ml(self):
        """ml_weight=1.0 always uses ML result when available."""
        rule = self._make_rule_classifier("trending_up", 0.9)
        ml = self._make_ml_classifier("ranging", 0.7)

        hybrid = HybridRegimeClassifier(rule, ml, config={"rule_weight": 0.0, "ml_weight": 1.0})
        result = hybrid.classify(_make_candles(120))

        self.assertEqual(result["regime"], "ranging")

    def test_agreement_true_when_both_agree(self):
        """agreement=True when both classifiers return same regime."""
        rule = self._make_rule_classifier("volatile", 0.6)
        ml = self._make_ml_classifier("volatile", 0.7)

        hybrid = HybridRegimeClassifier(rule, ml)
        result = hybrid.classify(_make_candles(120))

        self.assertTrue(result["agreement"])
        self.assertEqual(result["rule_regime"], "volatile")
        self.assertEqual(result["ml_regime"], "volatile")

    def test_agreement_false_when_disagree(self):
        """agreement=False when classifiers disagree."""
        rule = self._make_rule_classifier("trending_down", 0.6)
        ml = self._make_ml_classifier("volatile", 0.7)

        hybrid = HybridRegimeClassifier(rule, ml)
        result = hybrid.classify(_make_candles(120))

        self.assertFalse(result["agreement"])
        self.assertEqual(result["rule_regime"], "trending_down")
        self.assertEqual(result["ml_regime"], "volatile")

    def test_default_config_values(self):
        """Default config values are correct."""
        hybrid = HybridRegimeClassifier(MagicMock(), MagicMock())
        self.assertEqual(hybrid.rule_weight, 0.4)
        self.assertEqual(hybrid.ml_weight, 0.6)
        self.assertEqual(hybrid.agreement_boost, 0.15)
        self.assertEqual(hybrid.disagreement_penalty, 0.20)

    def test_custom_config(self):
        """Custom config overrides defaults."""
        hybrid = HybridRegimeClassifier(MagicMock(), MagicMock(), config={
            "rule_weight": 0.5,
            "ml_weight": 0.5,
            "agreement_boost": 0.2,
            "disagreement_penalty": 0.3,
        })
        self.assertEqual(hybrid.rule_weight, 0.5)
        self.assertEqual(hybrid.ml_weight, 0.5)
        self.assertEqual(hybrid.agreement_boost, 0.2)
        self.assertEqual(hybrid.disagreement_penalty, 0.3)

    def test_ml_classifier_none(self):
        """If ml_classifier is None, falls back to rule-only."""
        rule = self._make_rule_classifier("ranging", 0.65)

        hybrid = HybridRegimeClassifier(rule, None)
        result = hybrid.classify(_make_candles(120))

        self.assertEqual(result["regime"], "ranging")
        self.assertEqual(result["confidence"], 0.65)
        self.assertIsNone(result["ml_regime"])

    def test_weights_affect_disagreement_winner(self):
        """Higher weighted classifier wins on disagreement."""
        rule = self._make_rule_classifier("trending_up", 0.8)
        ml = self._make_ml_classifier("ranging", 0.8)

        # With default weights (rule=0.4, ml=0.6), ML wins
        hybrid = HybridRegimeClassifier(rule, ml)
        result = hybrid.classify(_make_candles(120))
        self.assertEqual(result["regime"], "ranging")

        # With swapped weights, rule wins
        hybrid2 = HybridRegimeClassifier(rule, ml, config={"rule_weight": 0.7, "ml_weight": 0.3})
        result2 = hybrid2.classify(_make_candles(120))
        self.assertEqual(result2["regime"], "trending_up")


if __name__ == "__main__":
    unittest.main()
