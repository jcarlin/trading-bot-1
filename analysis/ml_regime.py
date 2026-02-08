"""ML-based market regime classification using XGBoost.

Provides a machine learning approach to regime classification that can be
used standalone or combined with the rule-based MarketRegimeClassifier
via the HybridRegimeClassifier.

xgboost and sklearn are optional dependencies. If not installed,
classify() returns None and is_available() returns False.
"""

import logging
import json
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Valid regime labels
REGIMES = ["trending_up", "trending_down", "ranging", "volatile"]
REGIME_TO_INT = {r: i for i, r in enumerate(REGIMES)}
INT_TO_REGIME = {i: r for i, r in enumerate(REGIMES)}

# Feature names in canonical order
FEATURE_NAMES = [
    "sma_ratio_20_50",
    "atr_ratio",
    "atr_percentile",
    "rsi_14",
    "bb_width",
    "volume_ratio",
    "momentum_1h",
    "momentum_4h",
    "momentum_24h",
    "vol_of_vol",
]


class MLRegimeClassifier:
    """XGBoost-based market regime classifier.

    Lazy-imports xgboost and sklearn so they remain optional dependencies.
    When unavailable, classify() returns None.

    Features: SMA ratios, ATR ratio/percentile, RSI, BB width, volume ratio,
              price momentum (1h/4h/24h), volatility of volatility.

    Regimes: trending_up, trending_down, ranging, volatile.
    """

    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        self.model_path = config.get("model_path", None)
        self.feature_window = config.get("feature_window", 100)
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self.retrain_interval_hours = config.get("retrain_interval_hours", 168)

        self._model = None
        self._feature_importance = {}

        # Try to load model if path provided
        if self.model_path:
            try:
                self.load_model(self.model_path)
            except Exception as e:
                logger.warning("Failed to load model from %s: %s", self.model_path, e)

    def is_available(self) -> bool:
        """Check if xgboost and sklearn are installed."""
        try:
            import xgboost  # noqa: F401
            import sklearn  # noqa: F401
            return True
        except ImportError:
            return False

    def classify(self, candles) -> Optional[dict]:
        """Classify regime from OHLCV candles (DataFrame or list of dicts).

        Returns:
            dict with keys: regime, confidence, probabilities, features.
            Returns None if model not available or insufficient data.
        """
        if not self.is_available():
            logger.debug("ML regime classifier unavailable (missing dependencies)")
            return None

        if self._model is None:
            logger.debug("ML regime classifier has no trained model")
            return None

        features = self._compute_features(candles)
        if features is None:
            return None

        feature_array = np.array([[features[name] for name in FEATURE_NAMES]])

        try:
            probabilities = self._model.predict_proba(feature_array)[0]
            predicted_class = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_class])

            regime = INT_TO_REGIME[predicted_class]
            prob_dict = {INT_TO_REGIME[i]: float(p) for i, p in enumerate(probabilities)}

            if confidence < self.confidence_threshold:
                return None

            return {
                "regime": regime,
                "confidence": round(confidence, 4),
                "probabilities": prob_dict,
                "features": {k: round(v, 6) for k, v in features.items()},
            }
        except Exception as e:
            logger.error("ML classify failed: %s", e)
            return None

    def _compute_features(self, candles) -> Optional[dict]:
        """Compute feature vector from candle data.

        Args:
            candles: DataFrame with OHLCV columns or list of dicts.

        Returns:
            dict of feature_name -> value, or None if insufficient data.
        """
        if candles is None:
            return None

        if isinstance(candles, list):
            if len(candles) == 0:
                return None
            df = pd.DataFrame(candles)
        elif isinstance(candles, pd.DataFrame):
            df = candles.copy()
        else:
            return None

        required_cols = {"open", "high", "low", "close", "volume"}
        if not required_cols.issubset(set(df.columns)):
            return None

        min_rows = max(self.feature_window, 50)
        if len(df) < min_rows:
            return None

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float)

        # SMA ratio
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        current_sma_20 = float(sma_20.iloc[-1])
        current_sma_50 = float(sma_50.iloc[-1])
        sma_ratio_20_50 = current_sma_20 / current_sma_50 if current_sma_50 != 0 else 1.0

        # ATR computation
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_7 = true_range.rolling(7).mean()
        atr_21 = true_range.rolling(21).mean()

        current_atr_7 = float(atr_7.iloc[-1])
        current_atr_21 = float(atr_21.iloc[-1])
        atr_ratio = current_atr_7 / current_atr_21 if current_atr_21 != 0 else 1.0

        # ATR percentile
        atr_14 = true_range.rolling(14).mean()
        lookback = min(self.feature_window, len(atr_14.dropna()))
        if lookback > 0:
            recent_atr = atr_14.dropna().iloc[-lookback:]
            current_atr = float(atr_14.iloc[-1])
            atr_percentile = float((recent_atr <= current_atr).sum() / len(recent_atr))
        else:
            atr_percentile = 0.5

        # RSI(14)
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        rsi_14 = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0

        # Bollinger Band width
        bb_sma = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_sma + 2 * bb_std
        bb_lower = bb_sma - 2 * bb_std
        bb_mid = float(bb_sma.iloc[-1])
        bb_width = float((bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_mid) if bb_mid != 0 else 0.0

        # Volume ratio
        avg_volume = volume.rolling(20).mean()
        current_volume = float(volume.iloc[-1])
        avg_vol_val = float(avg_volume.iloc[-1])
        volume_ratio = current_volume / avg_vol_val if avg_vol_val != 0 else 1.0

        # Price momentum
        momentum_1h = float(close.iloc[-1] / close.iloc[-2] - 1) if len(close) >= 2 else 0.0
        momentum_4h = float(close.iloc[-1] / close.iloc[-4] - 1) if len(close) >= 4 else 0.0
        momentum_24h = float(close.iloc[-1] / close.iloc[-24] - 1) if len(close) >= 24 else 0.0

        # Volatility of volatility (std of ATR changes)
        atr_changes = atr_14.diff().dropna()
        vol_of_vol = float(atr_changes.iloc[-20:].std()) if len(atr_changes) >= 20 else 0.0
        if np.isnan(vol_of_vol):
            vol_of_vol = 0.0

        return {
            "sma_ratio_20_50": sma_ratio_20_50,
            "atr_ratio": atr_ratio,
            "atr_percentile": atr_percentile,
            "rsi_14": rsi_14,
            "bb_width": bb_width,
            "volume_ratio": volume_ratio,
            "momentum_1h": momentum_1h,
            "momentum_4h": momentum_4h,
            "momentum_24h": momentum_24h,
            "vol_of_vol": vol_of_vol,
        }

    def train(self, candles, labels=None) -> Optional[dict]:
        """Train XGBoost model on labeled data.

        If no labels provided, auto-label with rule-based classifier.

        Args:
            candles: DataFrame with OHLCV data.
            labels: list of regime labels (same length as candles windowed).

        Returns:
            dict with accuracy, f1_per_class, feature_importance.
            None if deps unavailable.
        """
        if not self.is_available():
            return None

        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score

        if isinstance(candles, list):
            candles = pd.DataFrame(candles)

        if labels is None:
            labels = self._auto_label_regimes(candles)

        if labels is None or len(labels) == 0:
            return None

        # Build feature matrix by sliding window
        min_rows = max(self.feature_window, 50)
        features_list = []
        valid_labels = []

        for i in range(min_rows, len(candles)):
            window = candles.iloc[max(0, i - self.feature_window):i + 1]
            feat = self._compute_features(window)
            if feat is not None and i < len(labels):
                features_list.append([feat[name] for name in FEATURE_NAMES])
                valid_labels.append(REGIME_TO_INT.get(labels[i], 2))

        if len(features_list) < 10:
            return None

        X = np.array(features_list)
        y = np.array(valid_labels)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
            if len(np.unique(y)) > 1 else None,
        )

        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            objective="multi:softprob",
            num_class=len(REGIMES),
            use_label_encoder=False,
            eval_metric="mlogloss",
            verbosity=0,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = float(accuracy_score(y_test, y_pred))

        f1_per_class = {}
        for i, regime in enumerate(REGIMES):
            y_binary_true = (y_test == i).astype(int)
            y_binary_pred = (y_pred == i).astype(int)
            if y_binary_true.sum() > 0:
                f1_per_class[regime] = float(f1_score(y_binary_true, y_binary_pred, zero_division=0))
            else:
                f1_per_class[regime] = 0.0

        importance = model.feature_importances_
        self._feature_importance = {
            FEATURE_NAMES[i]: float(importance[i]) for i in range(len(FEATURE_NAMES))
        }

        self._model = model

        return {
            "accuracy": round(accuracy, 4),
            "f1_per_class": f1_per_class,
            "feature_importance": self._feature_importance,
        }

    def _auto_label_regimes(self, candles: pd.DataFrame) -> Optional[list]:
        """Auto-label regimes using the rule-based classifier for training data."""
        from analysis.market_regime import MarketRegimeClassifier

        rule_classifier = MarketRegimeClassifier()
        labels = []

        min_rows = max(rule_classifier.sma_slow_period + 1, 51)

        for i in range(len(candles)):
            if i < min_rows:
                labels.append("ranging")
                continue
            window = candles.iloc[max(0, i - self.feature_window):i + 1]
            result = rule_classifier.classify(window)
            labels.append(result.get("regime", "ranging"))

        return labels

    def save_model(self, path: str) -> None:
        """Save model and metadata to disk."""
        if self._model is None:
            raise ValueError("No model to save")

        if not self.is_available():
            raise RuntimeError("xgboost not available")

        self._model.save_model(path)

        meta_path = path + ".meta.json"
        meta = {
            "feature_names": FEATURE_NAMES,
            "feature_importance": self._feature_importance,
            "feature_window": self.feature_window,
            "confidence_threshold": self.confidence_threshold,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Saved ML regime model to %s", path)

    def load_model(self, path: str) -> None:
        """Load model and metadata from disk."""
        if not self.is_available():
            raise RuntimeError("xgboost not available")

        import xgboost as xgb

        self._model = xgb.XGBClassifier()
        self._model.load_model(path)

        meta_path = path + ".meta.json"
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            self._feature_importance = meta.get("feature_importance", {})
        except FileNotFoundError:
            logger.warning("No metadata file found at %s", meta_path)

        logger.info("Loaded ML regime model from %s", path)

    def get_feature_importance(self) -> dict:
        """Return feature importance dict from the trained model."""
        return dict(self._feature_importance)
