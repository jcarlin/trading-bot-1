"""OODA decision framework for autonomous strategy management.

Implements the OBSERVE -> ORIENT -> DECIDE -> ACT -> EVALUATE cycle
for automated strategy evaluation and adjustment.
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class OODAOrchestrator:
    """Central decision engine implementing the OODA loop.

    Coordinates all evaluation components and makes autonomous
    decisions about strategy management.
    """

    def __init__(self, strategy_manager, health_scorer, regime_classifier,
                 timescale, redis_store, config: Optional[dict] = None,
                 backtest_comparator=None, signal_assessor=None,
                 execution_tracker=None, report_generator=None,
                 correlation_analyzer=None, portfolio_tracker=None):
        self.strategy_manager = strategy_manager
        self.health_scorer = health_scorer
        self.regime_classifier = regime_classifier
        self.timescale = timescale
        self.redis_store = redis_store
        self.config = config or {}

        # Optional components
        self.backtest_comparator = backtest_comparator
        self.signal_assessor = signal_assessor
        self.execution_tracker = execution_tracker
        self.report_generator = report_generator
        self.correlation_analyzer = correlation_analyzer
        self.portfolio_tracker = portfolio_tracker

        # Config
        self.symbol = self.config.get("symbol", "BTC/USDC")
        self.strategy_name = self.config.get("strategy_name", "funding_rate_arb")
        self.decay_pause_threshold = self.config.get("decay_pause_threshold", 50.0)
        self.health_pause_threshold = self.config.get("health_pause_threshold", 20)

        # Multi-strategy config
        self.correlation_limit = self.config.get("correlation_limit", 0.8)
        self.portfolio_dd_limit = self.config.get("portfolio_dd_limit", 15.0)

        # Strategy-regime suitability mapping
        self.regime_suitability = {
            "momentum": ["trending_up", "trending_down"],
            "mean_reversion": ["ranging"],
            "funding_rate_arb": ["trending_up", "trending_down", "ranging"],
        }

    async def evaluate(self, checkpoint_type: str) -> dict:
        """Execute a full OODA cycle.

        Args:
            checkpoint_type: "hourly", "daily", "weekly", or "monthly"

        Returns:
            dict with evaluation results.
        """
        start_time = time.time()

        try:
            # OBSERVE
            metrics = self._observe(checkpoint_type)

            # ORIENT
            regime, assessment = self._orient(metrics)

            # DECIDE
            decision = self._decide(metrics, regime, assessment, checkpoint_type)

            # ACT
            outcome = None
            if decision:
                outcome = await self._act(decision)

            # EVALUATE
            self._evaluate(decision, outcome, checkpoint_type, time.time() - start_time)

            # Update Prometheus
            self._update_prometheus(metrics)

            return {
                "checkpoint_type": checkpoint_type,
                "decision": decision,
                "outcome": outcome,
                "metrics": metrics,
                "regime": regime,
                "duration_seconds": time.time() - start_time,
            }

        except Exception:
            logger.exception("OODA evaluation failed for %s", checkpoint_type)
            return {
                "checkpoint_type": checkpoint_type,
                "decision": None,
                "outcome": None,
                "metrics": {},
                "regime": {},
                "duration_seconds": time.time() - start_time,
                "error": True,
            }

    def _observe(self, checkpoint_type: str) -> dict:
        """OBSERVE: Gather metrics based on checkpoint level."""
        metrics = {}

        # Always gather health score
        try:
            health = self.health_scorer.compute_health_score(
                window_hours=168 if checkpoint_type != "hourly" else 24)
            metrics["health"] = health
        except Exception:
            logger.exception("Failed to compute health score")
            metrics["health"] = {"health_score": 0, "grade": "N/A"}

        # Signal quality (hourly+)
        if self.signal_assessor:
            try:
                signal_quality = self.signal_assessor.assess(
                    window_hours=24 if checkpoint_type == "hourly" else 168)
                metrics["signal_quality"] = signal_quality
            except Exception:
                logger.debug("Failed to assess signal quality")

        # Execution quality (hourly+)
        if self.execution_tracker:
            try:
                exec_quality = self.execution_tracker.compute(
                    window_hours=24 if checkpoint_type == "hourly" else 168)
                metrics["execution_quality"] = exec_quality
            except Exception:
                logger.debug("Failed to compute execution quality")

        # Backtest comparison (daily+)
        if checkpoint_type in ("daily", "weekly", "monthly") and self.backtest_comparator:
            try:
                comparison = self.backtest_comparator.compare(lookback_hours=168)
                metrics["backtest_comparison"] = comparison
            except Exception:
                logger.debug("Failed to run backtest comparison")

        # Reports (weekly/monthly)
        if checkpoint_type in ("weekly", "monthly") and self.report_generator:
            try:
                if checkpoint_type == "weekly":
                    report = self.report_generator.generate_weekly_report()
                else:
                    report = self.report_generator.generate_monthly_report()
                metrics["report"] = report

                # Store report as system event
                self.timescale.insert_system_event({
                    "time": datetime.now(timezone.utc),
                    "event_type": "report",
                    "severity": "info",
                    "component": "report_generator",
                    "message": f"{checkpoint_type} report generated",
                    "details": {"report_text": report[:5000]},  # truncate
                })
            except Exception:
                logger.debug("Failed to generate report")

        # Multi-strategy: correlation matrix
        if self.correlation_analyzer:
            try:
                corr = self.correlation_analyzer.compute_correlation_matrix()
                metrics["correlation"] = corr
            except Exception:
                logger.debug("Failed to compute correlation matrix")

        # Multi-strategy: portfolio-level metrics
        if self.portfolio_tracker:
            try:
                portfolio = self.portfolio_tracker.compute_portfolio_metrics()
                metrics["portfolio"] = portfolio
            except Exception:
                logger.debug("Failed to compute portfolio metrics")

        # Multi-strategy: per-strategy health scores
        active_strategies = self.strategy_manager.get_active_strategies()
        if len(active_strategies) > 1:
            per_strategy_health = {}
            for strat_name in active_strategies:
                try:
                    h = self.health_scorer.compute_health_score(
                        window_hours=168 if checkpoint_type != "hourly" else 24,
                        strategy_name=strat_name)
                    per_strategy_health[strat_name] = h
                except Exception:
                    try:
                        h = self.health_scorer.compute_health_score(
                            window_hours=168 if checkpoint_type != "hourly" else 24)
                        per_strategy_health[strat_name] = h
                    except Exception:
                        per_strategy_health[strat_name] = {"health_score": 0, "grade": "N/A"}
            metrics["per_strategy_health"] = per_strategy_health

        return metrics

    def _orient(self, metrics: dict) -> tuple:
        """ORIENT: Classify market regime and assess strategy suitability."""
        regime = {"regime": "unknown", "confidence": 0.0}

        try:
            # Get recent candles for regime classification
            end = datetime.now(timezone.utc)
            start = end - timedelta(hours=100)
            candles = self.timescale.query_candles(self.symbol, "1h", start, end)

            if candles:
                df = pd.DataFrame(candles)
                if "time" in df.columns:
                    df.index = pd.to_datetime(df["time"], utc=True)
                regime = self.regime_classifier.classify(df)

                # Store in Redis for real-time access
                self.redis_store.set_market_regime(
                    self.symbol, regime["regime"],
                    regime["confidence"],
                    datetime.now(timezone.utc).isoformat(),
                )

                # Store in TimescaleDB for history
                self.timescale.insert_market_regime(
                    self.symbol, "1h", regime["regime"],
                    regime["confidence"], regime.get("indicators", {}),
                    datetime.now(timezone.utc),
                )
        except Exception:
            logger.exception("Failed to classify market regime")

        # Assess strategy suitability for regime
        assessment = {
            "suitable": True,
            "reason": "No regime-specific concerns",
        }

        # Funding rate arb works best in ranging/low volatility
        if regime.get("regime") == "volatile":
            assessment = {
                "suitable": False,
                "reason": "Funding rate arb may underperform in volatile regime",
            }

        return regime, assessment

    def _decide(self, metrics: dict, regime: dict, assessment: dict,
                checkpoint_type: str) -> Optional[dict]:
        """DECIDE: Choose an action from the decision menu."""
        health = metrics.get("health", {})
        health_score = health.get("health_score", 100)
        max_dd = health.get("raw_metrics", {}).get("max_drawdown", 0)

        # Rule 1: Health score critically low -> pause
        if health_score < self.health_pause_threshold:
            return {
                "action": "pause_strategy",
                "strategy_name": self.strategy_name,
                "reason": f"Health score critically low: {health_score:.1f}",
                "hypothesis": "Pausing will prevent further losses until strategy is reviewed",
                "confidence": 0.9,
                "alternatives": [
                    {"action": "adjust_risk", "reason": "Could tighten risk instead"},
                ],
            }

        # Rule 2: Backtest decay > threshold -> pause
        comparison = metrics.get("backtest_comparison", {})
        decay_pct = comparison.get("decay_pct", 0)
        if decay_pct > self.decay_pause_threshold:
            return {
                "action": "pause_strategy",
                "strategy_name": self.strategy_name,
                "reason": f"Backtest-live decay {decay_pct:.1f}% > {self.decay_pause_threshold}%",
                "hypothesis": "Strategy behavior has diverged from backtest — possible regime change or bug",
                "confidence": 0.8,
                "alternatives": [
                    {"action": "adjust_risk", "reason": "Could reduce position sizes instead"},
                ],
            }

        # Rule 3: Max drawdown exceeds limit -> adjust risk
        if max_dd > 5.0:
            return {
                "action": "adjust_risk",
                "strategy_name": self.strategy_name,
                "reason": f"Max drawdown {max_dd:.2f}% exceeds 5% threshold",
                "hypothesis": "Tightening risk parameters will limit further drawdown",
                "confidence": 0.7,
                "alternatives": [
                    {"action": "pause_strategy", "reason": "Could pause entirely"},
                ],
            }

        # Rule 4: Regime not suitable -> adjust risk
        if not assessment.get("suitable", True):
            return {
                "action": "adjust_risk",
                "strategy_name": self.strategy_name,
                "reason": f"Market regime ({regime.get('regime')}) not suitable: {assessment.get('reason')}",
                "hypothesis": "Reducing exposure in unsuitable regime will protect capital",
                "confidence": 0.6,
                "alternatives": [
                    {"action": "pause_strategy", "reason": "Could pause until regime changes"},
                ],
            }

        # --- Multi-strategy rules (5-7) ---
        active_strategies = self.strategy_manager.get_active_strategies()

        # Rule 5: Correlation too high between any pair
        corr_data = metrics.get("correlation", {})
        corr_matrix = corr_data.get("matrix", [])
        corr_names = corr_data.get("strategy_names", [])
        if len(corr_matrix) >= 2 and len(corr_names) >= 2:
            for i in range(len(corr_names)):
                for j in range(i + 1, len(corr_names)):
                    try:
                        corr_val = float(corr_matrix[i][j])
                    except (IndexError, TypeError, ValueError):
                        continue
                    if corr_val > self.correlation_limit:
                        return {
                            "action": "adjust_allocation",
                            "strategy_name": corr_names[j],
                            "current_allocation": 1.0,
                            "recommended_allocation": 0.5,
                            "reason": (f"Correlation between {corr_names[i]} and "
                                       f"{corr_names[j]} is {corr_val:.2f} > "
                                       f"{self.correlation_limit}"),
                            "hypothesis": "Reducing allocation to correlated strategies diversifies risk",
                            "confidence": 0.7,
                            "alternatives": [
                                {"action": "pause_strategy",
                                 "reason": "Could pause one of the correlated strategies"},
                            ],
                        }

        # Rule 6: Portfolio-level drawdown -> pause worst strategy
        portfolio_data = metrics.get("portfolio", {})
        portfolio_dd = portfolio_data.get("portfolio_max_dd", 0)
        if portfolio_dd > self.portfolio_dd_limit and len(active_strategies) > 1:
            # Find worst-performing strategy by health score
            per_health = metrics.get("per_strategy_health", {})
            worst_name = None
            worst_score = float("inf")
            for sname, shealth in per_health.items():
                s = shealth.get("health_score", 100)
                if s < worst_score:
                    worst_score = s
                    worst_name = sname
            if worst_name:
                return {
                    "action": "pause_strategy",
                    "strategy_name": worst_name,
                    "reason": (f"Portfolio drawdown {portfolio_dd:.1f}% exceeds "
                               f"{self.portfolio_dd_limit}% limit. Pausing worst "
                               f"strategy: {worst_name} (health={worst_score:.0f})"),
                    "hypothesis": "Pausing worst performer will reduce portfolio drawdown",
                    "confidence": 0.8,
                    "alternatives": [
                        {"action": "adjust_risk",
                         "reason": "Could tighten risk on all strategies"},
                    ],
                }

        # Rule 7: Strategy-regime suitability for multi-strategy
        current_regime = regime.get("regime", "unknown")
        if current_regime != "unknown" and len(active_strategies) > 1:
            for strat_name in active_strategies:
                strat_info = self.strategy_manager._strategies.get(strat_name, {})
                strategy_obj = strat_info.get("strategy")
                if strategy_obj is None:
                    continue
                try:
                    category = strategy_obj.get_metadata().get("category", "")
                except Exception:
                    continue
                suitable_regimes = self.regime_suitability.get(category, [])
                if suitable_regimes and current_regime not in suitable_regimes:
                    return {
                        "action": "adjust_risk",
                        "strategy_name": strat_name,
                        "reason": (f"Strategy {strat_name} (category={category}) not "
                                   f"suited for regime '{current_regime}'. "
                                   f"Suited for: {suitable_regimes}"),
                        "hypothesis": "Reducing exposure for mismatched strategy-regime preserves capital",
                        "confidence": 0.6,
                        "alternatives": [
                            {"action": "pause_strategy",
                             "reason": "Could pause until regime changes"},
                        ],
                    }

        # No action needed
        return None

    async def _act(self, decision: dict) -> dict:
        """ACT: Execute the decision via strategy manager."""
        action = decision.get("action")
        strategy_name = decision.get("strategy_name", self.strategy_name)

        try:
            if action == "pause_strategy":
                await self.strategy_manager.pause_strategy(strategy_name)
                return {"executed": True, "action": action, "strategy": strategy_name}

            elif action == "resume_strategy":
                await self.strategy_manager.resume_strategy(strategy_name)
                return {"executed": True, "action": action, "strategy": strategy_name}

            elif action == "adjust_allocation":
                logger.warning("Allocation adjustment recommended for %s: %s",
                               strategy_name, decision.get("reason"))
                return {
                    "executed": True, "action": action,
                    "strategy": strategy_name,
                    "recommended_allocation": decision.get("recommended_allocation"),
                    "note": "logged_recommendation",
                }

            elif action == "adjust_risk":
                # Log the recommendation — actual risk param changes are manual for now
                logger.warning("Risk adjustment recommended: %s", decision.get("reason"))
                return {"executed": True, "action": action, "note": "logged_recommendation"}

            else:
                logger.warning("Unknown action: %s", action)
                return {"executed": False, "reason": f"Unknown action: {action}"}

        except Exception as e:
            logger.exception("Failed to execute action: %s", action)
            return {"executed": False, "error": str(e)}

    def _evaluate(self, decision: Optional[dict], outcome: Optional[dict],
                  checkpoint_type: str, duration: float) -> None:
        """EVALUATE: Log the decision and outcome."""
        try:
            self.timescale.insert_decision({
                "time": datetime.now(timezone.utc),
                "decision_type": f"ooda_{checkpoint_type}",
                "strategy": self.strategy_name,
                "context": {
                    "checkpoint_type": checkpoint_type,
                    "duration_seconds": duration,
                },
                "hypothesis": decision.get("hypothesis", "No action needed") if decision else "No action needed",
                "action": decision or {"action": "none"},
                "alternatives": decision.get("alternatives", []) if decision else [],
                "confidence": decision.get("confidence", 0.0) if decision else 0.0,
                "outcome": outcome or {},
            })
        except Exception:
            logger.exception("Failed to log OODA evaluation")

        # Update Prometheus evaluation duration
        try:
            from monitoring.metrics import evaluation_cycle_duration_seconds
            evaluation_cycle_duration_seconds.labels(
                checkpoint_type=checkpoint_type).observe(duration)
        except Exception:
            pass

    def _update_prometheus(self, metrics: dict) -> None:
        """Update Prometheus gauges with latest metrics."""
        try:
            from monitoring.metrics import (
                strategy_health_score, market_regime_indicator,
                backtest_live_decay_pct, signal_accuracy_pct,
            )

            health = metrics.get("health", {})
            if "health_score" in health:
                strategy_health_score.labels(
                    strategy_name=self.strategy_name
                ).set(health["health_score"])

            comparison = metrics.get("backtest_comparison", {})
            if "decay_pct" in comparison:
                backtest_live_decay_pct.labels(
                    strategy_name=self.strategy_name
                ).set(comparison["decay_pct"])

            sig_quality = metrics.get("signal_quality", {})
            if "signal_accuracy_pct" in sig_quality:
                signal_accuracy_pct.labels(
                    strategy_name=self.strategy_name
                ).set(sig_quality["signal_accuracy_pct"])

        except Exception:
            logger.debug("Failed to update Prometheus metrics")
