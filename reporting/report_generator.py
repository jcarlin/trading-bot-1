"""Natural language report generation from trading metrics."""

import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates structured text reports from strategy metrics and decisions.

    Uses template-based formatting (f-strings) for weekly and monthly reports.
    """

    def __init__(self, performance_tracker, health_scorer, timescale,
                 strategy_name: str):
        self.performance_tracker = performance_tracker
        self.health_scorer = health_scorer
        self.timescale = timescale
        self.strategy_name = strategy_name

    def generate_hourly_report(self) -> str:
        """Generate a brief hourly status report."""
        try:
            metrics = self.performance_tracker.compute_metrics(1)
        except Exception:
            metrics = {}

        try:
            health = self.health_scorer.compute_health_score(24)
        except Exception:
            health = {"health_score": 0, "grade": "N/A"}

        report = f"""
=== HOURLY STATUS REPORT ===
Strategy: {self.strategy_name}
Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

Health Score: {health.get('health_score', 0):.1f} ({health.get('grade', 'N/A')})
Trades (1h): {metrics.get('trade_count', 0)}
PnL (1h): {metrics.get('total_pnl', 0):.4f}
Win Rate: {metrics.get('win_rate', 0):.1f}%
Sharpe: {metrics.get('sharpe_ratio', 0):.4f}
Max DD: {metrics.get('max_drawdown', 0):.2f}%
""".strip()
        return report

    def generate_weekly_report(self) -> str:
        """Generate a comprehensive weekly performance report."""
        try:
            metrics = self.performance_tracker.compute_metrics(168)
        except Exception:
            metrics = {}

        try:
            health = self.health_scorer.compute_health_score(168)
        except Exception:
            health = {"health_score": 0, "grade": "N/A", "components": {}}

        # Fetch decisions for the week
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=168)
        try:
            decisions = self.timescale.query_decisions(
                self.strategy_name, start, end)
        except Exception:
            decisions = []

        signal_decisions = [d for d in decisions
                           if d.get("decision_type") == "signal_generated"]
        state_changes = [d for d in decisions
                        if d.get("decision_type") == "state_change"]

        components = health.get("components", {})

        report = f"""
{'=' * 60}
WEEKLY PERFORMANCE REPORT
{'=' * 60}
Strategy: {self.strategy_name}
Period: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}
Generated: {end.strftime('%Y-%m-%d %H:%M UTC')}

--- PERFORMANCE SUMMARY ---
Total Trades: {metrics.get('trade_count', 0)}
Total PnL: {metrics.get('total_pnl', 0):.4f}
Win Rate: {metrics.get('win_rate', 0):.1f}%
Avg Win: {metrics.get('avg_win', 0):.4f}
Avg Loss: {metrics.get('avg_loss', 0):.4f}
Profit Factor: {metrics.get('profit_factor', 0):.4f}
Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}
Sortino Ratio: {metrics.get('sortino_ratio', 0):.4f}
Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%
Calmar Ratio: {metrics.get('calmar_ratio', 0):.4f}

--- STRATEGY HEALTH ---
Health Score: {health.get('health_score', 0):.1f}/100 (Grade: {health.get('grade', 'N/A')})
Components:
  Sharpe Score: {components.get('sharpe', 0):.1f}/100
  Sortino Score: {components.get('sortino', 0):.1f}/100
  Drawdown Score: {components.get('max_drawdown', 0):.1f}/100
  Win Rate Score: {components.get('win_rate', 0):.1f}/100
  Profit Factor Score: {components.get('profit_factor', 0):.1f}/100
  Calmar Score: {components.get('calmar', 0):.1f}/100

--- RISK EVENTS ---
Max Drawdown This Week: {metrics.get('max_drawdown', 0):.2f}%
Strategy Pauses: {len(state_changes)}

--- NOTABLE DECISIONS ---
Total Signals: {len(signal_decisions)}
State Changes: {len(state_changes)}

--- RECOMMENDATIONS ---
{self._generate_recommendations(metrics, health)}
""".strip()

        return report

    def generate_monthly_report(self) -> str:
        """Generate a monthly overview report."""
        try:
            metrics = self.performance_tracker.compute_metrics(720)
        except Exception:
            metrics = {}

        try:
            health = self.health_scorer.compute_health_score(720)
        except Exception:
            health = {"health_score": 0, "grade": "N/A"}

        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=720)
        try:
            decisions = self.timescale.query_decisions(
                self.strategy_name, start, end)
        except Exception:
            decisions = []

        report = f"""
{'=' * 60}
MONTHLY PERFORMANCE REPORT
{'=' * 60}
Strategy: {self.strategy_name}
Period: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}
Generated: {end.strftime('%Y-%m-%d %H:%M UTC')}

--- MONTHLY OVERVIEW ---
Total Trades: {metrics.get('trade_count', 0)}
Total PnL: {metrics.get('total_pnl', 0):.4f}
Win Rate: {metrics.get('win_rate', 0):.1f}%
Profit Factor: {metrics.get('profit_factor', 0):.4f}
Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}
Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%

--- STRATEGY RANKING ---
Health Score: {health.get('health_score', 0):.1f}/100 (Grade: {health.get('grade', 'N/A')})

--- ATTRIBUTION ANALYSIS ---
Avg Win: {metrics.get('avg_win', 0):.4f}
Avg Loss: {metrics.get('avg_loss', 0):.4f}
Calmar Ratio: {metrics.get('calmar_ratio', 0):.4f}
Sortino Ratio: {metrics.get('sortino_ratio', 0):.4f}

--- DECISION AUDIT ---
Total Decisions Logged: {len(decisions)}
Signal Decisions: {sum(1 for d in decisions if d.get('decision_type') == 'signal_generated')}
State Changes: {sum(1 for d in decisions if d.get('decision_type') == 'state_change')}
Hold Decisions: {sum(1 for d in decisions if d.get('decision_type') == 'hold')}

--- RECOMMENDATIONS ---
{self._generate_recommendations(metrics, health)}
""".strip()

        return report

    def _generate_recommendations(self, metrics: dict, health: dict) -> str:
        """Generate actionable recommendations based on metrics."""
        recommendations = []
        score = health.get("health_score", 0)

        if score < 20:
            recommendations.append(
                "CRITICAL: Health score below 20. Consider pausing strategy.")
        elif score < 40:
            recommendations.append(
                "WARNING: Health score below 40. Review strategy parameters.")
        elif score >= 80:
            recommendations.append(
                "Strategy performing well. Consider increasing allocation.")

        max_dd = metrics.get("max_drawdown", 0)
        if max_dd > 5:
            recommendations.append(
                f"Drawdown at {max_dd:.1f}% exceeds 5% threshold. "
                "Tighten risk controls.")

        win_rate = metrics.get("win_rate", 0)
        if 0 < win_rate < 40:
            recommendations.append(
                f"Win rate at {win_rate:.0f}% is low. "
                "Review entry signal quality.")

        profit_factor = metrics.get("profit_factor", 0)
        if 0 < profit_factor < 1.0:
            recommendations.append(
                "Profit factor below 1.0 — strategy is losing money. "
                "Consider pausing.")

        if not recommendations:
            recommendations.append("No action needed. Continue monitoring.")

        return "\n".join(f"  - {r}" for r in recommendations)
