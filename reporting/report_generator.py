"""Natural language report generation from trading metrics."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates structured text reports from strategy metrics and decisions.

    Uses template-based formatting (f-strings) for weekly and monthly reports.
    """

    def __init__(self, performance_tracker, health_scorer, timescale,
                 strategy_name: str, correlation_analyzer=None,
                 portfolio_tracker=None, ai_report_writer=None):
        self.performance_tracker = performance_tracker
        self.health_scorer = health_scorer
        self.timescale = timescale
        self.strategy_name = strategy_name
        self.correlation_analyzer = correlation_analyzer
        self.portfolio_tracker = portfolio_tracker
        self.ai_report_writer = ai_report_writer

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

        # Append AI analysis if available
        if self.ai_report_writer:
            try:
                ai_analysis = self.ai_report_writer.generate_weekly_report(
                    metrics=metrics, health=health, decisions=decisions,
                    regime={})
                if ai_analysis:
                    report += f"\n\n--- AI ANALYSIS ---\n{ai_analysis}"
            except Exception:
                logger.debug("AI report writer unavailable for weekly report")

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

    def generate_portfolio_report(self, strategy_health_map: Optional[dict] = None,
                                   wallet_highlights: Optional[list] = None) -> str:
        """Generate a portfolio-level report across all strategies.

        Args:
            strategy_health_map: {strategy_name: {health_score, grade}} for ranking table.
            wallet_highlights: List of wallet discovery dicts for intelligence section.

        Returns:
            Formatted report string.
        """
        now = datetime.now(timezone.utc)

        # Portfolio metrics
        portfolio_metrics = {}
        if self.portfolio_tracker:
            try:
                portfolio_metrics = self.portfolio_tracker.compute_portfolio_metrics()
            except Exception:
                logger.debug("Failed to compute portfolio metrics for report")

        total_pnl = portfolio_metrics.get("total_pnl", 0)
        portfolio_sharpe = portfolio_metrics.get("portfolio_sharpe", 0)
        portfolio_dd = portfolio_metrics.get("portfolio_max_dd", 0)
        capital_efficiency = portfolio_metrics.get("capital_efficiency", 0)
        contributions = portfolio_metrics.get("strategy_contributions", {})

        # Strategy ranking
        ranking_lines = []
        if strategy_health_map:
            sorted_strats = sorted(
                strategy_health_map.items(),
                key=lambda x: x[1].get("health_score", 0),
                reverse=True,
            )
            for rank, (sname, shealth) in enumerate(sorted_strats, 1):
                score = shealth.get("health_score", 0)
                grade = shealth.get("grade", "N/A")
                pnl = contributions.get(sname, 0)
                ranking_lines.append(
                    f"  {rank}. {sname}: Score={score:.0f} Grade={grade} PnL={pnl:.4f}")
        else:
            ranking_lines.append("  No strategy health data available.")

        # Correlation matrix
        corr_lines = []
        if self.correlation_analyzer:
            try:
                corr = self.correlation_analyzer.compute_correlation_matrix()
                corr_matrix = corr.get("matrix", [])
                corr_names = corr.get("strategy_names", [])
                if corr_names:
                    header = "  " + " ".join(f"{n[:12]:>12}" for n in corr_names)
                    corr_lines.append(header)
                    for i, name in enumerate(corr_names):
                        row_vals = " ".join(
                            f"{corr_matrix[i][j]:12.4f}" for j in range(len(corr_names)))
                        corr_lines.append(f"  {name[:12]:>12} {row_vals}")
            except Exception:
                corr_lines.append("  Correlation data unavailable.")
        else:
            corr_lines.append("  Correlation analyzer not configured.")

        # Allocation recommendations
        alloc_lines = []
        if strategy_health_map:
            for sname, shealth in strategy_health_map.items():
                score = shealth.get("health_score", 0)
                if score >= 80:
                    alloc_lines.append(f"  - {sname}: Consider increasing allocation (health={score:.0f})")
                elif score < 30:
                    alloc_lines.append(f"  - {sname}: Consider reducing allocation (health={score:.0f})")
        if not alloc_lines:
            alloc_lines.append("  - No allocation changes recommended.")

        # Wallet intelligence
        wallet_lines = []
        if wallet_highlights:
            for w in wallet_highlights[:5]:
                addr = w.get("address", "unknown")
                wscore = w.get("score", 0)
                wallet_lines.append(f"  - {addr[:16]}...: score={wscore:.1f}")
        else:
            wallet_lines.append("  No wallet intelligence data available.")

        report = f"""
{'=' * 60}
PORTFOLIO REPORT
{'=' * 60}
Generated: {now.strftime('%Y-%m-%d %H:%M UTC')}

--- PORTFOLIO METRICS ---
Total PnL: {total_pnl:.4f}
Portfolio Sharpe: {portfolio_sharpe:.4f}
Portfolio Max DD: {portfolio_dd:.2f}%
Capital Efficiency: {capital_efficiency:.6f}

--- STRATEGY RANKING ---
{chr(10).join(ranking_lines)}

--- CORRELATION MATRIX ---
{chr(10).join(corr_lines)}

--- ALLOCATION RECOMMENDATIONS ---
{chr(10).join(alloc_lines)}

--- WALLET INTELLIGENCE ---
{chr(10).join(wallet_lines)}
""".strip()

        return report
